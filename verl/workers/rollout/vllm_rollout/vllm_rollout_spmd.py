# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        self.actual_vocab_size = len(tokenizer)

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        use_tqdm = prompts.meta_info.get("use_tqdm", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=use_tqdm,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            # rollout_log_probs = []
            finish_reasons = []
            generated_token_num = 0
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    # curr_log_prob = []
                    # for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                    #     curr_log_prob.append(logprob[response_ids[i]].logprob)
                    # rollout_log_probs.append(curr_log_prob)
                    finish_reason = output.outputs[sample_id].finish_reason
                    finish_reasons.append(finish_reason)
                    generated_token_num += len(response_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            # rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            # rollout_log_probs = rollout_log_probs.to(torch.float32)
            non_tensor_batch["finish_reasons"] = np.array(finish_reasons)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)  # [1, 2, ..., response_length]
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)  # (1, response_length) -> (bs, response_length)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        meta_info = {"generated_token_num": generated_token_num}

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences_mc(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        # idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        idx = prompts.batch["prompts"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        # position_ids = prompts.batch["position_ids"]
        token_idxs = prompts.batch["token_idxs"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            raise RuntimeError("raw_prompt_ids not found in prompts.non_tensor_batch.")

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            try:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
            except Exception as e:
                print(e)
                import time, json
                date = time.strftime("%Y%m%d-%H%M%S")
                with open(f"vllm_error/{date}.json", "w") as f:
                    json.dump(vllm_inputs, f, indent=4)

                raise ValueError

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            # rollout_log_probs = []
            finish_reasons = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    i = len(response)
                    prompt_length = attention_mask[i, :idx.size(1)].sum().item()
                    prefix_length = len(vllm_inputs[i]["prompt_token_ids"])
                    max_length = self.config.response_length - (prefix_length - prompt_length)
                    if len(response_ids) > max_length:
                        response_ids = response_ids[:max_length]
                    response.append(response_ids)
                    # curr_log_prob = []
                    # for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                    #     curr_log_prob.append(logprob[response_ids[i]].logprob)
                    # rollout_log_probs.append(curr_log_prob)
                    finish_reason = output.outputs[sample_id].finish_reason
                    finish_reasons.append(finish_reason)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            # rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            # rollout_log_probs = rollout_log_probs.to(torch.float32)
            non_tensor_batch["finish_reasons"] = np.array(finish_reasons)

            # seq = torch.cat([idx, response], dim=-1)

        # response_length = response.size(1)
        # delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)  # [1, 2, ..., response_length]
        # delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)  # (1, response_length) -> (bs, response_length)
        # if position_ids.dim() == 3:  # qwen2vl mrope
        #     delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # response_position_ids = position_ids[..., -1:] + delta_position_id
        # position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        new_attention_mask_lst = []
        for i in range(batch_size):
            new_attention_mask = torch.cat([attention_mask[i, :idx.size(1)], response_attention_mask[i]])
            if len(new_attention_mask) > self.config.prompt_length + self.config.response_length:
                new_attention_mask = new_attention_mask[:self.config.prompt_length + self.config.response_length]
            new_attention_mask_lst.append(new_attention_mask)
        # attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        attention_mask = torch.stack(new_attention_mask_lst, dim=0)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "response_mask": response_attention_mask,
                # "input_ids": seq,  # here input_ids become the whole sentences
                # "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                # "position_ids": position_ids,
                "token_idxs": token_idxs,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences_mixed(self, prompts: DataProto, **kwargs) -> DataProto:
        max_try_times = 5
        for try_idx in range(max_try_times):
            if try_idx > 0:
                logger.warning(f"Retrying vllm mixed generation, try {try_idx + 1}/{max_try_times}...")

            # rebuild vllm cache engine
            if (
                vllm_version
                in (
                    "0.5.4",
                    "0.6.3",
                )
                and self.config.free_cache_engine
            ):
                self.inference_engine.init_cache_engine()

            idx = prompts.batch["input_ids"]  # (bs, prompt_length)
            # left-padded attention_mask
            attention_mask = prompts.batch["attention_mask"]
            position_ids = prompts.batch["position_ids"]

            # used to construct attention_mask
            eos_token_id = prompts.meta_info["eos_token_id"]
            # batch_split_idx = prompts.meta_info["batch_split_idx"]

            batch_size = idx.size(0)

            non_tensor_batch = prompts.non_tensor_batch
            # if "raw_prompt_ids" not in non_tensor_batch:
            #     non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

            if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                raise RuntimeError("vllm sharding manager is not work properly.")

            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch["raw_prompt_ids"]]

            # ensure the type of `prompt_token_ids` passed to vllm is list[int]
            # https://github.com/volcengine/verl/pull/772
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

            do_sample = prompts.meta_info.get("do_sample", True)
            is_validate = prompts.meta_info.get("validate", False)
            if not do_sample:
                kwargs = {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0,
                    "n": 1,  # if greedy, only 1 response
                }
            elif is_validate:
                # TODO: try **
                kwargs = {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }

            lora_requests = None
            if self.lora_kwargs:
                lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    lora_int_id = lora_int_ids[0]
                    lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

            success = True
            # users can customize different sampling_params at different run
            try:
                with self.update_sampling_params(**kwargs):
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )

                    response = []
                    finish_reasons = []
                    generated_token_num = []
                    for output in outputs:
                        for sample_id in range(len(output.outputs)):
                            response_ids = output.outputs[sample_id].token_ids
                            i = len(response)
                            if non_tensor_batch["is_grpo"][i]:
                                generated_token_num.append(len(response_ids))
                                np_response_ids = np.array(response_ids)
                                oov_idxs = np.where(np_response_ids >= self.actual_vocab_size)[0]
                                if len(oov_idxs) > 0:
                                    success = False
                                    oov_idxs = np.sort(oov_idxs)
                                    response_ids = response_ids[:oov_idxs[0]]
                            else:
                                prompt_length = attention_mask[i, :idx.size(1)].sum().item()
                                prefix_length = len(vllm_inputs[i]["prompt_token_ids"])
                                max_length = self.config.response_length - (prefix_length - prompt_length)
                                if len(response_ids) > max_length:
                                    response_ids = response_ids[:max_length]
                                    generated_token_num.append(max_length)
                                else:
                                    generated_token_num.append(len(response_ids))
                            response.append(response_ids)
                            finish_reason = output.outputs[sample_id].finish_reason
                            finish_reasons.append(finish_reason)

                    response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
                    non_tensor_batch["finish_reasons"] = np.array(finish_reasons)
                    non_tensor_batch["generated_token_num"] = np.array(generated_token_num)

                    seq = torch.cat([idx, response], dim=-1)
                    break

            except Exception as e:
                success = False
                print(e)
                import time, json
                date = time.strftime("%Y%m%d-%H%M%S")
                if os.path.exists(f"/ytech_milm"):
                    os.makedirs(f"/ytech_milm/lrz/verl-0610/vllm_inputs", exist_ok=True)
                    with open(f"/ytech_milm/lrz/verl-0610/vllm_inputs/{date}.json", "w") as f:
                        json.dump(vllm_inputs, f, indent=4)
                else:
                    os.makedirs(f"/mmu_nlp_hdd/liurunze05/verl-0610/vllm_inputs", exist_ok=True)
                    with open(f"/mmu_nlp_hdd/liurunze05/verl-0610/vllm_inputs/{date}.json", "w") as f:
                        json.dump(vllm_inputs, f, indent=4)

                # if not success and try_idx < max_try_times - 1:
                #     continue

        # grpo_results = ...
        # mc_results = ...
        # response = ...

        non_tensor_batch.pop("raw_prompt_ids", None)  # remove raw_prompt_ids to save memory
        # non_tensor_batch.pop("is_grpo", None)  # remove is_grpo to save memory
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)  # [1, 2, ..., response_length]
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)  # (1, response_length) -> (bs, response_length)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        meta_info = {"generated_token_num": generated_token_num}

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
