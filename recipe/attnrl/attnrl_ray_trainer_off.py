# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    ResourcePoolManager,
    compute_response_mask,
    apply_kl_penalty,
)
from recipe.attnrl import attnrl_core_algos
from verl.trainer.dynamic_filtering.data_filter import DataFilter
from recipe.attnrl.attnrl_ray_trainer import compute_advantage

WorkerType = Type[Worker]


class RayAttnRLTrainerOff(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        tree_worker=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""
        super().__init__(
            config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls,
            processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name,
            tree_worker
        )

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.tree_worker = tree_worker

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            "attnrl"
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.data_filter = DataFilter(self.train_dataset, use_mc=True)
        self.start_epoch = 0
        self.current_epoch = 0
        self.collate_fn = collate_fn

        self.dataloader_offset = 2
        self.batch_size_ra = attnrl_core_algos.RunningAverage(alpha=self.config.data.get("train_bsz_ema_alpha", 0.0))
        self.real_train_batch_size = self.config.data.train_batch_size
        self.best_steps = [[0, 0], [0, 0], [0, 0]]

    def _save_dataloader(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps + self.dataloader_offset}")

        print(f"local_global_step_folder: {local_global_step_folder}")

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, f"data_{self.global_steps}.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, f"data_{self.global_steps - self.dataloader_offset}.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            for _ in range(10):
                print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
            # raise FileNotFoundError(f"No dataloader state found at {dataloader_local_path}")
            dataloader_local_path = os.path.join(global_step_folder, f"data.pt")
            if os.path.exists(dataloader_local_path):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        _snapshot_step = dataloader_state_dict["_snapshot"]["_snapshot_step"]
        _sampler_iter_yielded = dataloader_state_dict["_snapshot"]["_main_snapshot"]["_sampler_iter_yielded"]
        samples_yielded = dataloader_state_dict["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]["samples_yielded"]
        print(f"Resuming from _snapshot_step: {_snapshot_step}, _sampler_iter_yielded: {_sampler_iter_yielded}, samples_yielded: {samples_yielded}")

        try:
            # load data filter
            local_data_filter = os.path.join(global_step_folder, "data_filter.pt")
            if os.path.exists(local_data_filter):  # TODO (lrz): check
                self.data_filter.load(local_data_filter)
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load the current epoch to the checkpoint
            local_current_epoch = os.path.join(self.config.trainer.default_local_dir, "current_epoch.txt")
            if os.path.exists(local_current_epoch):
                with open(local_current_epoch, "r") as f:
                    self.start_epoch = int(f.read())
                print(f"Loaded current epoch: {self.start_epoch}")
            else:
                print(f"Warning: No current epoch found at {local_current_epoch}, set epoch = 0")
                self.start_epoch = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load self.sampling_num
            local_sampling_num = os.path.join(self.config.trainer.default_local_dir, "sampling_num.txt")
            if os.path.exists(local_sampling_num):
                with open(local_sampling_num, "r") as f:
                    self.sampling_num = int(f.read())
                print(f"Loaded sampling num: {self.sampling_num}")
            else:
                print(f"Warning: No sampling num found at {local_sampling_num}, set sampling num = 0")
                self.sampling_num = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load self.generated_token_num
            local_generated_token_num = os.path.join(self.config.trainer.default_local_dir, "generated_token_num.txt")
            if os.path.exists(local_generated_token_num):
                with open(local_generated_token_num, "r") as f:
                    self.generated_token_num = int(f.read())
                print(f"Loaded generated_token_num: {self.generated_token_num}")
            else:
                print(f"Warning: No generated_token_num found at {local_generated_token_num}, set generated_token_num = 0")
                self.generated_token_num = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load data filter
            local_ra = os.path.join(self.config.trainer.default_local_dir, "running_average.pt")
            if os.path.exists(local_ra):  # TODO (lrz): check
                self.tree_worker.ra.load_ra(local_ra)
                print(f"Loaded tree_worker.ra: {self.tree_worker.ra.mean}")
            else:
                print(f"Warning: No tree_worker.ra found at {local_ra}, set tree_worker.ra = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load data filter
            local_ra = os.path.join(self.config.trainer.default_local_dir, "batch_size_running_average.pt")
            if os.path.exists(local_ra):  # TODO (lrz): check
                self.batch_size_ra.load_ra(local_ra)
                print(f"Loaded batch_size_ra: {self.batch_size_ra.mean}")
            else:
                print(f"Warning: No batch_size_ra found at {local_ra}, set batch_size_ra = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load best_steps
            local_best_steps = os.path.join(self.config.trainer.default_local_dir, "best_steps.pt")
            if os.path.exists(local_best_steps):  # TODO (lrz): check
                obj = torch.load(local_best_steps, weights_only=False)
                self.best_steps = obj
                print(f"Loaded best_steps: {self.best_steps}")
            else:
                print(f"Warning: No best_steps found at {local_best_steps}, set best_steps = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load self.generated_token_num
            local_real_train_batch_size = os.path.join(self.config.trainer.default_local_dir, "real_train_batch_size.txt")
            if os.path.exists(local_real_train_batch_size):
                with open(local_real_train_batch_size, "r") as f:
                    self.real_train_batch_size = int(f.read())
                print(f"Loaded real_train_batch_size: {self.real_train_batch_size}")
            else:
                print(f"Warning: No real_train_batch_size found at {local_real_train_batch_size}, set real_train_batch_size = default")
                # self.real_train_batch_size = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        reverse_global_idx = torch.argsort(global_idx)
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
        return global_idx, reverse_global_idx

    def filter_zero_std_adv(self, grpo_batch, mc_batch):
        batch = grpo_batch.concat([grpo_batch, mc_batch])
        is_correct = torch.tensor(batch.non_tensor_batch["accs"])
        data_indices = batch.non_tensor_batch["index"]
        unique_data_indices, data_indices_indices, unique_data_cnts = np.unique(data_indices, return_index=True, return_counts=True)
        unique_data_indices = unique_data_indices.astype(int)
        avg_accs = np.zeros(len(unique_data_indices), dtype=np.float32)

        repeat_times = np.ones(len(batch), dtype=int)
        for i, data_index in enumerate(unique_data_indices):
            data_index_mask = data_indices == data_index
            index_avg_acc = is_correct[data_index_mask].mean()  # Sum rewards for each sequence
            avg_accs[i] = index_avg_acc.item()

            if avg_accs[i] == 0 or avg_accs[i] == 1:
                repeat_times[data_index_mask] = 0

        batch = batch.sample_level_repeat(repeat_times)
        grpo_idxs = np.where(batch.non_tensor_batch["is_grpo"] == 1)[0]
        grpo_batch = batch.select_idxs(grpo_idxs)
        mc_idxs = np.where(batch.non_tensor_batch["is_grpo"] == 0)[0]
        mc_batch = batch.select_idxs(mc_idxs)

        del batch
        return grpo_batch, mc_batch

    def crop_or_pad_training_batch(self, batch: DataProto):
        new_batch_lst = []
        world_size = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        if len(batch) % world_size != 0:
            target_size = int((len(batch) // world_size + 1) * world_size)
            padding_size = int(target_size - len(batch))
            size_per_gpu = target_size // world_size
            start, end = 0, 0
            for i in range(world_size):
                if i < padding_size:
                    end = start + size_per_gpu - 1
                    temp_batch = batch.select_idxs(list(range(start, end)) + [end - 1])
                    # temp_batch.batch["attention_mask"][-1][self.config.data.max_prompt_length:] = torch.zeros_like(temp_batch.batch["attention_mask"][-1][self.config.data.max_prompt_length:])
                    temp_batch.batch["advantages"][-1] = torch.zeros_like(temp_batch.batch["advantages"][-1])
                else:
                    end = start + size_per_gpu
                    temp_batch = batch.select_idxs(list(range(start, end)))
                try:
                    assert len(temp_batch) == size_per_gpu, f"{len(temp_batch)=}, {size_per_gpu=}"
                except Exception as e:
                    breakpoint()
                    print(e)
                new_batch_lst.append(temp_batch)
                start = end
            try:
                assert end == len(batch), f"{end=}, {len(batch)=}"
            except Exception as e:
                breakpoint()
                print(e)
            batch = DataProto.concat(new_batch_lst)

        return batch

    def custom_collate_fn(self, batch_dict_lst):
        collated = {}
        for key in batch_dict_lst[0].keys():
            try:
                values = [d[key][0] for d in batch_dict_lst]
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    if isinstance(values[0], np.ndarray):
                        values = [v.tolist() for v in values]
                    collated[key] = np.array(values, dtype=object)
            except Exception as e:
                print(f"Collate error: {e}")
                breakpoint()
        return collated

    def _batch_sampler(self):
        if self.config.data.get("dynamic_bsz", False):
            if self.global_steps == 1:
                batch_size = self.config.data.train_batch_size
            else:
                ori_valid_batch_size = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n * 5
                ema_valid_batch_size = self.batch_size_ra.mean.item()
                # batch_size = round(self.config.data.train_batch_size * ori_valid_batch_size // ema_valid_batch_size)  # 64 * ori / ema
                # batch_size = round(self.real_train_batch_size * ori_valid_batch_size // ema_valid_batch_size)  # real * ori / ema
                # batch_size = round(np.clip(batch_size, self.real_train_batch_size * 0.9, self.real_train_batch_size * 1.1))
                target_batch_size = self.real_train_batch_size * ori_valid_batch_size // ema_valid_batch_size  # real * ori / ema
                self.real_train_batch_size = int(np.ceil(self.config.data.dynamic_bsz_ema_alpha * target_batch_size + (1 - self.config.data.dynamic_bsz_ema_alpha) * self.real_train_batch_size))
                batch_size = self.real_train_batch_size
            batch_dict_lst = []
            for _ in range(int(batch_size)):
                try:
                    single_batch_dict = next(self.iterator)
                    batch_dict_lst.append(single_batch_dict)
                except StopIteration:  # End of epoch: increment epoch and reset iterator
                    self.current_epoch += 1
                    self.data_filter.preprocess_epoch(self.current_epoch, self.train_dataset)
                    self.iterator = iter(self.train_dataloader)
                    single_batch_dict = next(self.iterator)
                    batch_dict_lst.append(single_batch_dict)
            batch_dict = self.custom_collate_fn(batch_dict_lst)
        else:
            try:
                batch_dict = next(self.iterator)
            except StopIteration:  # End of epoch: increment epoch and reset iterator
                self.current_epoch += 1
                self.data_filter.preprocess_epoch(self.current_epoch, self.train_dataset)
                self.iterator = iter(self.train_dataloader)
                batch_dict = next(self.iterator)

        return batch_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.sampling_num = 0
        self.generated_token_num = 0

        # load checkpoint before doing anything
        if not self.config.trainer.get("val_only", False):
            self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            if self.global_steps % self.config.trainer.test_freq == 0 or self.config.trainer.get("val_only", False):
                val_metrics = self._validate()
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        first_step_flag = True
        curr_batch = None

        self.current_epoch = self.start_epoch
        self.iterator = iter(self.train_dataloader)

        # for epoch in range(self.start_epoch, self.config.trainer.total_epochs):
        #     self.current_epoch = epoch
        #     self.data_filter.preprocess_epoch(epoch, self.train_dataset)
        #
        #     for batch_dict in self.train_dataloader:

        for _ in range(1):
            while self.current_epoch < self.config.trainer.total_epochs:
                batch_dict = self._batch_sampler()

                metrics = {}
                timing_raw = {}

                with _timer("step", timing_raw):
                    # prepare for next batch
                    if first_step_flag:
                        next_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        if curr_batch is None:
                            curr_batch = deepcopy(next_batch)
                            continue
                    else:
                        with _timer("reward", timing_raw):
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_result = ray.get(future_reward)
                            else:
                                reward_result = compute_reward(curr_batch, self.reward_fn, return_all=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                            curr_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            is_correct = reward_tensor.max(dim=-1).values
                            if "acc" not in reward_extra_infos_dict:
                                curr_batch.non_tensor_batch["acc"] = is_correct.numpy()
                            metrics = self._maybe_log_extra_reward_metrics(reward_result, is_correct.tolist(), metrics)
                            metrics = self.compute_valid_metrics(self.current_epoch, curr_batch, is_correct, metrics, mode="grpo")

                            max_response_length = curr_batch.batch["responses"].shape[-1]
                            response_mask = curr_batch.batch["attention_mask"][:, -max_response_length:]
                            response_length = response_mask.sum(-1).float()
                            response_clip_mask = ~torch.ge(response_length, max_response_length)
                            metrics["batch/clip_overlong"] = len(curr_batch) - response_clip_mask.sum()

                        if self.config.actor_rollout_ref.rollout.get("n_real", False):
                            curr_batch = self.tree_worker.filter_initial_responses(curr_batch)

                        mc_batch = deepcopy(curr_batch)
                        mc_batch.non_tensor_batch["is_grpo"] = np.zeros_like(mc_batch.non_tensor_batch["is_grpo"])
                        last_batch = deepcopy(curr_batch)
                        curr_batch = deepcopy(next_batch)
                        next_batch: DataProto = DataProto.from_single_dict(batch_dict)

                        if self.config.data.get("dynamic_filtering", False):
                            valid_indices_mc = np.isin(mc_batch.non_tensor_batch["index"], self.valid_data_indices)
                            mc_batch = mc_batch.select_idxs(valid_indices_mc)

                        with _timer("build_root_nodes", timing_raw):
                            mc_batch, padded_token_idxs = self.tree_worker.build_root_nodes(mc_batch)
                            metrics["batch/token_idxs"] = mc_batch.batch["token_idxs"].float().mean().item()

                    # pop those keys for generation
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "index"]
                    gen_curr_batch = curr_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)

                    # pass global_steps to trace
                    gen_curr_batch.meta_info["global_steps"] = self.global_steps
                    gen_curr_batch = gen_curr_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_curr_batch.non_tensor_batch["is_grpo"] = np.ones(len(gen_curr_batch))
                    gen_curr_batch_size = len(gen_curr_batch)

                    if not first_step_flag:
                        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "index"]

                        mc_batch.batch["input_ids"] = mc_batch.batch["input_ids"][:, :self.config.actor_rollout_ref.rollout.prompt_length]
                        mc_batch.batch["attention_mask"] = mc_batch.batch["attention_mask"][:, :self.config.actor_rollout_ref.rollout.prompt_length]
                        mc_batch.batch["position_ids"] = mc_batch.batch["position_ids"][:, :self.config.actor_rollout_ref.rollout.prompt_length]
                        mc_gen_batch = mc_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
                        T = 2
                        mc_gen_batch = mc_gen_batch.repeat(repeat_times=T, interleave=True)
                        mc_gen_batch.non_tensor_batch["is_grpo"] = np.zeros(len(mc_gen_batch))

                        gen_curr_batch = gen_curr_batch.concat([gen_curr_batch, mc_gen_batch])

                    gen_curr_batch.meta_info["batch_split_idx"] = gen_curr_batch_size
                    gen_batch_size = len(gen_curr_batch)

                    is_last_step = self.global_steps >= self.total_training_steps

                    # generate a batch
                    with _timer("gen", timing_raw):
                        input_ids = deepcopy(gen_curr_batch.batch["input_ids"])
                        if self.config.actor_rollout_ref.rollout.get("balance_gen_batch", False):
                            print(f"Using balance batch for generation")
                            gen_global_idx, gen_reverse_global_idx = self._balance_gen_batch(gen_curr_batch, mode="gen")
                        gen_curr_batch, padding_size = pad_dataproto_to_divisor(gen_curr_batch, self.actor_rollout_wg.world_size)
                        if self.config.actor_rollout_ref.rollout.get("balance_gen_batch", False):
                            assert padding_size == 0
                        gen_curr_batch_output = self.actor_rollout_wg.generate_sequences_mixed(gen_curr_batch)
                        gen_curr_batch_output = unpad_dataproto(gen_curr_batch_output, padding_size)
                        if self.config.actor_rollout_ref.rollout.get("balance_gen_batch", False):
                            gen_curr_batch_output = self.reorder_and_remove_padding(gen_curr_batch_output, gen_reverse_global_idx)

                        timing_raw.update(gen_curr_batch_output.meta_info["timing"])
                        gen_curr_batch_output.meta_info.pop("timing", None)

                    if not first_step_flag:
                        mc_gen_batch_output = gen_curr_batch_output.select_idxs(list(range(gen_curr_batch_size, gen_batch_size)))
                        self.sampling_num += len(gen_curr_batch_output)
                        self.generated_token_num += np.sum(gen_curr_batch_output.non_tensor_batch["generated_token_num"])
                    elif self.global_steps == 1:
                        self.sampling_num += len(gen_curr_batch_output)
                        self.generated_token_num += np.sum(gen_curr_batch_output.non_tensor_batch["generated_token_num"])
                    metrics["batch/sampling_num"] = self.sampling_num
                    metrics["batch/generated_token_num"] = self.generated_token_num
                    gen_curr_batch_output = gen_curr_batch_output.select_idxs(list(range(gen_curr_batch_size)))

                    curr_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(curr_batch.batch))], dtype=object)  # (bsz, )
                    # repeat to align with repeated responses in rollout
                    curr_batch = curr_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)  # (bsz * n = 64 * 8, ...), batch does not have tensors now
                    curr_batch = curr_batch.union(gen_curr_batch_output)
                    new_input_ids = curr_batch.batch["input_ids"]
                    if not first_step_flag:
                        new_input_ids = torch.cat([new_input_ids, mc_gen_batch_output.batch["input_ids"]], dim=0)
                    self.check_batch_order(DataProto.from_dict({"input_ids": new_input_ids}), {"input_ids": input_ids})
                    # Add sub_uid for each response in the batch
                    response_uid = np.arange(self.config.actor_rollout_ref.rollout.n)
                    sub_uid = np.tile(response_uid, len(curr_batch.non_tensor_batch["uid"]) // self.config.actor_rollout_ref.rollout.n)
                    curr_batch.non_tensor_batch["sub_uid"] = sub_uid  # (bsz * n, )

                    curr_batch.batch["response_mask"] = compute_response_mask(curr_batch)

                    # compute global_valid tokens
                    curr_batch.meta_info["global_token_num"] = torch.sum(curr_batch.batch["attention_mask"], dim=-1).tolist()

                    if not first_step_flag:
                        # repeat to align with repeated responses in rollout
                        mc_batch = mc_batch.repeat(repeat_times=T, interleave=True)  # (bsz * n * N * T = 64 * 8 * 2 * 2, ...)
                        mc_batch.pop(batch_keys=["responses", "response_mask"], non_tensor_batch_keys=["finish_reasons", "generated_token_num"], meta_info_keys=["generated_token_num"])
                        mc_batch = mc_batch.union(mc_gen_batch_output)

                        with _timer("reward_mc", timing_raw):  # ~3 seconds
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward_mc = compute_reward_async.remote(mc_batch, self.config, self.tokenizer)

                    with _timer("reward", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(curr_batch, self.config, self.tokenizer, return_all=True)

                    if self.config.actor_rollout_ref.actor.get("output_attentions", False):
                        with _timer("attention", timing_raw):
                            curr_batch = self.attention_preprocess(curr_batch)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        curr_batch, padding_size = pad_dataproto_to_divisor(curr_batch, self.actor_rollout_wg.world_size)
                        if self.config.trainer.balance_batch:
                            global_idx, reverse_global_idx = self._balance_batch(curr_batch, metrics=metrics)
                        curr_batch.meta_info["calculate_entropy"] = True
                        if self.config.actor_rollout_ref.actor.get("output_attentions", False):
                            curr_batch.meta_info["output_attentions"] = True
                            if self.config.actor_rollout_ref.actor.get("attn_block_size", 0):
                                curr_batch.meta_info["attn_block_size"] = self.config.actor_rollout_ref.actor.attn_block_size
                            if self.config.algorithm.get("process_attn_type", ""):
                                curr_batch.meta_info["process_attn_type"] = self.config.algorithm.process_attn_type
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(curr_batch)  # (bsz * n, r_len), dict_keys(['entropys', 'old_log_probs'])
                        if "output_attentions" in curr_batch.meta_info:
                            curr_batch.meta_info.pop("output_attentions", None)
                        if "process_attn_type" in curr_batch.meta_info:
                            curr_batch.meta_info.pop("process_attn_type", None)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = curr_batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # old_log_prob.batch.pop("entropys")  # TODO (lrz): should we reserve entropys in batch for all settings?
                        curr_batch = curr_batch.union(old_log_prob)
                        if self.config.trainer.balance_batch:  # TODO (lrz): check here
                            curr_batch.reorder(reverse_global_idx)
                        curr_batch = unpad_dataproto(curr_batch, padding_size)

                    if first_step_flag:
                        first_step_flag = False
                        continue

                    # build_root_nodes 需要 reward，所以在这里get reward
                    with _timer("reward_mc", timing_raw):  # ~3 seconds
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor_mc, reward_extra_infos_dict_mc = ray.get(future_reward_mc)
                        else:
                            reward_tensor_mc, reward_extra_infos_dict_mc = compute_reward(batch, self.reward_fn)
                        mc_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict_mc.items()})

                        is_correct_mc = reward_tensor_mc.max(dim=-1).values
                        if "acc" not in reward_extra_infos_dict_mc:
                            mc_batch.non_tensor_batch["acc"] = is_correct_mc.numpy()
                        metrics = self.compute_valid_metrics(self.current_epoch, mc_batch, is_correct_mc, metrics, mode="mc")

                    with _timer("build_trees", timing_raw):  # ~
                        grpo_batch, mc_batch = self.tree_worker.build_trees(mc_batch, last_batch)  # TODO (lrz): should get attention_mask first
                        metrics.update(self.tree_worker.metrics)
                        print(f"{len(grpo_batch)=}, {len(mc_batch)=}, {len(grpo_batch) + len(mc_batch)=}")

                    metrics["batch/original_bs"] = len(grpo_batch) + len(mc_batch)
                    if self.config.algorithm.get("filter_zero_std_adv_to_train", False):
                        grpo_batch, mc_batch = self.filter_zero_std_adv(grpo_batch, mc_batch)
                        print(f"After zero_std_adv filtering: {len(grpo_batch)=}, {len(mc_batch)=}, {len(grpo_batch) + len(mc_batch)=}")

                    # recompute old_log_probs
                    with _timer("old_log_prob_mc", timing_raw):  # ~3 seconds
                        mc_batch, padding_size = pad_dataproto_to_divisor(mc_batch, self.actor_rollout_wg.world_size)
                        mc_batch.meta_info["calculate_entropy"] = False
                        temp_old_log_probs = mc_batch.pop(batch_keys=["old_log_probs"])
                        # TODO (lrz): reorder the batch for ref_log_prob computation
                        if self.config.trainer.balance_batch:
                            global_idx_mc, reverse_global_idx_mc = self._balance_batch(mc_batch, metrics=metrics)

                        old_log_probs = self.actor_rollout_wg.compute_log_prob(mc_batch)  # (bsz * n, r_len), dict_keys(['entropys', 'old_log_probs'])
                        if self.config.trainer.balance_batch:
                            mc_batch.reorder(reverse_global_idx_mc)
                            old_log_probs.reorder(reverse_global_idx_mc)
                        mc_batch = unpad_dataproto(mc_batch, padding_size)
                        temp_old_log_probs = unpad_dataproto(temp_old_log_probs, padding_size)
                        old_log_probs = unpad_dataproto(old_log_probs, padding_size)
                        grpo_token_num = mc_batch.non_tensor_batch["grpo_token_num"]
                        for i in range(len(grpo_token_num)):
                            old_log_probs.batch["old_log_probs"][i, :grpo_token_num[i]] = temp_old_log_probs.batch["old_log_probs"][i, :grpo_token_num[i]]
                        mc_batch = mc_batch.union(old_log_probs)
                        del temp_old_log_probs

                    batch = grpo_batch.concat([grpo_batch, mc_batch])
                    is_correct_all = torch.tensor(batch.non_tensor_batch["accs"])
                    metrics = self.compute_valid_metrics(self.current_epoch, batch, is_correct_all, metrics, mode="all")
                    self.batch_size_ra.add_batch(torch.tensor(len(batch)).float().reshape(1, 1))
                    metrics["batch/filtered_bs"] = len(batch)
                    metrics["batch/real_train_bs"] = self.real_train_batch_size
                    metrics["batch/bs_running_average"] = self.batch_size_ra.mean.item()

                    if self.use_reference_policy:
                        batch, padding_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # (bsz * n, r_len), dict_keys(['ref_log_prob'])
                            batch = batch.union(ref_log_prob)
                        batch = unpad_dataproto(batch, padding_size)

                    with _timer("adv", timing_raw):
                        if "token_level_scores" not in batch.batch:
                            batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.algorithm.get("use_outcome_reward", False):
                            outcome_rewards = torch.tensor(batch.non_tensor_batch["accs"])
                            batch.batch["outcome_rewards"] = torch.where(outcome_rewards > 0, 1, 0).float()

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )  # (bsz * n, r_len), dict_keys([..., 'advantages', 'returns'])

                    valid_train_response = torch.where(masked_sum(batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:], batch.batch["advantages"] != 0, axis=1) > 0, 1, 0)
                    metrics["batch/valid_train_response_num"] = valid_train_response.sum().item()
                    metrics["batch/valid_train_response_ratio"] = valid_train_response.sum().item() / len(batch)
                    valid_train_token_num = masked_sum(batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:].sum(-1), valid_train_response)
                    metrics["batch/valid_train_token_num"] = valid_train_token_num
                    metrics["batch/valid_train_token_ratio"] = valid_train_token_num / batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:].sum().item()

                    batch = self.crop_or_pad_training_batch(batch)
                    metrics["batch/padded_bs"] = len(batch)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        rollout_data_freq = self.config.trainer.get("rollout_data_freq", 1)
                        if self.global_steps <= 2 or self.global_steps % rollout_data_freq == 0:
                            with _timer("dump_rollout_generations", timing_raw):
                                try:
                                    print(last_batch.batch.keys())
                                    scores = last_batch.non_tensor_batch["acc"].tolist()
                                    kwargs = {"token_idxs": padded_token_idxs.tolist()}
                                    if "token_ranges" in last_batch.batch:
                                        kwargs["token_ranges"] = last_batch.batch["token_ranges"].tolist()
                                    if "step_nums" in last_batch.batch:
                                        kwargs["step_nums"] = last_batch.batch["step_nums"].tolist()
                                    if "seq_idxs" in last_batch.batch:
                                        kwargs["seq_idxs"] = last_batch.batch["seq_idxs"].tolist()
                                    if "seq_cnts" in last_batch.batch:
                                        kwargs["seq_cnts"] = last_batch.batch["seq_cnts"].tolist()
                                    if "receivers" in last_batch.batch:
                                        kwargs["receivers"] = last_batch.batch["receivers"].tolist()
                                    if "token_idxs" in last_batch.batch:
                                        kwargs["token_idxs"] = last_batch.batch["token_idxs"].tolist()
                                    if "attn_scores" in last_batch.batch:
                                        kwargs["attn_scores"] = last_batch.batch["attn_scores"].tolist()
                                    if "stop_think_step_idxs" in last_batch.batch:
                                        kwargs["stop_think_step_idxs"] = last_batch.batch["stop_think_step_idxs"].tolist()
                                    if "stop_think_token_idxs" in last_batch.batch:
                                        kwargs["stop_think_token_idxs"] = last_batch.batch["stop_think_token_idxs"].tolist()
                                    self._dump_generations(
                                        inputs=last_batch.batch["prompts"].tolist(),
                                        outputs=last_batch.batch["responses"].tolist(),
                                        scores=scores,
                                        reward_extra_infos_dict=reward_extra_infos_dict,
                                        dump_path=rollout_data_dir,
                                        prompt_length=last_batch.batch["attention_mask"][:, :last_batch.batch["prompts"].shape[-1]].sum(-1).tolist(),
                                        entropy=last_batch.batch["entropys"].tolist(),
                                        response_length=last_batch.batch["response_mask"].sum(-1).tolist(),
                                        **kwargs,
                                    )

                                    accs = batch.non_tensor_batch["accs"].tolist()
                                    avg_accs = batch.non_tensor_batch["avg_accs"].tolist()
                                    kwargs = {"advantages": batch.batch["advantages"].cpu().tolist()}
                                    if "token_ranges" in batch.batch:
                                        kwargs["token_ranges"] = batch.batch["token_ranges"].tolist()
                                    if "step_nums" in batch.batch:
                                        kwargs["step_nums"] = batch.batch["step_nums"].tolist()
                                    if "seq_idxs" in batch.batch:
                                        kwargs["seq_idxs"] = batch.batch["seq_idxs"].tolist()
                                    if "seq_cnts" in batch.batch:
                                        kwargs["seq_cnts"] = batch.batch["seq_cnts"].tolist()
                                    if "receivers" in batch.batch:
                                        kwargs["receivers"] = batch.batch["receivers"].tolist()
                                    if "stop_think_step_idxs" in batch.batch:
                                        kwargs["stop_think_step_idxs"] = batch.batch["stop_think_step_idxs"].tolist()
                                    if "stop_think_token_idxs" in batch.batch:
                                        kwargs["stop_think_token_idxs"] = batch.batch["stop_think_token_idxs"].tolist()
                                    self._dump_generations(
                                        inputs=batch.batch["input_ids"][:, :-batch.batch["responses"].shape[-1]].tolist(),
                                        outputs=batch.batch["responses"].tolist(),
                                        scores=accs,
                                        avg_accs=avg_accs,
                                        reward_extra_infos_dict=None,
                                        dump_path=rollout_data_dir.replace("rollout", "rollout_mc"),
                                        # input_id=batch.batch["input_ids"][:, :-batch.batch["responses"].shape[-1]].tolist(),
                                        prompt_length=batch.batch["attention_mask"][:, :-batch.batch["responses"].shape[-1]].sum(-1).tolist(),
                                        # output_id=batch.batch["responses"].tolist(),
                                        # entropy=batch.batch["entropys"].tolist(),
                                        response_length=batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:].sum(-1).tolist(),
                                        **kwargs,
                                    )

                                    os.makedirs(rollout_data_dir.replace("rollout", "rollout_tree"), exist_ok=True)
                                    filename = os.path.join(rollout_data_dir.replace("rollout", "rollout_tree"), f"{self.global_steps}.pkl")
                                    import pickle
                                    with open(filename, "wb") as f:
                                        pickle.dump(self.tree_worker.paths, f)
                                except Exception as e:
                                    print(e)
                                    breakpoint()

                    # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (self.global_steps + self.dataloader_offset) % self.config.trainer.save_freq == 0:
                        self._save_dataloader()
                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    if is_last_step:
                        self._save_last_step_flag()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": self.current_epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, accs=torch.tensor(batch.non_tensor_batch["accs"])))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    # pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
