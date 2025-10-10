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
from verl.trainer.dynamic_filtering.data_filter import DataFilter
from recipe.attnrl import attnrl_core_algos

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            batch=data.batch,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
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

        self.data_filter = DataFilter(self.train_dataset)
        self.start_epoch = 0
        self.current_epoch = 0
        self.collate_fn = collate_fn

        self.candidate_sep_strs = []
        self.candidate_sep_ids = []
        self.candidate_sep_strs2 = []
        self.candidate_sep_ids2 = []
        for token_id in range(tokenizer.vocab_size):
            sep_str = tokenizer.decode(token_id)
            if "\n" in sep_str:
                self.candidate_sep_strs.append(sep_str)
                self.candidate_sep_ids.append(token_id)
                if "\n\n" in sep_str:
                    self.candidate_sep_strs2.append(sep_str)
                    self.candidate_sep_ids2.append(token_id)
        print(f"{self.candidate_sep_strs=}")
        print(f"{self.candidate_sep_ids=}")
        print(f"{self.candidate_sep_strs2=}")
        print(f"{self.candidate_sep_ids2=}")
        self.candidate_sep_ids = torch.tensor(self.candidate_sep_ids)
        self.candidate_sep_ids2 = torch.tensor(self.candidate_sep_ids2)
        self.stop_think_token_id = torch.tensor(tokenizer.encode('</think>')[-1])

        self.batch_size_ra = attnrl_core_algos.RunningAverage(alpha=self.config.data.get("train_bsz_ema_alpha", 0.0))
        self.real_train_batch_size = self.config.data.train_batch_size
        self.best_steps = [[0, 0], [0, 0], [0, 0]]

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.train_dataset.dataframe_ori = deepcopy(self.train_dataset.dataframe)
        print(f"train_dataset[0]: {self.train_dataset[0]}")
        print(f"val_dataset[0]: {self.val_dataset[0]}")

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=1 if self.config.data.get("dynamic_bsz", False) else self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.data.get("dynamic_bsz", False):
            total_training_steps = len(self.train_dataloader) // self.config.data.train_batch_size * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _create_dataloader_new(self, epoch):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_sampler

        filter_mode = self.config.data.get("dynamic_filtering", "")  # TODO (lrz): check
        self.train_dataset, info = self.data_filter.filter_all_dataset(epoch, self.train_dataset, mode=filter_mode)
        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        return info

    def _dump_generations(self, inputs, outputs, scores=None, reward_extra_infos_dict=None, dump_path="", **kwargs):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            # "score": scores,
            "step": [self.global_steps] * n,
        }
        if scores is not None:
            base_data["score"] = scores

        if reward_extra_infos_dict is not None:
            for k, v in reward_extra_infos_dict.items():
                if scores is not None and k != "score":
                    if len(v) == n:
                        base_data[k] = v

        # Add any additional kwargs to the base data
        for k, v in kwargs.items():
            base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                if "input_id" in entry.keys() and "prompt_length" in entry.keys():
                    entry["input_id"] = entry["input_id"][-entry["prompt_length"]:]
                elif "input" in entry.keys() and "prompt_length" in entry.keys() and not isinstance(entry["input"], str):
                    entry["input"] = entry["input"][-entry["prompt_length"]:]
                if "output_id" in entry.keys() and "response_length" in entry.keys():
                    entry["output_id"] = entry["output_id"][:entry["response_length"]]
                elif "output" in entry.keys() and "response_length" in entry.keys() and not isinstance(entry["output"], str):
                    entry["output"] = entry["output"][:entry["response_length"]]
                if "entropy" in entry.keys():
                    if sum(entry["entropy"]) == 0:
                        entry["entropy"] = []
                    elif "response_length" in entry.keys():
                        entry["entropy"] = entry["entropy"][:entry["response_length"]]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _maybe_log_extra_reward_metrics(self, reward_result, acc, metrics: dict):
        if "thinking_tokens_info" in reward_result:
            thinking_tokens_infos_dict = reward_result["thinking_tokens_info"]
            for key_info in list(thinking_tokens_infos_dict.keys()):
                lst = thinking_tokens_infos_dict[key_info]
                assert len(lst) == 0 or len(lst) == len(acc), f"{key_info}: {len(lst)=}, {len(acc)=}"
                for value, score in zip(lst, acc):
                    if score > 0:
                        thinking_tokens_infos_dict["pos_" + key_info].append(value)
                    else:
                        thinking_tokens_infos_dict["neg_" + key_info].append(value)

            for key_info, lst in thinking_tokens_infos_dict.items():
                metrics[key_info] = sum(lst) / len(lst)

        if "repetition_info" in reward_result:
            repetition_infos_dict = reward_result["repetition_info"]
            for key_info in list(repetition_infos_dict.keys()):
                lst = repetition_infos_dict[key_info]
                assert len(lst) == 0 or len(lst) == len(acc), f"{key_info}: {len(lst)=}, {len(acc)=}"
                for value, score in zip(lst, acc):
                    if score > 0:
                        repetition_infos_dict["pos_" + key_info].append(value)
                    else:
                        repetition_infos_dict["neg_" + key_info].append(value)

            for key_info, lst in repetition_infos_dict.items():
                metrics[key_info] = sum(lst) / len(lst)

        return metrics

    def _validate(self):
        data_source_lst = []
        response_ids_lst = []
        response_length_lst = []
        log_probs_lst = []
        entropys_lst = []

        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        thinking_tokens_infos_dict: dict[str, list] = defaultdict(list)
        repetition_infos_dict: dict[str, list] = defaultdict(list)
        response_length_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "data_source" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("data_source")
            if "index" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("index")
            test_gen_batch = test_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)

            val_only = self.config.trainer.get("val_only", None)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "use_tqdm": True if val_only else False,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            # input_ids = deepcopy(test_gen_batch.batch["input_ids"])
            if self.config.actor_rollout_ref.rollout.get("balance_val_batch", False):
                print(f"Using balance batch for validation")
                test_global_idx, test_reverse_global_idx = self._balance_gen_batch(test_gen_batch, mode="val")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)  # unpad
            if self.config.actor_rollout_ref.rollout.get("balance_val_batch", False):
                test_output_gen_batch.reorder(test_reverse_global_idx)
            print("validation generation end")

            # log_probs_lst.extend(torch.zeros_like(test_output_gen_batch_padded.batch["attention_mask"][:, -test_output_gen_batch_padded.batch["responses"].size(-1):]).detach().cpu().numpy().tolist())
            entropys_lst.extend(torch.zeros_like(test_output_gen_batch_padded.batch["attention_mask"][:, -test_output_gen_batch_padded.batch["responses"].size(-1):]).detach().cpu().numpy().tolist())
            response_length_lst.extend(test_output_gen_batch_padded.batch["attention_mask"][:, -test_output_gen_batch_padded.batch["responses"].size(-1):].sum(-1).detach().cpu().numpy().tolist())

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            max_response_length = output_ids.shape[1]
            # print(max_response_length)
            response_ids_lst.extend(output_ids.detach().cpu().numpy().tolist())
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            self.check_batch_order(test_batch, {"input_ids": input_ids})
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            if "thinking_tokens_info" in result:
                for key, lst in result["thinking_tokens_info"].items():
                    thinking_tokens_infos_dict[key].extend(lst)

            if "repetition_info" in result:
                for key, lst in result["repetition_info"].items():
                    repetition_infos_dict[key].extend(lst)

            if "response_length_info" in result:
                for key, lst in result["response_length_info"].items():
                    response_length_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_only:
            import datetime
            curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            val_ckpt = str(self.config.trainer.get("val_ckpt", "0"))
            val_data_dir = val_data_dir.replace("val", "val_results")
            balance_val_batch = self.config.actor_rollout_ref.rollout.get("balance_val_batch", False)
            val_data_dir = os.path.join(val_data_dir, f"{val_ckpt}_{balance_val_batch}_{self.actor_rollout_wg.world_size}_{curr_time}")

        if val_data_dir:
            self._dump_generations(
                # inputs=sample_inputs,
                # outputs=sample_outputs,
                inputs=test_batch.batch["input_ids"][:, :self.config.data.max_prompt_length].tolist(),
                outputs=response_ids_lst,
                scores=sample_scores,
                prompt_length=test_batch.batch["attention_mask"][:, :self.config.data.max_prompt_length].sum(-1).tolist(),
                entropy=entropys_lst,
                response_length=response_length_lst,
                # log_probs=log_probs_lst,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        for key_info in list(thinking_tokens_infos_dict.keys()):
            lst = thinking_tokens_infos_dict[key_info]
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            for value, score in zip(lst, sample_scores):
                if score > 0:
                    thinking_tokens_infos_dict["pos_" + key_info].append(value)
                else:
                    thinking_tokens_infos_dict["neg_" + key_info].append(value)

        for key_info in list(repetition_infos_dict.keys()):
            lst = repetition_infos_dict[key_info]
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            for value, score in zip(lst, sample_scores):
                if score > 0:
                    repetition_infos_dict["pos_" + key_info].append(value)
                else:
                    repetition_infos_dict["neg_" + key_info].append(value)

        for key_info in list(response_length_infos_dict.keys()):
            lst = response_length_infos_dict[key_info]
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            for value, score in zip(lst, sample_scores):
                if score > 0:
                    response_length_infos_dict[key_info + "/pos"].append(value)
                else:
                    response_length_infos_dict[key_info + "/neg"].append(value)

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        all_mean_lst = []
        all_response_length_mean_lst = []
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if var_name == "response_length":
                    continue
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "pass", "maj", "std"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                        if metric_name.startswith("mean") and f"@{n_max}" in metric_name:
                            all_mean_lst.append(metric_val)
                        if "maj@" in metric_name and "std" in metric_name:
                            continue
                    else:
                        metric_sec = "val-aux"
                        continue

                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
            all_response_length_mean_lst.append(data_src2var2metric2val[data_source]["response_length"]["mean"])
            metric_dict[f"val_response_length/{data_source}/mean"] = data_src2var2metric2val[data_source]["response_length"]["mean"]
        metric_dict[f"val_response_length/all/mean"] = sum(all_response_length_mean_lst) / len(all_response_length_mean_lst)

        metric_dict["val-core/all/acc/mean"] = sum(all_mean_lst) / len(all_mean_lst)
        self.best_steps = sorted(self.best_steps, key=lambda x: x[1], reverse=True)
        if metric_dict["val-core/all/acc/mean"] > self.best_steps[-1][1]:
            self.best_steps[-1][0] = self.global_steps
            self.best_steps[-1][1] = metric_dict["val-core/all/acc/mean"]
            print(f"New best steps: {self.best_steps}")
            self.best_steps = sorted(self.best_steps, key=lambda x: x[1], reverse=True)

        for key_info, lst in thinking_tokens_infos_dict.items():
            metric_dict[f"val_{key_info}"] = sum(lst) / len(lst)

        for key_info, lst in repetition_infos_dict.items():
            metric_dict[f"val_{key_info}"] = sum(lst) / len(lst)

        for key_info, lst in response_length_infos_dict.items():
            metric_dict[f"val_{key_info}/max"] = max(lst)
            metric_dict[f"val_{key_info}/mean"] = sum(lst) / len(lst)
            metric_dict[f"val_{key_info}/min"] = min(lst)
            metric_dict[f"val_{key_info}/clip_ratio"] = sum([_ == max_response_length for _ in lst]) / len(lst)

        if val_data_dir:
            with open(os.path.join(val_data_dir, f"metrics_{self.global_steps}.json"), "w", encoding="utf-8") as f:
                json.dump(metric_dict, f, ensure_ascii=False, indent=4)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        should_save_hf_model = False
        if self.best_steps is not None and len(self.best_steps) > 0 and self.global_steps == self.best_steps[0][0] or self.global_steps == self.best_steps[1][0] or self.global_steps == self.best_steps[2][0]:
            should_save_hf_model = True
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep, should_save_hf_model=should_save_hf_model)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        try:
            # save data filter self.data_filter
            local_data_filter = os.path.join(local_global_step_folder, "data_filter.pt")
            if self.data_filter is not None:
                self.data_filter.save(local_data_filter)
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save the current epoch to the checkpoint
            local_current_epoch = os.path.join(self.config.trainer.default_local_dir, "current_epoch.txt")
            with open(local_current_epoch, "w") as f:
                f.write(str(self.current_epoch))
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.sampling_num
            local_sampling_num = os.path.join(self.config.trainer.default_local_dir, "sampling_num.txt")
            with open(local_sampling_num, "w") as f:
                f.write(str(self.sampling_num))
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.generated_token_num
            local_generated_token_num = os.path.join(self.config.trainer.default_local_dir, "generated_token_num.txt")
            with open(local_generated_token_num, "w") as f:
                f.write(str(self.generated_token_num))
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.generated_token_num
            local_ra = os.path.join(self.config.trainer.default_local_dir, "running_average.pt")
            if self.tree_worker is not None:
                self.tree_worker.ra.save_ra(local_ra)
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.generated_token_num
            local_ra = os.path.join(self.config.trainer.default_local_dir, "batch_size_running_average.pt")
            if self.batch_size_ra is not None:
                self.batch_size_ra.save_ra(local_ra)
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.generated_token_num
            local_best_steps = os.path.join(self.config.trainer.default_local_dir, "best_steps.pt")
            self.best_steps = sorted(self.best_steps, key=lambda x: x[1], reverse=True)
            torch.save(self.best_steps, local_best_steps)
        except Exception as e:
            print(f"Saving extra components error: {e}")

        try:
            # save self.generated_token_num
            local_real_train_batch_size = os.path.join(self.config.trainer.default_local_dir, "real_train_batch_size.txt")
            with open(local_real_train_batch_size, "w") as f:
                f.write(str(self.real_train_batch_size))
        except Exception as e:
            print(f"Saving extra components error: {e}")

    def _save_last_step_flag(self):
        try:
            # save self.generated_token_num
            local_is_last_step = os.path.join(self.config.trainer.default_local_dir, "is_last_step.txt")
            with open(local_is_last_step, "w") as f:
                f.write(str(self.global_steps))
        except Exception as e:
            print(f"Saving is_last_step error: {e}")

    def _load_one(self, path):
        pass

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
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

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

    def _balance_gen_batch(self, batch: DataProto, mode):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        world_size = self.actor_rollout_wg.world_size
        # 初始化每张卡：当前负载（样本数）和索引列表
        gpu_loads = [0] * world_size
        global_partition_lst = [[] for _ in range(world_size)]

        curr = 0
        n_repeats = 0
        q_idx = 0
        if mode == "val":
            assert "data_source" in batch.non_tensor_batch.keys(), f"data_source key must be provided for test set"
            # 先把所有“题目组”拆成 (start_idx, size) 形式
            group_ranges: list[tuple[int, int, int]] = []
            last_key = f'{batch.non_tensor_batch["data_source"][0]}__{batch.non_tensor_batch["index"][0]}'
            for i in range(len(batch)):
                assert batch.non_tensor_batch["data_source"][i], f'data_source: {batch.non_tensor_batch["data_source"][i]} must be provided for test set'
                key = f'{batch.non_tensor_batch["data_source"][i]}__{batch.non_tensor_batch["index"][i]}'
                if key != last_key:
                    group_ranges.append((curr, n_repeats, q_idx))
                    curr += n_repeats
                    last_key = key
                    n_repeats = 0
                    q_idx += 1
                n_repeats += 1
            group_ranges.append((curr, n_repeats, q_idx))
            # print(f"Testing curr={curr}")

            # 按组大小降序排列（把重复次数多的题先分配）
            group_ranges.sort(key=lambda x: (-x[1], x[2]), reverse=False)
            # print(f"Testing group_ranges={group_ranges}")

            # 贪心装箱：对每个题目组，分配到当前负载最小的那张卡
            for start_idx, size, q_idx in group_ranges:
                tgt = min(range(world_size), key=lambda i: gpu_loads[i])  # 找到最空闲的 GPU
                global_partition_lst[tgt].extend(range(start_idx, start_idx + size))  # 分配 [start_idx, start_idx+size) 这段连续样本
                gpu_loads[tgt] += size

            global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
            batch.reorder(global_idx)
            reverse_global_idx = torch.argsort(global_idx)

            return global_idx, reverse_global_idx

        elif mode == "gen":
            assert "index" in batch.non_tensor_batch.keys() and "is_grpo" in batch.non_tensor_batch.keys(), f"index and is_grpo key must be provided for gen set"
            group_ranges: list[tuple[int, int, int]] = []
            last_key = f'{batch.non_tensor_batch["is_grpo"][0]}__{batch.non_tensor_batch["index"][0]}'
            for i in range(len(batch)):
                key = f'{batch.non_tensor_batch["is_grpo"][i]}__{batch.non_tensor_batch["index"][i]}'
                if key != last_key:
                    group_ranges.append((curr, int(batch.non_tensor_batch["is_grpo"][i - 1]), n_repeats))
                    curr += n_repeats
                    last_key = key
                    n_repeats = 0
                    q_idx += 1
                n_repeats += 1
            group_ranges.append((curr, int(batch.non_tensor_batch["is_grpo"][-1]), n_repeats))

            # 按组大小降序排列（把重复次数多的题先分配）
            group_ranges.sort(key=lambda x: (-x[1], -x[2]), reverse=False)
            # print(f"Gen group_ranges={group_ranges}")

            # 贪心装箱：对每个题目组，分配到当前负载最小的那张卡
            for start_idx, is_grpo, size in group_ranges:
                tgt = min(range(world_size), key=lambda i: gpu_loads[i])  # 找到最空闲的 GPU
                global_partition_lst[tgt].extend(range(start_idx, start_idx + size))  # 分配 [start_idx, start_idx+size) 这段连续样本
                gpu_loads[tgt] += size
            max_size = max([load for load in gpu_loads])
            padding_sizes = [max_size - load for load in gpu_loads]
            # print(f"Gen gpu_loads before padding: {gpu_loads}, padding_sizes: {padding_sizes}")

            is_padding = []
            for i in range(world_size):
                is_padding.extend([0] * len(global_partition_lst[i]))
                if padding_sizes[i] > 0:
                    global_partition_lst[i].extend([global_partition_lst[i][-1]] * padding_sizes[i])
                    is_padding.extend([1] * padding_sizes[i])

            global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
            batch.reorder(global_idx)
            batch.non_tensor_batch["is_padding"] = np.array(is_padding)
            reverse_global_idx = torch.argsort(global_idx)

            return global_idx, reverse_global_idx

        else:
            raise NotImplementedError(f"mode {mode} not supported in _balance_gen_batch")

    def reorder_and_remove_padding(self, gen_curr_batch_output: DataProto, reverse_global_idx):
        gen_curr_batch_output.reorder(reverse_global_idx)
        gen_curr_batch_output = gen_curr_batch_output[gen_curr_batch_output.non_tensor_batch["is_padding"] == 0]

        return gen_curr_batch_output

    def check_batch_order(self, batch: DataProto, dict_to_check: dict):
        try:
            for key, tensor in dict_to_check.items():
                if key == "input_ids":
                    assert torch.all(torch.eq(batch.batch[key][:, :self.config.data.max_prompt_length], tensor)), f"Batch order changed! {key} not equal!"
                else:
                    assert torch.all(torch.eq(batch.batch[key], tensor)), f"Batch order changed! {key} not equal!"
        except Exception as e:
            for key, tensor in dict_to_check.items():
                if key == "input_ids":
                    for i in range(batch.batch[key].shape[0]):
                        if not torch.all(torch.eq(batch.batch[key][i, :self.config.data.max_prompt_length], tensor[i])):
                            print(f"{i}: b: {batch.batch[key][i, :self.config.data.max_prompt_length][-20:-16]}, t: {tensor[i][-20:-16]}")
                else:
                    raise NotImplementedError(f"Batch order changed! {key} not equal!!")

            print(f"Batch order changed! {key}: {e}")
            breakpoint()

    def compute_valid_metrics(self, epoch, batch: DataProto, is_correct, metrics: dict, mode="grpo"):
        data_indices = batch.non_tensor_batch["index"]
        # data_sources = batch.non_tensor_batch["data_source"].tolist()
        unique_data_indices, data_indices_indices, unique_data_cnts = np.unique(data_indices, return_index=True, return_counts=True)
        unique_data_indices = unique_data_indices.astype(int)
        avg_accs = np.zeros(len(unique_data_indices), dtype=np.float32)
        # unique_data_sources = data_sources[data_indices_indices].tolist()

        if mode == "grpo":
            self.valid_data_indices = []
        elif mode == "all":
            self.index2acc = defaultdict(float)
        avg_acc, cnt = 0.0, 0
        for i, data_index in enumerate(unique_data_indices):
            data_index_mask = data_indices == data_index
            index_avg_acc = is_correct[data_index_mask].mean()  # Sum rewards for each sequence
            avg_accs[i] = index_avg_acc.item()
            avg_acc += is_correct[data_index_mask].sum().item()
            cnt += len(is_correct[data_index_mask])
            if mode == "grpo":
                filter_mode = self.config.data.get("dynamic_filtering", "")
                if filter_mode == "all_correct" and avg_accs[i] < 1:
                    self.valid_data_indices.append(data_index)
                elif 0 < avg_accs[i] < 1:
                    self.valid_data_indices.append(data_index)
            elif mode == "all":
                self.index2acc[data_index] = avg_accs[i]

        if mode == "grpo":
            suffix = ""
            self.data_filter.add_reward_batch(epoch, unique_data_indices, avg_accs, unique_data_cnts)
            self.data_indices = data_indices
            self.unique_data_indices = unique_data_indices
            self.data_indices_indices = data_indices_indices
            self.unique_data_cnts = unique_data_cnts
            self.is_correct = is_correct
            self.avg_accs = avg_accs
            self.valid_data_indices = np.array(self.valid_data_indices, dtype=np.int32)
        elif mode == "mc":
            self.data_filter.add_reward_batch_mc(epoch, unique_data_indices, avg_accs, unique_data_cnts)
            suffix = "_mc"
        elif mode == "all":
            suffix = "_all"

        solve_none = (avg_accs == 0).sum()
        solve_all = (avg_accs == 1).sum()
        valid = len(unique_data_indices) - solve_none - solve_all
        metrics[f"batch/solve_none{suffix}"] = solve_none
        metrics[f"batch/solve_all{suffix}"] = solve_all
        metrics[f"batch/valid{suffix}"] = valid
        metrics[f"batch/solve_none_ratio{suffix}"] = solve_none / len(unique_data_indices)
        metrics[f"batch/solve_all_ratio{suffix}"] = solve_all / len(unique_data_indices)
        metrics[f"batch/valid_ratio{suffix}"] = valid / len(unique_data_indices)
        metrics[f"batch/avg_acc{suffix}"] = avg_acc / cnt
        if mode == "all":
            metrics[f"batch/filtered_ratio{suffix}"] = (self.real_train_batch_size - len(unique_data_indices)) / self.real_train_batch_size
        print(f"valid prompt [{suffix}]:", len(unique_data_indices) - solve_all - solve_none, solve_none, solve_all)

        return metrics

    def attention_preprocess(self, curr_batch: DataProto):
        all_token_ranges = []
        stop_think_step_idxs = []
        stop_think_token_idxs = []
        step_nums = []

        max_token_range_length = 0
        for i in range(len(curr_batch.batch["responses"])):
            response_ids = curr_batch.batch["responses"][i]
            response_length = response_ids.shape[-1]
            attention_mask = curr_batch.batch["attention_mask"][i]
            valid_prompt_length = attention_mask[:-response_length].sum()
            valid_response_length = attention_mask[-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Find stop_think tokens and candidate separator tokens
            if "qwen2.5" in self.config.actor_rollout_ref.model.path.lower() and "7b" in self.config.actor_rollout_ref.model.path.lower():
                stop_think_mask = torch.zeros(len(valid_response_ids), dtype=torch.bool)
            else:
                stop_think_mask = valid_response_ids == self.stop_think_token_id
            sep_mask = torch.isin(valid_response_ids, self.candidate_sep_ids2)
            if torch.nonzero(sep_mask).view(-1).numel() == 0:
                sep_mask = torch.isin(valid_response_ids, self.candidate_sep_ids)
            if torch.nonzero(sep_mask).view(-1).numel() == 0:
                sep_mask = torch.ones(len(valid_response_ids), dtype=torch.bool)

            # Get indices of the tokens
            stop_think_indices = torch.nonzero(stop_think_mask).view(-1)
            sep_indices = torch.nonzero(sep_mask).view(-1) + 1 + valid_prompt_length
            token_ranges = torch.cat([torch.tensor([0, valid_prompt_length]), sep_indices], dim=0)

            # Calculate token ranges based on candidate separators
            if sep_indices[-1].item() != valid_prompt_length + valid_response_length:
                token_ranges = torch.cat([token_ranges, torch.tensor([valid_prompt_length + valid_response_length])], dim=0)

            # If there's at least one stop_think token, update indices
            if stop_think_indices.numel() > 0:
                stop_think_token_idx = stop_think_indices[-1].item()
                stop_think_step_idx = (token_ranges > (stop_think_token_idx + valid_prompt_length)).nonzero()[0].item() - 1
            else:
                stop_think_step_idx, stop_think_token_idx = -1, -1

            token_ranges = torch.cat([token_ranges[:-1].unsqueeze(-1), token_ranges[1:].unsqueeze(-1)], dim=-1).tolist()
            all_token_ranges.append(token_ranges)
            step_nums.append(len(token_ranges) - 1)
            max_token_range_length = max(max_token_range_length, len(token_ranges))
            stop_think_step_idxs.append(stop_think_step_idx)
            stop_think_token_idxs.append(stop_think_token_idx)

        padded_token_ranges = [sub_list + [(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id)] * (max_token_range_length - len(sub_list)) for sub_list in all_token_ranges]
        padded_token_ranges = torch.tensor(padded_token_ranges)
        curr_batch.batch["token_ranges"] = padded_token_ranges
        curr_batch.batch["step_nums"] = torch.tensor(step_nums)
        curr_batch.batch["stop_think_step_idxs"] = torch.tensor(stop_think_step_idxs)
        curr_batch.batch["stop_think_token_idxs"] = torch.tensor(stop_think_token_idxs)

        return curr_batch

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

        for epoch in range(self.start_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch
            if epoch > 0 and self.config.data.get("dynamic_filtering", False):
                info = self._create_dataloader_new(epoch)
            self.data_filter.preprocess_epoch(epoch, self.train_dataset)

            for batch_dict in self.train_dataloader:
                metrics = {}
                if epoch > 0 and self.config.data.get("dynamic_filtering", False):
                    metrics.update(info)
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # if self.config.data.get("dynamic_filtering", False):
                #     if self.config.data["dynamic_filtering"] == "all_easy_and_hard":
                #         batch = self.data_filter.filter_all_easy_and_hard(batch, self.config.data.filter_easy_n)
                #     batch = self.data_filter.filter_examples_linear_backoff(epoch, batch, self.config.data.filter_easy_n)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # Add sub_uid for each response in the batch
                    response_uid = np.arange(self.config.actor_rollout_ref.rollout.n)
                    sub_uid = np.tile(response_uid, len(batch.non_tensor_batch["uid"]) // self.config.actor_rollout_ref.rollout.n)
                    batch.non_tensor_batch["sub_uid"] = sub_uid  # (bsz * n, )

                    # generation accumulation
                    self.sampling_num += len(batch)
                    self.generated_token_num += gen_batch_output.meta_info.pop("generated_token_num", None)
                    metrics["batch/sampling_num"] = self.sampling_num
                    metrics["batch/generated_token_num"] = self.generated_token_num

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer, return_all=True)

                    if self.config.actor_rollout_ref.actor.get("output_attentions", False):
                        with _timer("attention", timing_raw):
                            batch = self.attention_preprocess(batch)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        # TODO: Decouple the DP balancing and mini-batching.
                        if self.config.trainer.balance_batch:
                            global_idx, reverse_global_idx = self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["calculate_entropy"] = True
                        if self.config.actor_rollout_ref.actor.get("output_attentions", False):
                            batch.meta_info["output_attentions"] = True
                            if self.config.actor_rollout_ref.actor.get("attn_block_size", 0):
                                batch.meta_info["attn_block_size"] = self.config.actor_rollout_ref.actor.attn_block_size
                            if self.config.algorithm.get("process_attn_type", ""):
                                batch.meta_info["process_attn_type"] = self.config.algorithm.process_attn_type
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        if "output_attentions" in batch.meta_info:
                            batch.meta_info.pop("output_attentions", None)
                        if "process_attn_type" in batch.meta_info:
                            batch.meta_info.pop("process_attn_type", None)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        if not self.config.algorithm.get("use_entropy_advantage", False):
                            old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("reward", timing_raw):
                        if self.config.trainer.balance_batch:  # TODO (lrz): check here
                            batch.reorder(reverse_global_idx)

                        if self.config.reward_model.launch_reward_fn_async:
                            reward_result = ray.get(future_reward)
                        else:
                            reward_result = compute_reward(batch, self.reward_fn, return_all=True)
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_infos_dict = reward_result["reward_extra_info"]
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        is_correct = reward_tensor.max(dim=-1).values
                        if "acc" not in reward_extra_infos_dict:
                            batch.non_tensor_batch["acc"] = is_correct.numpy()
                        metrics = self._maybe_log_extra_reward_metrics(reward_result, is_correct.tolist(), metrics)
                        metrics = self.compute_valid_metrics(epoch, batch, is_correct, metrics, mode="grpo")

                        max_response_length = batch.batch["responses"].shape[-1]
                        response_mask = batch.batch["attention_mask"][:, -max_response_length:]
                        response_length = response_mask.sum(-1).float()
                        response_clip_mask = ~torch.ge(response_length, max_response_length)
                        metrics["batch/clip_overlong"] = len(batch) - response_clip_mask.sum()

                    with _timer("adv", timing_raw):
                        if self.config.algorithm.get("use_process_reward", False):
                            reward_max = torch.max(reward_tensor, dim=-1).values
                            process_rewards = torch.zeros_like(reward_tensor)
                            # batch.batch["attn_scores"][torch.isnan(batch.batch["attn_scores"])] = 0.0
                            if batch.batch["token_ranges"][i, 0, 0].item() == 0:  # TODO (lrz): check
                                batch.batch["token_ranges"] = batch.batch["token_ranges"][:, 1:]
                            for i in range(len(reward_tensor)):
                                if reward_max[i] > 0:
                                    if "attn_scores" in batch.batch.keys():
                                        attn_scores = batch.batch["attn_scores"][i, 1:]
                                        step_num = batch.batch["step_nums"][i]
                                        valid_length = (attn_scores >= 0).sum().item()
                                        try:
                                            assert valid_length == step_num
                                        except Exception as e:
                                            breakpoint()
                                        attn_scores = attn_scores[:step_num]
                                        # k = int(len(attn_scores) * 0.2)
                                        k = len(attn_scores)
                                        if len(attn_scores) >= k:
                                            topk_res = torch.topk(attn_scores, k=k, largest=True, sorted=True)
                                            seq_idxs, seq_cnts = topk_res.indices, topk_res.values
                                            if "divide_max" in self.config.algorithm.reward_postprocess:
                                                seq_cnts = seq_cnts / seq_cnts.max()
                                            elif "min_max" in self.config.algorithm.reward_postprocess:
                                                # seq_cnts = (seq_cnts - seq_cnts.min()) / (seq_cnts.max() - seq_cnts.min())
                                                nonzero_cnts = [cnt.item() for cnt in seq_cnts if cnt > 0]
                                                min_val = min(nonzero_cnts) if nonzero_cnts else 0.0
                                                for j in range(len(seq_idxs)):
                                                    if seq_cnts[j] > 0:
                                                        seq_cnts[j] = (seq_cnts[j] - min_val) / (seq_cnts.max() - min_val)
                                            elif "temperature" in self.config.algorithm.reward_postprocess:
                                                if seq_cnts.max().item() > 0:
                                                    seq_cnts = seq_cnts / seq_cnts.max()
                                                    seq_cnts = torch.softmax(seq_cnts, dim=0)
                                            if "reciprocal" in self.config.algorithm.reward_postprocess:
                                                nonzero_cnts = [cnt.item() for cnt in seq_cnts if cnt > 0]
                                                min_val = min(nonzero_cnts) if nonzero_cnts else 0.0
                                                for j in range(len(seq_idxs)):
                                                    if seq_cnts[j] > 0:
                                                        seq_cnts[j] = min_val / seq_cnts[j]
                                            token_range = batch.batch["token_ranges"][i]
                                            token_range = token_range - token_range[0, 0]  # make sure the first token is 0
                                            if not torch.isnan(seq_cnts).any():
                                                for j in range(len(seq_idxs)):
                                                    start_idx, end_idx = token_range[seq_idxs[j]]
                                                    process_rewards[i, start_idx:end_idx] = seq_cnts[j]

                            # try:
                            #     idxs = torch.where(reward_max <= 0)[0]
                            #     assert batch.batch["attn_scores"][idxs].sum() == 0
                            # except Exception as e:
                            #     breakpoint()

                            process_rewards = torch.clamp(process_rewards, max=5.0)
                            batch.batch["process_rewards"] = process_rewards
                            metrics.update({
                                'process_reward/mean': process_rewards.mean().item(),
                                'process_reward/max': process_rewards.max().item(),
                                'process_reward/min': process_rewards.min().item(),
                            })

                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

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
                        )

                    valid_train_response = torch.where(masked_sum(batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:], batch.batch["advantages"] != 0, axis=1) > 0, 1, 0)
                    metrics["batch/valid_train_response_num"] = valid_train_response.sum().item()
                    metrics["batch/valid_train_response_ratio"] = valid_train_response.sum().item() / len(batch)
                    valid_train_token_num = masked_sum(batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:].sum(-1), valid_train_response)
                    metrics["batch/valid_train_token_num"] = valid_train_token_num
                    metrics["batch/valid_train_token_ratio"] = valid_train_token_num / batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:].sum().item()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

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
                                print(batch.batch.keys())
                                scores = batch.non_tensor_batch["acc"].tolist()
                                kwargs = {"advantages": batch.batch["advantages"].cpu().tolist()}
                                if "token_ranges" in batch.batch:
                                    kwargs["token_ranges"] = batch.batch["token_ranges"].tolist()
                                if "attn_scores" in batch.batch:
                                    kwargs["attn_scores"] = batch.batch["attn_scores"].tolist()
                                if "step_nums" in batch.batch:
                                    kwargs["step_nums"] = batch.batch["step_nums"].tolist()
                                if "process_rewards" in batch.batch:
                                    kwargs["process_rewards"] = batch.batch["process_rewards"].cpu().tolist()
                                if "stop_think_step_idxs" in batch.batch:
                                    kwargs["stop_think_step_idxs"] = batch.batch["stop_think_step_idxs"].tolist()
                                if "stop_think_token_idxs" in batch.batch:
                                    kwargs["stop_think_token_idxs"] = batch.batch["stop_think_token_idxs"].tolist()
                                self._dump_generations(
                                    inputs=batch.batch["prompts"].tolist(),
                                    outputs=batch.batch["responses"].tolist(),
                                    scores=scores,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                    input_id=batch.batch["prompts"].tolist(),
                                    prompt_length=batch.batch["attention_mask"][:, :batch.batch["prompts"].shape[-1]].sum(-1).tolist(),
                                    output_id=batch.batch["responses"].tolist(),
                                    entropy=entropys.tolist(),
                                    response_length=batch.batch["response_mask"].sum(-1).tolist(),
                                    **kwargs,
                                )

                    # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    if is_last_step:
                        self._save_last_step_flag()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
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
