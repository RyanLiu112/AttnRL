from __future__ import annotations

import math
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import random
from collections import deque, defaultdict
import torch
import numpy as np
from copy import deepcopy
from scipy.stats import gaussian_kde

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


def compute_process_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """

    with torch.no_grad():
        token_level_rewards = token_level_rewards * response_mask

    return token_level_rewards, token_level_rewards


class MCTSNode(BaseModel):
    token_ids: List[int]
    old_log_probs: Optional[list[float]] = None
    is_grpo: bool = False
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    global_adv: float = -999
    local_adv: float = 0
    reward: float = 0
    depth: int = 0
    main_chain: bool = False
    terminal: bool = False
    terminal_in_subtree: int = 0
    correct_terminal_in_subtree: int = 0
    selected_terminal_in_subtree: int = 0
    accumulated_value: float = 0
    state_value: float = 0
    finish_reason: Optional[str] = None
    tree_idx: Optional[int] = None
    node_idx: Optional[int] = None
    uid: Optional[str] = None
    sub_uid: Optional[int] = None
    batch_idx: Optional[int] = None
    id: str = ""
    text: str = ""
    adv: float = 0
    entropys: Optional[list[float]] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        return setattr(self, key, value)

    def __repr__(self):
        return (
            f"MCTSNode(len={len(self.token_ids)}, reward={self.reward}, accumulated_value={self.accumulated_value},"
            f"state_value={self.state_value:.3f}, adv={self.adv:.3f},"
            f"global_adv={self.global_adv:.3f}, local_adv={self.local_adv:.3f})"
        )


class TreeNode:
    def __init__(
        self,
        tree_idx: int,
        node_idx: int,
        token_ids: List[int],
        old_log_probs: List[float],
        entropys: List[float],
        finish_reason: Optional[str] = None,
        parent_node: Optional['TreeNode'] = None,
        parent_node_idx: Optional[int] = None,
        parent_node_split_idx: Optional[int] = None,
        child_nodes: Optional[List['TreeNode']] = None,
        child_split_indices: Optional[List[int]] = None,
        reward: Optional[float] = None,
        uid: Optional[str] = None,
        sub_uid: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ):
        self.tree_idx: int = tree_idx
        self.node_idx: int = node_idx

        self.token_ids: List[int] = token_ids

        self.old_log_probs: List[float] = old_log_probs
        self.entropys: List[float] = entropys
        self.token_num: int = len(token_ids)
        self.finish_reason: Optional[str] = finish_reason

        self.parent_node = parent_node
        self.parent_node_idx = parent_node_idx
        self.parent_node_split_idx = parent_node_split_idx

        self.child_nodes: List['TreeNode'] = child_nodes if child_nodes else []
        self.child_split_indices: List[int] = child_split_indices if child_split_indices else []

        self.child_correct_num: List[int] = []
        self.child_total_num: List[int] = []

        self.aggregate_token_ids: List[int] = []
        if parent_node is not None:
            self.aggregate_token_ids = parent_node.aggregate_token_ids + parent_node.token_ids[:parent_node_split_idx]

        self.reward: Optional[float] = reward

        self.uid: Optional[str] = uid
        self.sub_uid: Optional[int] = sub_uid
        self.batch_idx: Optional[int] = batch_idx

    def add_child(self, child_node: 'TreeNode', split_index: int) -> None:
        self.child_nodes.append(child_node)
        self.child_split_indices.append(split_index)
        child_node.parent_node = self
        child_node.parent_split_index = split_index


def build_into_tree_format(tree_lists, add_entropys=False) -> MCTSNode:
    all_leaves = []
    def convert_to_json(node: MCTSNode):
        json_data = {
            "token_ids": node.token_ids,
            "old_log_probs": node.old_log_probs,
            "global_adv": node.global_adv,
            "reward": node.reward,
            "depth": node.depth,
            "main_chain": node.main_chain,
            "terminal": node.terminal,
            "terminal_in_subtree": node.terminal_in_subtree,
            "correct_terminal_in_subtree": node.correct_terminal_in_subtree,
            "selected_terminal_in_subtree": node.selected_terminal_in_subtree,
            "accumulated_value": node.accumulated_value,
            "finish_reason": node.finish_reason,
            "tree_idx": node.tree_idx,
            "node_idx": node.node_idx,
            "uid": node.uid,
            "sub_uid": node.sub_uid,
            "batch_idx": node.batch_idx,
        }
        if node.children:
            json_data["children"] = [convert_to_json(child) for child in node.children]
        return json_data

    def build_tree_node(tree_node: TreeNode, parent_mcts_node: Optional[MCTSNode] = None, is_grpo=False) -> MCTSNode:
        tree_node.child_nodes.sort(key=lambda x: x.parent_node_split_idx)
        child_split_indices = [child.parent_node_split_idx for child in tree_node.child_nodes]

        is_terminal = False
        reward = 0
        main_chain = False
        if not child_split_indices:
            first_child_split_idx = len(tree_node.token_ids)
            is_terminal = True
            reward = tree_node.reward
            if tree_node.reward == 1:
                main_chain = True
        else:
            first_child_split_idx = child_split_indices[0]

        root_node = MCTSNode(
            token_ids=tree_node.token_ids[:first_child_split_idx],
            old_log_probs=tree_node.old_log_probs[:first_child_split_idx],
            is_grpo=is_grpo,
            parent=parent_mcts_node,
            depth=(parent_mcts_node.depth + 1) if parent_mcts_node else 0,
            terminal=is_terminal,
            reward=reward,
            main_chain=main_chain,
            finish_reason=tree_node.finish_reason,
            tree_idx=tree_node.tree_idx,
            node_idx=tree_node.node_idx,
            uid=tree_node.uid,
            sub_uid=tree_node.sub_uid,
            batch_idx=tree_node.batch_idx,
        )

        if root_node.terminal:
            all_leaves.append(root_node)
        if add_entropys:
            root_node.entropys = tree_node.entropys[:first_child_split_idx]

        def add_segments_and_children(current_mcts_node: MCTSNode, start_idx: int):
            i = 0
            while i < len(tree_node.child_nodes):
                child_nodes_group = []
                current_split_idx = child_split_indices[i]

                while i < len(tree_node.child_nodes) and child_split_indices[i] == current_split_idx:
                    child_nodes_group.append(tree_node.child_nodes[i])
                    i += 1
                is_terminal = False
                reward = 0
                main_chain = False
                if i < len(tree_node.child_nodes):
                    next_split_idx = child_split_indices[i]
                else:
                    next_split_idx = len(tree_node.token_ids)
                    is_terminal = True
                    reward = tree_node.reward
                    if tree_node.reward == 1:
                        main_chain = True

                segment_node = MCTSNode(
                    token_ids=tree_node.token_ids[start_idx:next_split_idx],
                    old_log_probs=tree_node.old_log_probs[start_idx:next_split_idx],
                    is_grpo=True,
                    parent=current_mcts_node,
                    depth=current_mcts_node.depth + 1,
                    terminal=is_terminal,
                    reward=reward,
                    main_chain=main_chain,
                    finish_reason=tree_node.finish_reason,
                    tree_idx=tree_node.tree_idx,
                    node_idx=tree_node.node_idx,
                    uid=tree_node.uid,
                    sub_uid=tree_node.sub_uid,
                    batch_idx=tree_node.batch_idx,
                )
                if add_entropys:
                    segment_node.entropys = tree_node.entropys[start_idx:next_split_idx]
                current_mcts_node.children.append(segment_node)
                if segment_node.terminal:
                    all_leaves.append(segment_node)
                for child_node in child_nodes_group:
                    child_mcts_node = build_tree_node(child_node, current_mcts_node, is_grpo=False)
                    current_mcts_node.children.append(child_mcts_node)

                start_idx = next_split_idx
                current_mcts_node = segment_node

        add_segments_and_children(root_node, first_child_split_idx)

        return root_node

    root = MCTSNode(token_ids=[])

    for i, tree_list in enumerate(tree_lists):
        if len(tree_list) > 0:
            root.children.append(build_tree_node(tree_list[0], root, is_grpo=True))

    return root, all_leaves


def process_leaf(
    root,
    all_leaves,
):
    leaf_normalize(all_leaves, root)
    selected_terminals = all_leaves

    for leaf in selected_terminals:
        selected_backpropagate(leaf)

    return root, selected_terminals


def leaf_normalize(nodes, root):
    leaf_correctness = [leaf.reward for leaf in nodes]
    sum_correctness = sum(leaf_correctness)
    num = len(leaf_correctness) - 1
    assert num > 0, "entropy num_traces == 0"

    mean = [(sum_correctness - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
    root.state_value = sum(leaf_correctness) / len(leaf_correctness)
    for i, leaf in enumerate(nodes):
        leaf.global_adv = leaf.reward - root.state_value
        leaf.accumulated_value = leaf.reward
        leaf_backpropagate(leaf)


def leaf_backpropagate(node: MCTSNode):
    if node.terminal and node.main_chain:
        node.terminal_in_subtree += 1
        node.correct_terminal_in_subtree += 1
        parent = node.parent
        while parent:
            parent.terminal_in_subtree += 1
            parent.correct_terminal_in_subtree += 1
            parent.accumulated_value += node.accumulated_value
            parent = parent.parent
    elif node.terminal:
        node.terminal_in_subtree += 1
        parent = node.parent
        while parent:
            parent.terminal_in_subtree += 1
            parent.accumulated_value += node.accumulated_value
            parent = parent.parent


def selected_backpropagate(node: MCTSNode):
    node.selected_terminal_in_subtree += 1
    parent = node.parent
    while parent:  # 所有父节点的terminal_in_subtree都加1
        parent.selected_terminal_in_subtree += 1
        parent = parent.parent


def path_from_root_to_node(root, node: MCTSNode):
    dict_path, node_path = [], []

    while node.parent is not None:
        parent_value = node.parent.accumulated_value / node.parent.terminal_in_subtree
        child_value = node.accumulated_value / node.terminal_in_subtree
        local_adv = child_value - parent_value
        if node.global_adv == -999:
            node.global_adv = child_value - root.state_value
        adv = (node.global_adv + local_adv) / math.sqrt(node.terminal_in_subtree)
        if node.terminal:
            assert node.terminal_in_subtree == 1, f"terminal_in_subtree is not 1, {node.terminal_in_subtree}"
        dict_path.append({
            "token_ids": node.token_ids, "old_log_probs": node.old_log_probs, "is_grpo": node.is_grpo,
            "reward": node.reward, "state_value": child_value,
            "global_adv": node.global_adv, "local_adv": local_adv, "adv": adv,
            "pass_ratio": node.correct_terminal_in_subtree / node.terminal_in_subtree,
            "tree_idx": node.tree_idx, "node_idx": node.node_idx,
            "uid": node.uid, "sub_uid": node.sub_uid, "batch_idx": node.batch_idx,
        })
        node.state_value = child_value
        node.local_adv = local_adv
        node.adv = adv
        node_path.append(node)
        node = node.parent
    return dict_path[::-1], node_path[::-1]


def fill_in_paths(dict_paths, node_paths):
    for dict_path, node_path in zip(dict_paths, node_paths):
        for i in range(1, len(dict_path)):
            epsilon = 1e-8
            if abs(dict_path[i]["local_adv"]) < epsilon:  # 对于每个路径，如果存在value=0，就用前一个节点的value填充
                assert i > 0, "value=0 in the first node"
                assert -epsilon < dict_path[i]["local_adv"] < epsilon, "value is not 0"
                dict_path[i]["local_adv"] = dict_path[i - 1]["local_adv"]
                node_path[i].local_adv = node_path[i - 1].local_adv

    return dict_paths, node_paths


def gather_paths(root: MCTSNode, selected_terminals: list[MCTSNode], pass_k: int):
    dict_paths, node_paths = [], []
    # if len(selected_terminals) < pass_k:
    #     raise ValueError

    for terminal_node in selected_terminals:  # 添加 selected_terminal 的叶子节点路径
        dict_path, node_path = path_from_root_to_node(root, terminal_node)
        dict_paths.append(dict_path)
        node_paths.append(node_path)
    # assert len(dict_paths) == pass_k, f"Failed to generate {pass_k} paths, {len(dict_paths)} instead"

    return dict_paths, node_paths


def serialize_tree_list(tree_list):
    """
    serialize the single tree list.
    """
    return [{
        'tree_idx': node.tree_idx,
        'node_idx': node.node_idx,
        'token_ids': node.token_ids,
        'old_log_probs': node.old_log_probs,
        'entropys': node.entropys,
        'finish_reason': node.finish_reason,
        'reward': node.reward,
        'parent_node_idx': node.parent_node_idx,
        'parent_node_split_idx': node.parent_node_split_idx,
        'child_split_indices': node.child_split_indices,
        'uid': node.uid,
        'sub_uid': node.sub_uid,
        'batch_idx': node.batch_idx,
    } for node in tree_list]


class RunningAverage:
    def __init__(self, alpha=0.0, dtype=torch.float64):
        self.alpha = alpha
        self.mean: Optional[torch.Tensor] = None
        self.count: int = 0
        self.mean2: Optional[torch.Tensor] = None
        self.count2: int = 0
        self.data = []
        self.data2 = []
        self.dtype = dtype

    def add_batch(self, batch: torch.Tensor):
        if batch.numel() == 0:
            return
        batch = batch.reshape(-1)
        self.data.extend(batch.tolist())
        n = batch.shape[0]
        batch_mean = batch.mean()

        if self.count == 0:
            self.mean = batch_mean
            self.count = int(n)
        else:
            total = self.count + int(n)
            if self.alpha > 0:  # EMA
                self.mean = self.alpha * batch_mean + (1 - self.alpha) * self.mean
            else:
                self.mean = (self.mean * self.count + batch_mean * n) / total
            self.count = total

    def add_batch2(self, batch: torch.Tensor):
        if batch.numel() == 0:
            return
        batch = batch.reshape(-1)
        self.data2.extend(batch.tolist())
        n = batch.shape[0]
        batch_mean = batch.mean()

        if self.count2 == 0:
            self.mean2 = batch_mean
            self.count2 = int(n)
        else:
            total = self.count2 + int(n)
            if self.alpha > 0:  # EMA
                self.mean2 = self.alpha * batch_mean + (1 - self.alpha) * self.mean2
            else:
                self.mean2 = (self.mean2 * self.count2 + batch_mean * n) / total
            self.count2 = total

    def get_mean(self):
        return None if self.count == 0 else self.mean

    def reset(self):
        self.mean = None
        self.count = 0
        self.mean2 = None
        self.count2 = 0

    def save_ra(self, path: str):
        save_dict = {"mean": self.mean, "count": self.count, "mean2": self.mean2, "count2": self.count2}
        torch.save(save_dict, path)

    def load_ra(self, path: str):
        obj = torch.load(path, weights_only=False)
        self.mean, self.count = obj["mean"], obj["count"]
        if "mean2" in obj:
            self.mean2, self.count2 = obj["mean2"], obj["count2"]


class TreeWorker:
    def __init__(self, config, pad_token_id=151643, eos_token_id=151643):
        self.config = config
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.paths = defaultdict(dict)
        self.trees = defaultdict(dict)
        self.uid2tree = {}  # uid -> tree_idx
        self.uid2acc = {}  # uid -> question acc
        self.uid2idx = {}  # uid -> question acc
        self.tree2uid = []
        self.root_num = 0
        self.args = {
            "num_traces": 40,
        }

        self.ra = RunningAverage()
        self.not_gather = defaultdict(int)
        self.index2metrics = defaultdict(dict)
        self.metrics = defaultdict(float)
        self.token_distance = 512

    def reset_tree(self):
        self.paths = defaultdict(dict)
        self.trees = defaultdict(dict)
        self.uid2tree = {}  # uid -> tree_idx
        self.uid2idx = {}  # uid -> tree_idx
        self.tree2uid = []
        self.root_num = 0
        self.not_gather = defaultdict(int)
        self.index2metrics = defaultdict(dict)
        self.metrics = defaultdict(float)

    def filter_initial_responses(self, mc_batch: DataProto):
        larger_grpo_filter_criteria = self.config.actor_rollout_ref.rollout.get("larger_grpo_filter_criteria", "")

        attn_scores = mc_batch.batch["attn_scores"][:, 1:]  # (bsz * n, r_len)
        step_nums = mc_batch.batch["step_nums"]

        acc_per_q = torch.tensor(mc_batch.non_tensor_batch["acc"]).reshape(-1, self.config.actor_rollout_ref.rollout.n)
        ent_per_q = []
        attn_per_q = []
        repeat_times = []
        for i in range(len(mc_batch.batch["responses"])):
            response_length = mc_batch.batch["response_mask"][i].sum().item()
            ent_per_q.append(mc_batch.batch["entropys"][i, :response_length].mean().item())
            attn_per_q.append(attn_scores[i, :step_nums[i]].max().item())
        ent_per_q = torch.tensor(ent_per_q).reshape(-1, self.config.actor_rollout_ref.rollout.n)
        attn_per_q = torch.tensor(attn_per_q).reshape(-1, self.config.actor_rollout_ref.rollout.n)
        mean_acc = acc_per_q.mean(dim=-1)
        mean_ent = ent_per_q.mean(dim=-1)

        half_real = self.config.actor_rollout_ref.rollout.n_real // 2
        for i in range(len(acc_per_q)):
            repeat_times_per_q = torch.zeros(self.config.actor_rollout_ref.rollout.n, dtype=torch.int64)
            if larger_grpo_filter_criteria == "attn":
                scores = attn_per_q[i]
            elif larger_grpo_filter_criteria == "random":
                scores = torch.rand(self.config.actor_rollout_ref.rollout.n)
            else:
                raise ValueError(f"Unsupported larger_grpo_filter_criteria: {larger_grpo_filter_criteria}")
            try:
                if mean_acc[i] == 1.0:  # too easy
                    _, idxs_per_resp = torch.topk(scores, k=self.config.actor_rollout_ref.rollout.n_real)
                    idxs_per_resp, sort_indices = torch.sort(idxs_per_resp, dim=-1)
                    repeat_times_per_q[idxs_per_resp] = 1
                elif mean_acc[i] == 0.0:  # too hard
                    _, idxs_per_resp = torch.topk(scores, k=self.config.actor_rollout_ref.rollout.n_real)
                    idxs_per_resp, sort_indices = torch.sort(idxs_per_resp, dim=-1)
                    repeat_times_per_q[idxs_per_resp] = 1
                else:
                    correct_num = int(acc_per_q[i].sum().item())
                    incorrect_num = self.config.actor_rollout_ref.rollout.n - correct_num
                    correct_idxs = torch.where(acc_per_q[i] == 1)[0]
                    incorrect_idxs = torch.where(acc_per_q[i] == 0)[0]
                    if correct_num >= half_real and incorrect_num >= half_real:
                        temp_scores = deepcopy(scores)
                        temp_scores[incorrect_idxs] = -1e5
                        _, idxs_per_resp = torch.topk(temp_scores, k=half_real)
                        idxs_per_resp, sort_indices = torch.sort(idxs_per_resp, dim=-1)
                        repeat_times_per_q[idxs_per_resp] = 1
                        temp_scores = deepcopy(scores)
                        temp_scores[correct_idxs] = -1e5
                        _, idxs_per_resp = torch.topk(temp_scores, k=half_real)
                        idxs_per_resp, sort_indices = torch.sort(idxs_per_resp, dim=-1)
                        repeat_times_per_q[idxs_per_resp] = 1
                    elif incorrect_num < half_real:  # too easy
                        repeat_times_per_q[incorrect_idxs] = 1
                        temp_scores = deepcopy(scores)
                        temp_scores[incorrect_idxs] = -1e5
                        _, idxs_per_resp = torch.topk(temp_scores, k=self.config.actor_rollout_ref.rollout.n_real - incorrect_num)
                        idxs_per_resp, sort_indices = torch.sort(idxs_per_resp, dim=-1)
                        repeat_times_per_q[idxs_per_resp] = 1
                    elif correct_num < half_real:  # too hard
                        repeat_times_per_q[correct_idxs] = 1
                        temp_scores = deepcopy(scores)
                        temp_scores[correct_idxs] = -1e5
                        _, idxs_per_resp = torch.topk(temp_scores, k=self.config.actor_rollout_ref.rollout.n_real - correct_num)
                        idxs_per_resp, sort_indices = torch.sort(idxs_per_resp, dim=-1)
                        repeat_times_per_q[idxs_per_resp] = 1
                    else:
                        breakpoint()
                assert repeat_times_per_q.sum().item() == self.config.actor_rollout_ref.rollout.n_real, f"repeat_times_per_q.sum().item() = {repeat_times_per_q.sum().item()} != n_real = {self.config.actor_rollout_ref.rollout.n_real}"
            except Exception as e:
                print(f"Error in filtering responses: {e}")
                breakpoint()

            repeat_times.extend(repeat_times_per_q.tolist())

        mc_batch = mc_batch.sample_level_repeat(repeat_times=repeat_times)
        print(f"mc_batch size: {mc_batch.batch.batch_size}")
        return mc_batch

    def build_root_nodes(self, mc_batch: DataProto):
        self.reset_tree()
        split_criterion = self.config.algorithm.get("split_criterion", "entropy")  # TODO (lrz): change default value to ""

        for i in range(len(mc_batch.batch["responses"])):
            idx = len(self.trees)
            uid = mc_batch.non_tensor_batch["uid"][i]
            sub_uid = mc_batch.non_tensor_batch["sub_uid"][i]

            # if node.is_end and node.finish_reason == "stop":  # TODO (lrz): check is_end and finish_reason
            acc = mc_batch.non_tensor_batch["acc"][i]

            if uid not in self.uid2tree:
                self.uid2tree[uid] = idx
                self.uid2idx[uid] = int(mc_batch.non_tensor_batch["index"][i])
                self.paths[uid]["pass_k_result"] = [acc]
                self.paths[uid]["prompts"] = mc_batch.batch["prompts"][i, -mc_batch.batch["attention_mask"][i, :mc_batch.batch["prompts"].size(1)].sum().item():].tolist()
            else:
                self.paths[uid]["pass_k_result"].append(acc)
            response_length = mc_batch.batch["attention_mask"][i, self.config.actor_rollout_ref.rollout.prompt_length:].sum().item()
            content_token_ids = mc_batch.batch["responses"][i, :response_length].tolist()
            old_log_probs = mc_batch.batch["old_log_probs"][i, :response_length].tolist()

            root_node = TreeNode(
                tree_idx=sub_uid,
                node_idx=0,
                token_ids=content_token_ids,
                old_log_probs=old_log_probs,
                entropys=mc_batch.batch["entropys"][i, :response_length].tolist(),
                finish_reason=mc_batch.non_tensor_batch["finish_reasons"][i],
                reward=acc,
                uid=uid,
                sub_uid=sub_uid,
                batch_idx=self.root_num,
            )
            self.trees[uid][sub_uid] = [root_node]
            self.tree2uid.append(uid)
            self.root_num += 1

        index_per_q = mc_batch.non_tensor_batch["index"].reshape(-1, self.config.actor_rollout_ref.rollout.n)[:, 0].astype(int)
        acc_per_q = torch.tensor(mc_batch.non_tensor_batch["acc"]).reshape(-1, self.config.actor_rollout_ref.rollout.n)
        mean_acc = acc_per_q.mean(dim=-1)
        for i in range(len(acc_per_q)):
            mean_acc_i = mean_acc[i].item()
            self.index2metrics[index_per_q[i]]["acc"] = mean_acc_i
            self.index2metrics[index_per_q[i]]["n_correct"] = int(mean_acc_i * self.config.actor_rollout_ref.rollout.n)
            self.index2metrics[index_per_q[i]]["cnt"] = self.config.actor_rollout_ref.rollout.n
            self.metrics[f"batch/n_correct={int(mean_acc_i * self.config.actor_rollout_ref.rollout.n)}"] += 1

        if split_criterion == "entropy":
            mc_type = self.config.algorithm.get("mc_type", "uniform")
            N = self.config.algorithm.get("num_splits", 2)
            entropys = mc_batch.pop(batch_keys=["entropys"]).batch["entropys"]
            token_idxs, token_values = [], []
            if mc_type == "uniform":
                mc_batch = mc_batch.repeat(repeat_times=N, interleave=True)  # TODO (lrz): maybe change N to a dynamic value

                for i in range(len(entropys)):
                    response_length = mc_batch.batch["attention_mask"][i, self.config.actor_rollout_ref.rollout.prompt_length:].sum().item()
                    token_values_i, token_idxs_i = torch.topk(entropys[i, :response_length - 1], k=N, dim=-1)  # (k, )
                    token_idxs.append(token_idxs_i)
                    token_values.append(token_values_i)
                token_idxs = torch.stack(token_idxs, dim=0)
                token_idxs, sort_indices = torch.sort(token_idxs, dim=-1)
                # token_values = torch.gather(token_values, dim=-1, index=sort_indices)
                padded_token_idxs = token_idxs

            token_idxs = token_idxs.reshape(-1)
            token_idxs = token_idxs[token_idxs >= 0]
            mc_batch.batch["token_idxs"] = token_idxs
        elif split_criterion == "FCI":
            mc_type = self.config.algorithm.get("mc_type", "uniform")
            N = self.config.algorithm.get("num_splits", 2)  # TODO (lrz): change default value
            token_ranges = mc_batch.pop(batch_keys=["token_ranges"]).batch["token_ranges"][:, 1:]
            attn_scores = mc_batch.batch["attn_scores"][:, 1:]  # (bsz * n, r_len)
            step_nums = mc_batch.batch["step_nums"]
            mean_attn_per_q = []
            for i in range(len(attn_scores)):
                mean_attn_per_q.append(torch.mean(attn_scores[i, :step_nums[i]]))
            mean_attn_per_q = torch.stack(mean_attn_per_q)
            mean_attn = mean_attn_per_q.reshape(-1, self.config.actor_rollout_ref.rollout.n).mean(dim=-1)

            self.ra.add_batch(mean_attn[mean_acc == 1])
            self.ra.add_batch2(mean_attn[mean_acc == 0])

            def check_token_distance(idx, step_idx, token_ranges, token_distance):
                sep_token_start, sep_token_end = token_ranges[idx, 0].item(), token_ranges[idx, 1].item()
                if sep_token_start < token_distance:
                    return False
                for selected_idx in step_idx:
                    selected_token_start, selected_token_end = token_ranges[selected_idx, 0].item(), token_ranges[selected_idx, 1].item()
                    if selected_token_start >= sep_token_end and selected_token_start - sep_token_end < token_distance:
                        return False
                    elif sep_token_start >= selected_token_end and sep_token_start - selected_token_end < token_distance:
                        return False
                return True

            def find_top_k_token_idxs(i, sep_num, shuffle=False):
                attn_score = attn_scores[i, :step_nums[i]]
                threshold = torch.quantile(attn_score, 0.8).item()
                if threshold == 0 and attn_score[attn_score > 0].numel() > 0:
                    threshold = attn_score[attn_score > 0].min()
                all_step_idxs = torch.arange(1, step_nums[i])
                candidate_step_idxs = torch.where(attn_score >= threshold)[0]  # (N, )
                if candidate_step_idxs[0] == 0:
                    candidate_step_idxs = candidate_step_idxs[1:]
                if candidate_step_idxs.numel() > 0 and shuffle:  # random shuffle the candidate_step_idxs
                    candidate_step_idxs = candidate_step_idxs[torch.randperm(len(candidate_step_idxs))]
                assert 0 not in candidate_step_idxs.tolist(), f"candidate_step_idxs {candidate_step_idxs} should not contain 0, because it is the first step"
                token_idx = []
                step_idx = []
                ptr = 0
                while candidate_step_idxs.numel() > 0 and ptr < len(candidate_step_idxs):
                    idx = candidate_step_idxs[ptr]
                    sep_idx = token_ranges[i, idx, 0].item() - token_ranges[i, 0, 0].item()
                    assert 0 < sep_idx < self.config.actor_rollout_ref.rollout.response_length, f"sep_idx {sep_idx} out of range, should be in (0, {self.config.actor_rollout_ref.rollout.response_length})"
                    if check_token_distance(idx, step_idx, token_ranges[i], token_distance=self.token_distance):
                        token_idx.append(sep_idx)
                        step_idx.append(idx)
                    if len(token_idx) == sep_num:
                        break
                    ptr += 1
                if len(token_idx) < sep_num:
                    ptr = 0
                    while ptr < len(all_step_idxs):
                        idx = all_step_idxs[ptr]
                        sep_idx = token_ranges[i, idx, 0].item() - token_ranges[i, 0, 0].item()
                        if sep_idx not in token_idx:
                            token_idx.append(sep_idx)
                        if len(token_idx) == sep_num:
                            break
                        ptr += 1
                while len(token_idx) < sep_num:
                    ptr = 0
                    while ptr < len(all_step_idxs):
                        idx = all_step_idxs[ptr]
                        sep_idx = token_ranges[i, idx, 0].item() - token_ranges[i, 0, 0].item()
                        token_idx.append(sep_idx)
                        if len(token_idx) == sep_num:
                            return token_idx
                        ptr += 1
                return token_idx

            idx2token_idxs = defaultdict(list)
            selected_token_idxs = defaultdict(list)
            token_idxs = []
            if mc_type == "uniform":
                for i in range(len(acc_per_q)):
                    mean_acc_i = mean_acc[i].item()
                    if mean_acc_i == 1.0:  # too easy, 6/6
                        for j in range(self.config.actor_rollout_ref.rollout.n):
                            resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                            selected_token_idxs[resp_idx].extend(find_top_k_token_idxs(resp_idx, N))
                    elif mean_acc_i == 0.0:  # too hard, 0/6
                        for j in range(self.config.actor_rollout_ref.rollout.n):
                            resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                            selected_token_idxs[resp_idx].extend(find_top_k_token_idxs(resp_idx, N))
                    else:  # 1-5/6
                        for j in range(self.config.actor_rollout_ref.rollout.n):
                            resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                            selected_token_idxs[resp_idx].extend(find_top_k_token_idxs(resp_idx, N))

                repeat_times = torch.zeros(len(acc_per_q) * self.config.actor_rollout_ref.rollout.n, dtype=torch.int64)
                for i in range(len(acc_per_q)):
                    for j in range(self.config.actor_rollout_ref.rollout.n):
                        resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                        repeat_times[resp_idx] = len(selected_token_idxs[resp_idx])
                        token_idxs.append(torch.sort(torch.tensor(selected_token_idxs[resp_idx]))[0].tolist())

                print(f"{repeat_times.sum().item()=}, {len(self.not_gather)=}")
                token_idxs = pad_2d_list_to_length(token_idxs, pad_token_id=-1, max_length=N * 10)
                padded_token_idxs = token_idxs
                mc_batch = mc_batch.sample_level_repeat(repeat_times=repeat_times)
            else:
                mc_diff_type = self.config.algorithm.get("mc_diff_type", "exp")
                diff = 1 - np.arange(0, self.config.actor_rollout_ref.rollout.n + 1) / self.config.actor_rollout_ref.rollout.n
                if mc_diff_type == "exp":
                    exp = np.exp(diff - 1)
                    diff2num = exp * self.config.actor_rollout_ref.rollout.n
                    diff2num = np.ceil(diff2num).astype(np.int64)
                    print(f"{diff2num=}")
                elif mc_diff_type == "uniform":
                    exp = np.ones_like(diff)
                    diff2num = exp * self.config.actor_rollout_ref.rollout.n
                    diff2num = np.ceil(diff2num).astype(np.int64)
                    print(f"{diff2num=}")
                acc2num = {i / self.config.actor_rollout_ref.rollout.n: diff2num[i] for i in range(self.config.actor_rollout_ref.rollout.n + 1)}
                # 排序（优先级：难度 > attention 分数）
                scores4sort = []
                easy_idxs, hard_idxs, other_idxs = [], [], []
                for i in range(len(acc_per_q)):
                    mean_acc_i = mean_acc[i].item()
                    if mean_acc_i == 1.0:  # too easy, 6/6
                        easy_idxs.append(i)
                        for j in range(self.config.actor_rollout_ref.rollout.n):
                            resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                            idx2token_idxs[resp_idx].extend(find_top_k_token_idxs(resp_idx, N))
                        self.metrics[f"batch/initial_num={acc2num[mean_acc_i]}"] += 1
                    elif mean_acc_i == 0.0:  # too hard, 0/6
                        hard_idxs.append(i)
                        for j in range(self.config.actor_rollout_ref.rollout.n):
                            resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                            temp_idxs = find_top_k_token_idxs(resp_idx, N)
                            temp_idxs.extend(find_top_k_token_idxs(resp_idx, N))
                            idx2token_idxs[resp_idx].extend(temp_idxs)
                        self.metrics[f"batch/initial_num={acc2num[mean_acc_i]}"] += 1
                    else:  # 1-5/6
                        for j in range(self.config.actor_rollout_ref.rollout.n):
                            resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                            temp_idxs = find_top_k_token_idxs(resp_idx, N * 2)
                            idx2token_idxs[resp_idx].extend(temp_idxs)
                        self.metrics[f"batch/initial_num={acc2num[mean_acc_i]}"] += 1
                    s = (1 - mean_acc[i] + 1) * 100 + mean_attn[i]
                    scores4sort.append(s)

                scores4sort = torch.tensor(scores4sort)
                res = torch.sort(scores4sort, descending=True, stable=True)
                values, indices = res.values, res.indices

                repeat_times = torch.zeros(len(acc_per_q) * self.config.actor_rollout_ref.rollout.n, dtype=torch.int64)

                multi_flags = [True] * len(acc_per_q)
                temp_N = N
                for i in range(len(indices)):
                    q_idx = indices[i].item()
                    mean_acc_i = mean_acc[q_idx].item()
                    if not multi_flags[q_idx]:
                        continue
                    if mean_acc_i == 1.0:  # too easy
                        multi_flags[q_idx] = False
                        if mc_type == "attention_based_filtering":
                            if mean_attn[q_idx] > self.ra.mean:
                                for j in range(acc2num[mean_acc_i]):
                                    resp_idx = q_idx * self.config.actor_rollout_ref.rollout.n + j
                                    selected_token_idxs[resp_idx].extend(idx2token_idxs[resp_idx][:temp_N])
                            else:
                                resp_idx = q_idx * self.config.actor_rollout_ref.rollout.n + 0
                                uid = mc_batch.non_tensor_batch["uid"][resp_idx]
                                self.not_gather[uid] = 1
                        else:
                            raise NotImplementedError
                    elif mean_acc_i == 0.0:  # too hard
                        for j in range(acc2num[mean_acc_i]):
                            resp_idx = q_idx * self.config.actor_rollout_ref.rollout.n + j
                            curr_len = len(selected_token_idxs[resp_idx])
                            if mc_type == "easy_drop_0.8_hard_explore_1.2_attn" and mean_attn[q_idx] > 1.2 * self.ra.mean2:
                                if len(idx2token_idxs[resp_idx]) < curr_len + 2 * temp_N:
                                    continue
                                selected_token_idxs[resp_idx].extend(idx2token_idxs[resp_idx][curr_len:curr_len + 2 * temp_N])
                            else:
                                if len(idx2token_idxs[resp_idx]) < curr_len + temp_N:
                                    continue
                                selected_token_idxs[resp_idx].extend(idx2token_idxs[resp_idx][curr_len:curr_len + temp_N])
                    else:
                        multi_flags[q_idx] = False
                        temp_N_other = temp_N
                        num_to_sample = acc2num[mean_acc_i]
                        correct_idxs = torch.where(acc_per_q[q_idx] > 0)[0].numpy()
                        incorrect_idxs = torch.where(acc_per_q[q_idx] <= 0)[0].numpy()
                        idxs_to_sample = []
                        if mean_acc_i >= 0.5:
                            idxs_to_sample.extend(incorrect_idxs.tolist())
                            idxs_to_sample.extend(np.random.choice(correct_idxs, num_to_sample - len(incorrect_idxs), replace=False).tolist())
                        else:
                            idxs_to_sample.extend(correct_idxs.tolist())
                            idxs_to_sample.extend(np.random.choice(incorrect_idxs, num_to_sample - len(correct_idxs), replace=False).tolist())
                        for j in idxs_to_sample:
                            resp_idx = q_idx * self.config.actor_rollout_ref.rollout.n + j
                            curr_len = len(selected_token_idxs[resp_idx])
                            if len(idx2token_idxs[resp_idx]) < curr_len + temp_N_other:
                                continue
                            selected_token_idxs[resp_idx].extend(idx2token_idxs[resp_idx][curr_len:curr_len + temp_N_other])

                for i in range(len(acc_per_q)):
                    for j in range(self.config.actor_rollout_ref.rollout.n):
                        resp_idx = i * self.config.actor_rollout_ref.rollout.n + j
                        if len(selected_token_idxs[resp_idx]):
                            repeat_times[resp_idx] = len(selected_token_idxs[resp_idx])
                            token_idxs.append(torch.sort(torch.tensor(selected_token_idxs[resp_idx]))[0].tolist())
                        else:
                            token_idxs.append([-1, -1])

                print(f"{repeat_times.sum().item()=}, {len(self.not_gather)=}")

                token_idxs = pad_2d_list_to_length(token_idxs, pad_token_id=-1, max_length=N * 10)
                padded_token_idxs = token_idxs

                mc_batch = mc_batch.sample_level_repeat(repeat_times=repeat_times)
            token_idxs = token_idxs.reshape(-1)  # (bsz * n * k, )
            token_idxs = token_idxs[token_idxs >= 0]
            mc_batch.batch["token_idxs"] = token_idxs
        else:
            raise ValueError(f"Unsupported split criterion: {split_criterion}")

        mc_raw_prompt_ids = []
        for i in range(len(mc_batch.batch["responses"])):
            token_idx = token_idxs[i]
            prompt_id = mc_batch.batch["prompts"][i]
            prompt_length = mc_batch.batch["attention_mask"][i, :mc_batch.batch["prompts"][i].shape[-1]].sum().item()
            try:
                assert (mc_batch.batch["responses"][i, :token_idx] == self.pad_token_id).sum().item() == 0, f"pad_token_id {self.pad_token_id} appears in mc_batch.batch['responses'][{i}, :{token_idx}]"
            except Exception as e:
                print(e)
                breakpoint()
            mc_raw_prompt_id = prompt_id[-prompt_length:].tolist() + mc_batch.batch["responses"][i, :token_idx].tolist()
            mc_raw_prompt_ids.append(mc_raw_prompt_id)
        mc_batch.non_tensor_batch["raw_prompt_ids"] = mc_raw_prompt_ids

        if split_criterion == "entropy":
            del entropys
        torch.cuda.empty_cache()

        return mc_batch, padded_token_idxs

    def build_trees(self, mc_batch: DataProto, batch: DataProto):
        for i in range(len(mc_batch.batch["responses"])):
            response_length = mc_batch.batch["attention_mask"][i, self.config.actor_rollout_ref.rollout.prompt_length:].sum().item()
            content_token_ids = mc_batch.batch["responses"][i, :response_length].tolist()

            uid = mc_batch.non_tensor_batch["uid"][i]
            sub_uid = mc_batch.non_tensor_batch["sub_uid"][i]
            tree_idx = self.uid2tree[uid]
            parent_node = self.trees[uid][sub_uid][0]
            split_idx = mc_batch.batch["token_idxs"][i].item()
            acc = mc_batch.non_tensor_batch["acc"][i].item()
            new_node = TreeNode(
                tree_idx=sub_uid,
                node_idx=len(self.trees[uid][sub_uid]),
                token_ids=content_token_ids,
                old_log_probs=[0.0] * response_length,
                entropys=[0.0] * response_length,
                parent_node=parent_node,
                parent_node_idx=0,
                parent_node_split_idx=split_idx,
                finish_reason=mc_batch.non_tensor_batch["finish_reasons"][i],
                reward=acc,
                uid=uid,
                sub_uid=sub_uid,
                batch_idx=i + self.root_num,
            )

            parent_node.add_child(new_node, split_idx)
            self.trees[uid][sub_uid].append(new_node)
            self.paths[uid]["pass_k_result"].append(acc)

        data_indices = mc_batch.non_tensor_batch["index"]
        unique_data_indices, data_indices_indices, unique_data_cnts = np.unique(data_indices, return_index=True, return_counts=True)
        unique_data_indices = unique_data_indices.astype(int)
        for i, data_index in enumerate(unique_data_indices):
            data_index_mask = data_indices == data_index
            index_avg_acc = mc_batch.non_tensor_batch["acc"][data_index_mask].mean()  # Sum rewards for each sequence
            self.index2metrics[data_index]["acc_mc"] = index_avg_acc
            self.index2metrics[data_index]["n_correct_mc"] = int(index_avg_acc * len(mc_batch.non_tensor_batch["acc"][data_index_mask]))
            self.index2metrics[data_index]["cnt_mc"] = len(mc_batch.non_tensor_batch["acc"][data_index_mask])

        for uid, tree_dict in self.trees.items():
            self.paths[uid]['tree_structures'] = [serialize_tree_list(tree) for sub_uid, tree in tree_dict.items()]

        for uid, tree_dict in self.trees.items():
            tree_list = list(tree_dict.values())  # TODO (lrz): check here
            root, all_leaves = build_into_tree_format(tree_list)
            root, selected_terminals = process_leaf(
                root,
                all_leaves,
            )
            paths, nodes = gather_paths(
                root=root,
                selected_terminals=selected_terminals,
                pass_k=self.args["num_traces"],
            )
            self.paths[uid]["paths"] = paths

        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        # tensors
        prompts = []
        responses = []
        temp_old_log_probs_lst = []
        is_grpos = []
        grpo_token_nums = []
        attention_mask = []
        position_ids = []
        advantages = []
        # non tensors
        uid_lst = []
        avg_accs = []
        accs = []
        indexs = []
        grpo_len_one_cnt = 0

        for uid, uid_item in self.paths.items():
            if uid in self.not_gather.keys():
                continue
            uid_paths = uid_item["paths"]
            for path in uid_paths:
                if len(path) == 1:
                    grpo_len_one_cnt += 1
                    continue
                rewards = []
                response_ids = []
                temp_old_log_probs = []
                temp_accs = []
                is_grpo = True
                grpo_token_num = 0
                for i in range(len(path)):
                    is_grpo = path[i]["is_grpo"]
                    if is_grpo:
                        grpo_token_num += len(path[i]["token_ids"])
                        temp_old_log_probs += path[i]["old_log_probs"]
                    response_ids += path[i]["token_ids"]
                    reward = path[i]["adv"]
                    rewards += [reward] * len(path[i]["token_ids"])
                    temp_accs.append(path[i]["reward"])
                is_grpos.append(is_grpo)
                grpo_token_nums.append(grpo_token_num)
                if len(response_ids) > max_response_length:
                    response_ids = response_ids[:max_response_length]
                    temp_old_log_probs = temp_old_log_probs[:max_response_length]
                    rewards = rewards[:max_response_length]
                if len(temp_old_log_probs) > grpo_token_num:
                    temp_old_log_probs = temp_old_log_probs[:grpo_token_num]

                responses.append(response_ids)
                temp_old_log_probs_lst.append(temp_old_log_probs)
                avg_accs.append(sum(temp_accs) / len(temp_accs))
                accs.append(temp_accs[-1])
                indexs.append(self.uid2idx[uid])

                advantages.append(rewards)
                if path[-1]["node_idx"] == 0:
                    prompts.append(batch.batch["prompts"][path[-1]["batch_idx"]])
                    attention_mask.append(batch.batch["attention_mask"][path[-1]["batch_idx"]])
                    position_ids.append(batch.batch["position_ids"][path[-1]["batch_idx"]])
                    uid_lst.append(batch.non_tensor_batch["uid"][path[-1]["batch_idx"]])
                else:
                    prompts.append(mc_batch.batch["prompts"][path[-1]["batch_idx"] - self.root_num])
                    attention_mask.append(mc_batch.batch["attention_mask"][path[-1]["batch_idx"] - self.root_num])
                    position_ids.append(mc_batch.batch["position_ids"][path[-1]["batch_idx"] - self.root_num])
                    uid_lst.append(mc_batch.non_tensor_batch["uid"][path[-1]["batch_idx"] - self.root_num])

        prompts = torch.stack(prompts)
        responses = pad_2d_list_to_length(responses, self.pad_token_id, max_response_length)
        temp_old_log_probs_lst = pad_2d_list_to_length(temp_old_log_probs_lst, 0.0, max_response_length)
        is_grpos = np.where(np.array(is_grpos), 1, 0)
        grpo_token_nums = np.array(grpo_token_nums)
        advantages = pad_2d_list_to_length(advantages, 0, max_response_length).to(torch.float32)
        input_ids = torch.cat([prompts, responses], dim=-1)
        prompt_attention_mask = torch.stack(attention_mask)[:, :-max_response_length]
        response_attention_mask = get_response_mask(response_id=responses, eos_token=self.eos_token_id, dtype=prompt_attention_mask.dtype)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.stack(position_ids)

        meta_info = batch.meta_info

        batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "responses": responses,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_level_scores": advantages,
                "old_log_probs": temp_old_log_probs_lst,
            },
            non_tensors={
                "uid": np.array(uid_lst),
                "avg_accs": np.array(avg_accs),
                "accs": np.array(accs),
                "is_grpo": is_grpos,
                "grpo_token_num": grpo_token_nums,
                "index": np.array(indexs),
            },
            meta_info=meta_info,
        )
        grpo_idxs = np.where(batch.non_tensor_batch["is_grpo"] == 1)[0]
        grpo_batch = batch.select_idxs(grpo_idxs)
        mc_idxs = np.where(batch.non_tensor_batch["is_grpo"] == 0)[0]
        mc_batch = batch.select_idxs(mc_idxs)

        for i, data_index in enumerate(self.uid2idx.values()):
            data_index_mask = batch.non_tensor_batch["index"] == data_index
            if np.sum(data_index_mask) > 0:
                index_avg_acc = batch.non_tensor_batch["accs"][data_index_mask].mean()  # Sum rewards for each sequence
                self.index2metrics[data_index]["acc_all"] = index_avg_acc
                self.index2metrics[data_index]["n_correct_all"] = int(index_avg_acc * len(batch.non_tensor_batch["accs"][data_index_mask]))
                self.index2metrics[data_index]["cnt_all"] = len(batch.non_tensor_batch["accs"][data_index_mask])
            else:
                self.index2metrics[data_index]["acc_all"] = self.index2metrics[data_index]["acc"]
                self.index2metrics[data_index]["n_correct_all"] = self.index2metrics[data_index]["n_correct"]
                self.index2metrics[data_index]["cnt_all"] = self.index2metrics[data_index]["cnt"]
            if self.index2metrics[data_index]["cnt"] > 0:
                if self.index2metrics[data_index]["acc"] == 1:
                    self.metrics["batch/correct_invalid_once"] += 1
                    if "cnt_mc" in self.index2metrics[data_index].keys() and self.index2metrics[data_index]["cnt_mc"] > 0:
                        if self.index2metrics[data_index]["acc_mc"] == 1:
                            self.metrics["batch/correct_invalid_twice"] += 1
                    else:
                        self.metrics["batch/correct_filtered"] += 1
                if self.index2metrics[data_index]["acc"] == 0:
                    self.metrics["batch/incorrect_invalid_once"] += 1
                    if "cnt_mc" in self.index2metrics[data_index].keys() and self.index2metrics[data_index]["cnt_mc"] > 0:
                        if self.index2metrics[data_index]["acc_mc"] == 0:
                            self.metrics["batch/incorrect_invalid_twice"] += 1
                    else:
                        self.metrics["batch/incorrect_filtered"] += 1

        # filtered
        self.metrics["batch/total_filtered"] = self.metrics["batch/correct_filtered"] + self.metrics["batch/incorrect_filtered"]
        self.metrics["batch/correct_filtered_ratio"] = self.metrics["batch/correct_filtered"] / len(unique_data_indices)
        self.metrics["batch/incorrect_filtered_ratio"] = self.metrics["batch/incorrect_filtered"] / len(unique_data_indices)
        self.metrics["batch/total_filtered_ratio"] = (self.metrics["batch/correct_filtered"] + self.metrics["batch/incorrect_filtered"]) / len(unique_data_indices)
        self.metrics["batch/correct_once_filtered_ratio"] = self.metrics["batch/correct_filtered"] / self.metrics["batch/correct_invalid_once"] if self.metrics["batch/correct_invalid_once"] > 0 else 0.0
        self.metrics["batch/incorrect_once_filtered_ratio"] = self.metrics["batch/incorrect_filtered"] / self.metrics["batch/incorrect_invalid_once"] if self.metrics["batch/incorrect_invalid_once"] > 0 else 0.0

        # invalid twice
        self.metrics["batch/total_invalid_twice"] = self.metrics["batch/correct_invalid_twice"] + self.metrics["batch/incorrect_invalid_twice"]
        self.metrics["batch/correct_invalid_twice_ratio"] = self.metrics["batch/correct_invalid_twice"] / len(unique_data_indices)
        self.metrics["batch/incorrect_invalid_twice_ratio"] = self.metrics["batch/incorrect_invalid_twice"] / len(unique_data_indices)
        self.metrics["batch/total_invalid_twice_ratio"] = (self.metrics["batch/correct_invalid_twice"] + self.metrics["batch/incorrect_invalid_twice"]) / len(unique_data_indices)
        self.metrics["batch/correct_once_twice_invalid_ratio"] = self.metrics["batch/correct_invalid_twice"] / self.metrics["batch/correct_invalid_once"] if self.metrics["batch/correct_invalid_once"] > 0 else 0.0
        self.metrics["batch/incorrect_once_twice_invalid_ratio"] = self.metrics["batch/incorrect_invalid_twice"] / self.metrics["batch/incorrect_invalid_once"] if self.metrics["batch/incorrect_invalid_once"] > 0 else 0.0

        # invalid once but valid twice
        self.metrics["batch/correct_once_twice_valid"] = self.metrics["batch/correct_invalid_once"] - self.metrics["batch/correct_invalid_twice"]
        self.metrics["batch/incorrect_once_twice_valid"] = self.metrics["batch/incorrect_invalid_once"] - self.metrics["batch/incorrect_invalid_twice"]
        self.metrics["batch/correct_once_twice_valid_ratio"] = self.metrics["batch/correct_once_twice_valid"] / self.metrics["batch/correct_invalid_once"] if self.metrics["batch/correct_invalid_once"] > 0 else 0.0
        self.metrics["batch/incorrect_once_twice_valid_ratio"] = self.metrics["batch/incorrect_once_twice_valid"] / self.metrics["batch/incorrect_invalid_once"] if self.metrics["batch/incorrect_invalid_once"] > 0 else 0.0

        del input_ids, responses, attention_mask, position_ids, advantages, batch
        torch.cuda.empty_cache()

        return grpo_batch, mc_batch

    def load_trees(self, data):
        self.reset_tree()
        self.paths = data

        for idx, (uid, data_per_q) in enumerate(data.items()):
            tree_structures = data_per_q['tree_structures']
            for tree_idx, tree_list in enumerate(tree_structures):
                for node_idx, tree_dict in enumerate(tree_list):
                    try:
                        assert uid == tree_dict["uid"] and tree_idx == tree_dict["tree_idx"] and node_idx == tree_dict["node_idx"]
                    except AssertionError:
                        print(uid, tree_dict["uid"], tree_idx, tree_dict["tree_idx"], node_idx, tree_dict["node_idx"])
                    if uid not in self.uid2tree:
                        self.uid2tree[uid] = idx

                    if tree_dict["node_idx"] == 0:
                        parent_node = TreeNode(
                            tree_idx=tree_dict["tree_idx"],
                            node_idx=tree_dict["node_idx"],
                            token_ids=tree_dict["token_ids"],
                            old_log_probs=tree_dict["old_log_probs"],
                            entropys=tree_dict["entropys"],
                            finish_reason=tree_dict["finish_reason"],
                            reward=tree_dict["reward"],
                            uid=tree_dict["uid"],
                            sub_uid=tree_dict["sub_uid"],
                            batch_idx=tree_dict["batch_idx"],
                        )
                        self.uid2tree[uid] = idx
                        self.trees[uid][tree_dict["sub_uid"]] = [parent_node]
                        self.tree2uid.append(uid)
                        self.root_num += 1
                    else:
                        new_node = TreeNode(
                            tree_idx=tree_dict["tree_idx"],
                            node_idx=tree_dict["node_idx"],
                            token_ids=tree_dict["token_ids"],
                            old_log_probs=tree_dict["old_log_probs"],
                            entropys=tree_dict["entropys"],
                            parent_node=parent_node,
                            parent_node_idx=0,
                            parent_node_split_idx=tree_dict["parent_node_split_idx"],
                            finish_reason=tree_dict["finish_reason"],
                            reward=tree_dict["reward"],
                            uid=tree_dict["uid"],
                            sub_uid=tree_dict["sub_uid"],
                            batch_idx=tree_dict["batch_idx"],
                        )
                        parent_node.add_child(new_node, tree_dict["parent_node_split_idx"])
                        self.trees[uid][tree_dict["sub_uid"]].append(new_node)
