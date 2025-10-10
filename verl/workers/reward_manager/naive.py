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

from collections import defaultdict, Counter

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import re


# 思考与推理 (Thinking and Reasoning)：主动认知过程，如分析、推断、假设。Token数量：25个
thinking_and_reasoning = ["analyze", "analyzing", "analysis", "cogitate", "cogitation", "conclude", "conclusion", "deduce", "deduction", "determine", "determining", "determination", "hypothesize", "infer", "inference", "logic", "logical", "reason", "reasoning", "speculate", "speculation", "think", "thinking", "thought", "theorize"]

# 计划与策略 (Planning and Strategy)：制定计划、方法、预测或策略。Token数量：26个
planning_and_strategy = ["algorithm", "algorithmic", "approach", "approaches", "deliberate", "deliberation", "forecast", "forecasting", "method", "methods", "plan", "planning", "plans", "predict", "prediction", "process", "processing", "scheme", "scheming", "solution", "solve", "solving", "strategy", "strategize", "tactic", "tactics"]

# 评估与验证 (Evaluation and Verification)：判断、评估或验证信息。Token数量：12个
evaluation_and_verification = ["assess", "assessment", "evaluate", "evaluation", "judge", "judgment", "rationalize", "rationalization", "validate", "validation", "verify", "verification"]

# 决策与问题解决 (Decision Making and Problem Solving)：做选择、解决难题、处理疑问。Token数量：18个
decision_making = ["choose", "choosing", "decide", "deciding", "decision", "dilemma", "doubt", "issue", "option", "options", "problem", "query", "queries", "question", "resolve", "resolution", "select", "selecting"]

# 反思与回顾 (Reflection and Contemplation)：回顾、深思或权衡经验。Token数量：14个
reflection_and_contemplation = ["contemplate", "contemplation", "muse", "musing", "ponder", "pondering", "reflect", "reflection", "retrospect", "retrospection", "review", "reviewing", "weigh", "weighing"]

# 概念与理论 (Concepts and Theories)：抽象概念、想法、模型或原则。Token数量：16个
concepts_and_theories = ["concept", "concepts", "hypothesis", "hypotheses", "idea", "ideas", "model", "models", "notion", "notions", "paradox", "paradoxes", "principle", "principles", "theory", "theories"]

# 逻辑连接与可能性 (Logical Connectives and Possibility)：逻辑连接词、推理过渡词，或表示可能性的副词。Token数量：29个
logical_connectives = ["alternatively", "although", "and", "because", "but", "either", "even", "hence", "however", "if", "just", "maybe", "nevertheless", "neither", "nor", "only", "or", "perhaps", "possibly", "probably", "since", "so", "still", "then", "therefore", "though", "thus", "while", "yet"]


def normalize_and_tokenize(text: str) -> list:
    """
    Convert text to lowercase, remove punctuation, and tokenize.
    Returns a list of words.
    """
    # Convert to lowercase
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize by splitting on whitespace
    return text.split()


def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    if len(words) < ngram_size:
        return []
    return list(zip(*[words[i:] for i in range(ngram_size)]))


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def calculate_thinking_tokens(self, response_strs):
        frequency_thinking_tokens = defaultdict(list)
        for sequence in response_strs:
            tokens = normalize_and_tokenize(sequence)
            counts = Counter(tokens)
            frequency_thinking_tokens['token/thinking_and_reasoning'].append(sum(counts[term] for term in thinking_and_reasoning))
            frequency_thinking_tokens['token/planning_and_strategy'].append(sum(counts[term] for term in planning_and_strategy))
            frequency_thinking_tokens['token/evaluation_and_verification'].append(sum(counts[term] for term in evaluation_and_verification))
            frequency_thinking_tokens['token/decision_making'].append(sum(counts[term] for term in decision_making))
            frequency_thinking_tokens['token/reflection_and_contemplation'].append(sum(counts[term] for term in reflection_and_contemplation))
            frequency_thinking_tokens['token/concepts_and_theories'].append(sum(counts[term] for term in concepts_and_theories))
            frequency_thinking_tokens['token/logical_connectives'].append(sum(counts[term] for term in logical_connectives))

        return frequency_thinking_tokens

    def get_repetition_ratio(self, response_strs):
        repetition_ratios = []
        most_repeated = []
        batch_ngrams_counts = defaultdict(int)
        ngram_size = 20

        for idx, sequence in enumerate(response_strs):
            ngrams_counts = defaultdict(int)
            ngrams = zipngram(sequence, ngram_size)
            total_ngrams = len(ngrams)

            if total_ngrams == 0:
                repetition_ratios.append(0.0)
                most_repeated.append(0.0)
                continue

            seen_ngrams = set()
            repeated_count = 0

            for ng in ngrams:
                ng = (f"Sequence_id {idx}: ",) + ng
                ngrams_counts[ng] += 1
                batch_ngrams_counts[ng] += 1
                if ng in seen_ngrams:
                    repeated_count += 1
                else:
                    seen_ngrams.add(ng)

            repetition_ratios.append(repeated_count / total_ngrams)
            most_repeated.append(sorted(ngrams_counts.items(), key=lambda x: x[1], reverse=True)[0][1])

        if batch_ngrams_counts:
            batch_most_repeated = sorted(batch_ngrams_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            for i, (ngram, count) in enumerate(batch_most_repeated):
                print(f"Batch Top-{i + 1} Frequency: {count}")
                print(f"N-gram: {' '.join(ngram)}")

        return defaultdict(list, {'token/maximum_frequency': most_repeated, 'token/repetition_ratio': repetition_ratios})

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        response_strs = []

        # already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # prompt_ids = data_item.batch["prompts"]

            prompt_length = data_item.batch["prompts"].shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_strs.append(response_str)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0
            #
            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print("[prompt]", prompt_str)
            #     print("[response]", response_str)
            #     print("[ground_truth]", ground_truth)
            #     if isinstance(result, dict):
            #         for key, value in result.items():
            #             print(f"[{key}]", value)
            #     else:
            #         print("[score]", score)

        thinking_tokens_info = self.calculate_thinking_tokens(response_strs)
        repetition_info = self.get_repetition_ratio(response_strs)
        response_length_info = {"response_length": data.batch["attention_mask"][:, prompt_length:].sum(dim=-1).cpu().tolist()}

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "thinking_tokens_info": thinking_tokens_info,
                "repetition_info": repetition_info,
                "response_length_info": response_length_info,
            }
        else:
            return reward_tensor
