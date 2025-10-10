"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to
validate answers when necessary.
"""

from verl.workers.reward_manager.rllm.utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from verl.workers.reward_manager.rllm.reward_types import RewardConfig, RewardOutput, RewardType
import math_verify

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


class RewardMathFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        assert task_info["problem_type"] == RewardType.MATH, "Invalid problem type: expected 'MATH', but got '{}'".format(task_info["problem_type"])

        problem = task_info.get("problem", "")
        model_response = action
        pred = None

        # Handle None or empty response
        if model_response is None or model_response == "":
            print("DEBUG: Empty or None response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[-1]
        else:
            if self.config.apply_format_reward:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            model_solution = model_response

        # Process the ground truth(s)
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            # try:
            #     model_answer = math_verify.parse(model_solution, parsing_timeout=5)
            # except Exception:
            #     return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

        # Check against all possible correct answers
        # if isinstance(model_answer, str):
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True, pred=model_answer)

        if model_answer in processed_ground_truths:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True, pred=model_answer)

        if self.config.use_math_verify:
            # We now fallback to semantic verification
            for ground_truth in processed_ground_truths:
                try:
                    if math_verify.verify(
                        math_verify.parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=5),
                        model_answer,
                        timeout_seconds=5,
                    ):
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True, pred=model_answer)
                except Exception:
                    continue

        # elif isinstance(model_answer, list):
        #     # 0 if parsing is problematic
        #     if len(model_answer) < 2:
        #         return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
        #
        #     # We perform a quick string match first
        #     if model_answer[1] in ground_truths:
        #         return RewardOutput(reward=self.config.correct_reward, is_correct=True, pred=model_answer[1])
        #
        #     # We now fallback to semantic verification
        #     for truth in ground_truths:
        #         try:
        #             if math_verify.verify(
        #                 math_verify.parse(f"\\boxed{{{truth}}}", parsing_timeout=5),
        #                 model_answer,
        #                 timeout_seconds=5,
        #             ):
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True, pred=model_answer[1])
        #         except Exception:
        #             continue

        # # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        # if self.config.use_math_orm:
        #     for ground_truth in processed_ground_truths:
        #         try:
        #             orm_response = call_gemini_llm(
        #                 system_prompt=ORM_PROMPT,
        #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        #                 temperature=0.0,
        #             )
        #
        #             if "[[YES]]" in orm_response:
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        #         except Exception:
        #             print("Error calling Gemini ORM, trying OAI RM")
        #             orm_response = call_oai_rm_llm(
        #                 system_prompt=ORM_PROMPT,
        #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        #                 temperature=0.0,
        #                 model_id=OAI_RM_MODEL,
        #             )
        #
        #             if "[[YES]]" in orm_response:
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        #             continue

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False, pred=model_answer[1] if isinstance(model_answer, list) else model_answer)


def rllm_reward_fn_math(data_source: str, llm_solution: str, ground_truth: str | list[str], extra_info=None, **kwargs):
    """Evaluates mathematical solutions against ground truth answers.

    This function creates a reward function to evaluate mathematical solutions by comparing
    them against provided ground truth answers. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: Either a single string or list of strings containing valid answers
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution is deemed correct, False otherwise

    Example:
        >>> rllm_reward_fn_math("gsm8k", "x = 5", "5", False)
        True
    """
    if extra_info is None:
        extra_info = {}
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)

    # Convert to new format
    task_info = {"problem": None, "problem_type": RewardType.MATH, "data_source": data_source, "ground_truth": ground_truth, **extra_info}

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())
    task_info = {
        "data_source": "",
        "problem": ("Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."),
        "problem_type": RewardType.MATH,
        "ground_truth": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"],
        "has_toolcall": True,
    }
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."

    output = reward(task_info, action)
    print(output)
