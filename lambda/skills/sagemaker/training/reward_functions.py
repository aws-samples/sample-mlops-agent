"""Pre-approved reward functions for GRPO training.

To add a new reward function: implement a factory here that accepts (cfg, tokenizer)
and add its name to REGISTRY. Never accept reward function code as a string from
external config — that is arbitrary code execution.

Usage in training config JSON:
    {"reward_function": "length_penalty", "reward": {"target_length": 100}}
"""

from __future__ import annotations

from typing import Callable


def length_penalty(cfg: dict, tokenizer) -> Callable[[list[str]], list[float]]:
    """Penalise completions that deviate from a target token length.

    Args:
        cfg: Full training config dict. Must contain cfg["reward"]["target_length"] (int).
        tokenizer: HuggingFace tokenizer used to count tokens.

    Returns:
        Reward function callable: (completions: list[str], **kw) -> list[float]
    """
    target: int = cfg["reward"]["target_length"]

    def _fn(completions: list[str], **kw) -> list[float]:
        return [
            -(len(tokenizer.encode(c, add_special_tokens=False)) - target) ** 2 / 1000
            for c in completions
        ]

    return _fn


# Registry maps config string → factory function.
# Add new entries here as new reward functions are approved.
REGISTRY: dict[str, Callable] = {
    "length_penalty": length_penalty,
}


def get_reward_fn(name: str, cfg: dict, tokenizer) -> Callable[[list[str]], list[float]]:
    """Return an instantiated reward function by name.

    Args:
        name: Key in REGISTRY (e.g. "length_penalty").
        cfg: Full training config dict passed to the factory.
        tokenizer: HuggingFace tokenizer passed to the factory.

    Returns:
        Callable reward function ready to pass to GRPOTrainer.

    Raises:
        ValueError: If name is not in REGISTRY.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown reward function {name!r}. "
            f"Allowed values: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](cfg, tokenizer)
