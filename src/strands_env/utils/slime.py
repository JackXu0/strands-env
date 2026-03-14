# Copyright 2025-2026 Horizon RL Contributors
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

"""Utilities for logging `strands-env` metrics in `slime`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slime.utils.types import Sample  # type: ignore

logger = logging.getLogger(__name__)

# NOTE: response_len in default logging refers to loss_mask=1 tokens


def collect_env_metrics(samples: list[Sample]) -> dict[str, float]:
    """Aggregate strands-env observation metrics across samples.

    Extracts metrics from `sample.metrics` (populated by `Environment.step()`)
    and returns aggregated statistics (mean/median/max/min) suitable for logging.

    Args:
        samples: List of slime `Sample` objects with a `metrics` dict attribute.

    Returns:
        Dict with `{name}_mean`, `{name}_median`, `{name}_max`, `{name}_min`
        for each metric, including per-tool breakdowns.
    """
    from slime.utils.metric_utils import compute_statistics  # type: ignore

    if not samples:
        return {}

    per_sample: dict[str, list[float]] = {
        "message_count": [],
        "tool_iters": [],
        "model_calls": [],
        "model_latency_s": [],
        "cache_hit_rate": [],
    }

    for sample in samples:
        # NOTE: need to set sample.metrics = step_result.observation.metrics in generate()
        metrics: dict[str, Any] = getattr(sample, "metrics", None) or {}
        if not metrics:
            continue

        per_sample["message_count"].append(metrics.get("message_count", 0))
        per_sample["tool_iters"].append(metrics.get("tool_iters", 0))
        per_sample["model_calls"].append(metrics.get("model_calls", 0))
        latency = metrics.get("model_latency_s")
        per_sample["model_latency_s"].append(latency["total"] if latency else 0)
        per_sample["cache_hit_rate"].append(metrics.get("cache_hit_rate") or 0)

        for tool_name, tm in (metrics.get("per_tool_metrics") or {}).items():
            key = f"{tool_name}_tool"
            calls = tm["calls"]
            per_sample.setdefault(f"{key}_calls", []).append(calls)
            per_sample.setdefault(f"{key}_latency_s", []).append(tm["latency_s"])
            per_sample.setdefault(f"{key}_success_rate", []).append(tm["successes"] / calls)
            per_sample.setdefault(f"{key}_parse_error_rate", []).append(tm.get("parse_errors", 0) / calls)

    log_dict: dict[str, float] = {}
    for name, values in per_sample.items():
        log_dict |= {f"{name}_{k}": v for k, v in compute_statistics(values).items()}

    return log_dict


def log_rollout_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    _rollout_time: float,
) -> bool:
    """Custom rollout log function for slime.

    Injects strands-env environment metrics into `rollout_extra_metrics` so they
    are merged into SLiME's single `wandb.log()` call. Returns `False` so slime's
    default logging still runs.

    Args:
        rollout_id: Current rollout iteration number.
        args: slime training arguments.
        samples: List of samples from the rollout.
        rollout_extra_metrics: Additional metrics dict from rollout pipeline — mutated
            in place to include env metrics.
        _rollout_time: Wall-clock time for the rollout (unused).

    Returns:
        `False` to continue with default slime logging.
    """
    from slime.utils.metric_utils import dict_add_prefix  # type: ignore

    log_dict = collect_env_metrics(samples)
    if not log_dict:
        return False

    log_dict = dict_add_prefix(log_dict, "rollout/")

    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(log_dict)
    else:
        logger.warning("rollout_extra_metrics is None, env metrics will not be logged")

    return False
