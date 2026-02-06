# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluator for running agentic benchmarks with `strands-env` environments."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from functools import partial
from pathlib import Path

from pydantic import BaseModel

from strands_env.core import Action, Environment, StepResult

from .metrics import MetricFn, pass_at_k_metric

logger = logging.getLogger(__name__)

#: Type alias for environment factory function (async).
AsyncEnvFactory = Callable[[Action], Awaitable[Environment]]


class EvalSample(BaseModel):
    """Evaluation sample result."""

    action: Action
    """The action (task) that was evaluated."""

    step_result: StepResult
    """The result of the step (observation, reward, termination reason)."""


class Evaluator:
    """Evaluator for running concurrent environment evaluations."""

    benchmark_name: str = ""
    """Benchmark identifier. Override in subclasses."""

    def __init__(
        self,
        env_factory: AsyncEnvFactory,
        *,
        max_concurrency: int = 10,
        n_rollouts: int = 1,
        output_path: Path | str = Path.cwd() / "results.jsonl",
        save_interval: int = 10,
        keep_tokens: bool = False,
        metric_fns: list[MetricFn] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            env_factory: Async factory function that creates a fresh Environment per sample.
            max_concurrency: Maximum concurrent evaluate_sample() calls.
            n_rollouts: Number of rollouts per problem (for pass@k, set to max(k_values)).
            output_path: Path to JSONL file for saving results. Enables resume.
            save_interval: Flush results to disk every N completed samples.
            keep_tokens: Keep token-level observation in results (only valid for `SGLangModel` backends).
            metric_fns: List of metric functions. Defaults to [pass_at_k_metric].
        """
        self.env_factory: AsyncEnvFactory = env_factory
        self.max_concurrency = max_concurrency
        self.n_rollouts = n_rollouts
        self.output_path = Path(output_path)
        self.save_interval = save_interval
        self.keep_tokens = keep_tokens

        # Default metric: pass@k for k in 1..n_rollouts
        if metric_fns is None:
            metric_fns = [partial(pass_at_k_metric, k_values=list(range(1, n_rollouts + 1)), reward_threshold=1.0)]
        self.metric_fns = metric_fns

        # Runtime state
        self.results: dict[str, list[EvalSample]] = defaultdict(list)
        self.completed_ids: set[str] = set()

    def load_dataset(self) -> Iterable[Action]:
        """Load dataset. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement load_dataset()")

    def load_results(self) -> None:
        """Load completed samples from checkpoint file."""
        if not self.output_path.exists():
            return

        self.results = defaultdict(list)
        self.completed_ids = set()

        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                problem_id = data.pop("problem_id")
                sample = EvalSample.model_validate(data)
                self.results[problem_id].append(sample)
                self.completed_ids.add(sample.action.task_context.id)

        total = sum(len(s) for s in self.results.values())
        logger.info(f"Resumed {total} samples from {self.output_path}")

    def save_results(self) -> None:
        """Save all samples to checkpoint file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for problem_id, samples in self.results.items():
                for sample in samples:
                    data = sample.model_dump()
                    data["problem_id"] = problem_id
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def evaluate_sample(self, action: Action) -> EvalSample:
        """Evaluate a single sample."""
        env = await self.env_factory(action)
        await env.reset()
        step_result = await env.step(action)
        if not self.keep_tokens:
            step_result.observation.tokens = None
        await env.cleanup()
        return EvalSample(action=action, step_result=step_result)

    async def run(self, actions: Iterable[Action]) -> dict[str, list[EvalSample]]:
        """Run evaluation on actions with n_rollouts each.

        Args:
            actions: Actions to evaluate.

        Returns:
            Dict mapping problem_id to list of EvalSample rollouts.
        """
        self.load_results()

        # Expand actions to (problem_id, sample_id, action) tuples
        to_process: list[tuple[str, str, Action]] = []
        for action in actions:
            problem_id = action.task_context.id
            for i in range(self.n_rollouts):
                sample_id = f"{problem_id}_{i}"
                if sample_id not in self.completed_ids:
                    expanded = action.model_copy(deep=True)
                    expanded.task_context.id = sample_id
                    to_process.append((problem_id, sample_id, expanded))

        semaphore = asyncio.Semaphore(self.max_concurrency)
        save_counter = 0
        completed = 0
        total = len(to_process)

        async def process(problem_id: str, sample_id: str, action: Action) -> None:
            nonlocal save_counter, completed
            async with semaphore:
                sample = await self.evaluate_sample(action)
                self.results[problem_id].append(sample)
                self.completed_ids.add(sample_id)
                completed += 1
                save_counter += 1
                if save_counter >= self.save_interval:
                    self.save_results()
                    logger.info(f"Progress: {completed}/{total}")
                    save_counter = 0

        await asyncio.gather(*[process(pid, sid, a) for pid, sid, a in to_process])

        logger.info(f"Completed: {completed}/{total}")
        self.save_results()
        return dict(self.results)

    def compute_metrics(self, results: dict[str, list[EvalSample]]) -> dict[str, float]:
        """Compute all metrics on results.

        Args:
            results: Dict mapping problem_id to sample rollouts.

        Returns:
            Dict mapping metric names to values.
        """
        metrics = {}
        for fn in self.metric_fns:
            metrics.update(fn(results))
        return metrics
