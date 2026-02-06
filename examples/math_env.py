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

"""Math environment example — run math problems through a Strands agent with a calculator tool.

Usage:
    # SGLang backend (requires a running SGLang server)
    python examples/math_env.py --backend sglang --sglang-base-url http://localhost:30000

    # Bedrock backend (requires AWS credentials)
    python examples/math_env.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
"""

from __future__ import annotations

import asyncio
import logging

import click
from common import ModelConfig, SamplingParams

from strands_env.core.types import Action, TaskContext
from strands_env.environments.calculator.env import CalculatorEnv
from strands_env.rewards.math_reward import MathRewardFunction

MATH_PROBLEMS = [
    ("What is 123 * 456?", "56088"),
    ("What is the square root of 144?", "12"),
    ("What is 2^10?", "1024"),
]


async def run_math_env(config: ModelConfig, verbose: bool) -> None:
    model_factory = config.create_factory()
    env = CalculatorEnv(model_factory=model_factory, reward_fn=MathRewardFunction(), verbose=verbose)

    for question, ground_truth in MATH_PROBLEMS:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Question: {question}")
        click.echo(f"Expected: {ground_truth}")
        click.echo("-" * 60)

        action = Action(message=question, task_context=TaskContext(ground_truth=ground_truth))
        result = await env.step(action)

        click.echo(f"Termination: {result.termination_reason.value}")
        click.echo(f"Response:    {result.observation.final_response}")
        click.echo(f"Reward:      {result.reward.reward if result.reward else None}")
        click.echo(f"Metrics:     {result.observation.metrics}")


@click.command()
@click.option("--backend", required=True, type=click.Choice(["sglang", "bedrock"]), help="Model backend")
@click.option("--model-id", default=None, help="Model ID (auto-detected for SGLang)")
@click.option("--sglang-base-url", default="http://localhost:30000", help="SGLang server URL")
@click.option("--verbose", is_flag=True, help="Print agent callback output")
def main(backend: str, model_id: str | None, sglang_base_url: str, verbose: bool) -> None:
    """Run math problems through a Strands agent with a calculator tool."""
    logging.basicConfig(level=logging.WARNING)

    config = ModelConfig(
        backend=backend,
        model_id=model_id,
        base_url=sglang_base_url,
        sampling_params=SamplingParams(),
    )

    asyncio.run(run_math_env(config, verbose))


if __name__ == "__main__":
    main()
