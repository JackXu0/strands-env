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

"""AIME evaluation example — run pass@k evaluation on AIME math problems.

Usage:
    # SGLang backend (requires a running SGLang server)
    python examples/aime_eval.py --backend sglang --sglang-base-url http://localhost:30000

    # Bedrock backend (requires AWS credentials)
    python examples/aime_eval.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514

    # With multiple samples for pass@k
    python examples/aime_eval.py --backend sglang --n-samples 8
"""

from __future__ import annotations

import asyncio
import logging

import click
from common import ModelConfig, SamplingParams

from strands_env.environments.simple_math_env import SimpleMathEnv
from strands_env.eval import AIMEEvaluator
from strands_env.rewards.math_reward import MathRewardFunction


async def run_eval(
    config: ModelConfig,
    aime_version: str,
    n_samples_per_prompt: int,
    max_concurrency: int,
    output: str,
) -> None:
    model_factory = config.create_factory()
    reward_fn = MathRewardFunction()

    async def env_factory(_):
        env = SimpleMathEnv(model_factory=model_factory, reward_fn=reward_fn, verbose=False)
        env.system_prompt = None
        env.get_tools = lambda: []
        return env

    evaluator = AIMEEvaluator(
        env_factory=env_factory,
        n_samples_per_prompt=n_samples_per_prompt,
        max_concurrency=max_concurrency,
        output_path=output,
    )

    actions = evaluator.load_dataset(aime_version)
    results = await evaluator.run(actions)
    evaluator.compute_metrics(results)

    click.echo(f"\nResults saved to: {output}")


@click.command()
@click.option("--backend", required=True, type=click.Choice(["sglang", "bedrock"]), help="Model backend")
@click.option("--model-id", default=None, help="Model ID (auto-detected for SGLang)")
@click.option("--sglang-base-url", default="http://localhost:30000", help="SGLang server URL")
@click.option("--aime-version", default="2024", type=click.Choice(["2024", "2025"]), help="AIME dataset version")
@click.option("--n-samples", default=8, type=int, help="Number of samples per prompt")
@click.option("--max-concurrency", default=30, type=int, help="Max concurrent evaluations")
@click.option("--output", default="aime_results.jsonl", help="Output file for results")
def main(
    backend: str,
    model_id: str | None,
    sglang_base_url: str,
    aime_version: str,
    n_samples_per_prompt: int,
    max_concurrency: int,
    output: str,
) -> None:
    """Run pass@k evaluation on AIME math problems."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("strands_env").setLevel(logging.INFO)

    config = ModelConfig(
        backend=backend,
        model_id=model_id,
        base_url=sglang_base_url,
        sampling_params=SamplingParams(),
    )

    asyncio.run(
        run_eval(
            config=config,
            aime_version=aime_version,
            n_samples_per_prompt=n_samples_per_prompt,
            max_concurrency=max_concurrency,
            output=output,
        )
    )


if __name__ == "__main__":
    main()
