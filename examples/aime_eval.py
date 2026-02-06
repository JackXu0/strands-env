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
    # Chat mode (pure reasoning, no tools)
    python examples/aime_eval.py --backend sglang --env chat

    # Code mode (Python sandbox, requires AWS credentials)
    python examples/aime_eval.py --backend sglang --env code

    # Code mode with assumed role
    python examples/aime_eval.py --backend sglang --env code --role-arn arn:aws:iam::123456789012:role/MyRole

    # Bedrock backend
    python examples/aime_eval.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514 --env code

    # With multiple samples for pass@k
    python examples/aime_eval.py --backend sglang --env chat --n-samples 8
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import boto3
import click
from common import ModelConfig, SamplingParams

from strands_env.core.environment import Environment
from strands_env.environments.code_sandbox import CodeMode, CodeSandboxEnv
from strands_env.eval import AIMEEvaluator
from strands_env.rewards.math_reward import MathRewardFunction
from strands_env.utils.aws import get_assumed_role_session, get_boto3_session

# System prompt for AIME problems - instructs model to box final answer
AIME_SYSTEM_PROMPT = """You are a math problem solver. Solve the given problem step by step.

IMPORTANT: You must wrap your final numerical answer in \\boxed{} format.
For example: \\boxed{42}

AIME problems always have integer answers between 0 and 999 (inclusive).
"""

AIME_CODE_SYSTEM_PROMPT = """You are a math problem solver with access to a Python code interpreter.

You can use the execute_code tool to run Python code to help solve problems.
Use code for calculations, symbolic math, or verification.

IMPORTANT: You must wrap your final numerical answer in \\boxed{} format.
For example: \\boxed{42}

AIME problems always have integer answers between 0 and 999 (inclusive).
"""

EnvMode = Literal["chat", "code"]


async def run_eval(
    config: ModelConfig,
    env_mode: EnvMode,
    boto3_session: boto3.Session | None,
    aime_version: str,
    n_samples_per_prompt: int,
    max_concurrency: int,
    output: str,
) -> None:
    model_factory = config.create_factory()
    reward_fn = MathRewardFunction()

    async def env_factory(_):
        if env_mode == "chat":
            return Environment(
                model_factory=model_factory,
                system_prompt=AIME_SYSTEM_PROMPT,
                reward_fn=reward_fn,
                verbose=False,
            )
        else:  # code
            return CodeSandboxEnv(
                model_factory=model_factory,
                system_prompt=AIME_CODE_SYSTEM_PROMPT,
                reward_fn=reward_fn,
                boto3_session=boto3_session,
                mode=CodeMode.CODE,
                verbose=False,
            )

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
@click.option(
    "--env",
    "env_mode",
    default="chat",
    type=click.Choice(["chat", "code"]),
    help="Environment mode: chat (pure reasoning) or code (Python sandbox)",
)
@click.option("--role-arn", default=None, help="AWS IAM role ARN to assume (for code mode)")
@click.option("--aime-version", default="2024", type=click.Choice(["2024", "2025"]), help="AIME dataset version")
@click.option("--n-samples-per-prompt", default=8, type=int, help="Number of samples per prompt")
@click.option("--max-concurrency", default=30, type=int, help="Max concurrent evaluations")
@click.option("--output", default="aime_results.jsonl", help="Output file for results")
def main(
    backend: str,
    model_id: str | None,
    sglang_base_url: str,
    env_mode: str,
    role_arn: str | None,
    aime_version: str,
    n_samples_per_prompt: int,
    max_concurrency: int,
    output: str,
) -> None:
    """Run pass@k evaluation on AIME math problems."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("strands_env").setLevel(logging.INFO)

    # Get AWS session for code mode
    boto3_session = None
    if env_mode == "code":
        boto3_session = get_assumed_role_session(role_arn) if role_arn else get_boto3_session()
        # Validate credentials
        arn = boto3_session.client("sts").get_caller_identity()["Arn"]
        click.echo(f"Using AWS identity: {arn}")

    config = ModelConfig(
        backend=backend,
        model_id=model_id,
        base_url=sglang_base_url,
        sampling_params=SamplingParams(),
    )

    asyncio.run(
        run_eval(
            config=config,
            env_mode=env_mode,
            boto3_session=boto3_session,
            aime_version=aime_version,
            n_samples_per_prompt=n_samples_per_prompt,
            max_concurrency=max_concurrency,
            output=output,
        )
    )


if __name__ == "__main__":
    main()
