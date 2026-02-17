Review a pull request against this project's conventions and code quality standards.

The user provides a PR number or URL as $ARGUMENTS. If not provided, ask.

## Steps

1. **Fetch the PR** using `gh pr view <number> --json title,body,additions,deletions,files` and `gh pr diff <number>`.

2. **Review against these criteria**, in order of importance:

### Naming & Design
- Does the environment/module name accurately describe what it does? (e.g., "synthetic_env" is vague — prefer descriptive names like "world_model" or "awm")
- Are class names consistent with existing conventions? (`<Name>Env`, `<Name>RewardFunction`, `<Name>Evaluator`)
- Is the architecture aligned with the project? (Environment subclass, RewardFunction, get_tools/get_hooks pattern)

### Over-engineering
- Are there unnecessary abstractions? (e.g., 7 lazy-loading methods when a simple dict load would do)
- Is the code proportional to the problem? Flag if implementation seems bloated relative to what it does.
- Are there unnecessary files? (plan docs, task files, design docs should NOT be in the PR)
- Is the test-to-source ratio reasonable? Tests significantly exceeding source LOC may indicate over-testing of internals.

### Environment Convention
Every environment under `src/strands_env/environments/<name>/` should have:
- `__init__.py` — re-exports with license header
- `env.py` — Environment subclass
- `system_prompt.md` — agent system prompt
- `requirements.txt` — extra dependencies (or comment if none)
- `README.md` — with Setup, Usage, Tools, Reward, System Prompt sections
Check for missing required files and flag them.

### Code Style
- License headers on all `.py` files (check against existing files for the exact format)
- Line length <= 120 (ruff enforced)
- No leftover debug files, task.md, or personal paths in the code
- Imports follow project conventions (TYPE_CHECKING guard for type-only imports)
- Uses `typing_extensions.override` for method overrides

### CLI Integration
- Benchmark-specific options should NOT be hardcoded into generic CLI commands
- New environments should not modify existing CLI unless absolutely necessary

### Dependency Hygiene
- New pip dependencies should be documented in the environment's `requirements.txt`
- Heavy dependencies (FastAPI, starlette, etc.) should be optional, not core

3. **Output a structured review** with:
   - A one-paragraph summary of what the PR does and overall assessment
   - A categorized list of issues (Critical / Should Fix / Nit)
   - Specific file:line references where possible
   - Suggested alternatives for major issues

Do NOT suggest changes for things that are fine. Focus on actionable feedback.
