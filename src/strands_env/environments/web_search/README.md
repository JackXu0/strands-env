# Web Search Environment

A web search environment that gives the agent search and optional web scraping tools. Supports Serper and Google Custom Search providers.

## Setup

Set API credentials as environment variables depending on your search provider:

```bash
# Serper (default)
export SERPER_API_KEY="your-key"

# Google Custom Search
export GOOGLE_API_KEY="your-key"
export GOOGLE_CSE_ID="your-cse-id"
```

## Usage

```python
from strands_env.environments.web_search import WebSearchEnv, SearchConfig, ScrapeConfig

# Search only (default)
env = WebSearchEnv(model_factory=model_factory)

# Search + scrape
env = WebSearchEnv(
    model_factory=model_factory,
    scrape_config=ScrapeConfig(),
)

# Search + scrape with LLM summarization
env = WebSearchEnv(
    model_factory=model_factory,
    scrape_config=ScrapeConfig(summarizer_model_factory=summarizer_factory),
)

result = await env.step(action)
await env.cleanup()  # Close HTTP sessions
```

## Tools

Depends on configuration:

| Config | Tools |
|---|---|
| Default | `serper_search` |
| `provider="google"` | `google_search` |
| `+ ScrapeConfig()` | search + `scrape` |
| `+ ScrapeConfig(summarizer_model_factory=...)` | search + `scrape_and_summarize` |

## Configuration

- **`SearchConfig`** — `provider` (`"serper"` or `"google"`), `timeout`, `max_concurrency`, `blocked_domains`
- **`ScrapeConfig`** — `token_budget` (default 5000 tokens), `timeout`, `max_concurrency`, `summarizer_model_factory`

## Reward

No built-in reward function. Supply a custom `reward_fn`.

## System Prompt

The agent is instructed to search the web, optionally scrape pages for detail, and synthesize findings into a clear, sourced answer.
