HARDCORE RULES:

- You use `just` as command runner and `uv` for running the app and deps
- After making any changes, run the build and linter/formatter scripts to ensure everything is working and formatted correctly.
- **UV-NATIVE DEVELOPMENT**: This project uses UV for local development (NO Docker locally). This is an exception to the standard docker-env rule.
  - Local dev: `just dev` runs uvicorn directly via `uv run`
  - Docker is ONLY used for Cloud Run deployment (built in CI)
  - Rationale: Faster iteration (uv run is instant), simpler setup for Python ML projects
- After code changes that affect the running app, verify the app works by running `just dev` and testing endpoints
- When installing a package, or adding a docker image or ... you should always verify that the tag/version you are using is the latest unless otherwise instructed. Never use 'latest' of something
- Whenever adding functionality, you should consider whether it makes sense to add/modify the config and allow controlling config via env vars
- There is a rich ecosystem of library/helper type stuff available in this repo. Look to see if there's any existing helpers available before re-implementing things
- You should work TDD-style whenever possible and appropriate
- This app is horizontally scalable, stateless and follows all the 12 factors
- You have access to several MCP servers. Use them a lot to verify your theories

## ML Service Specific Guidelines

### Testing Strategy
- **Free tests** (marked with `@pytest.mark.free`): Run without API costs, use EmbeddingGemma (local model). Run in CI on every commit.
- **Paid tests** (marked with `@pytest.mark.paid`): Use Gemini API, cost money. Run manually before releases via GitHub Actions workflow.
- Use `just test` for free tests, `just test-paid` for paid tests (requires GOOGLE_API_KEY)

### Package Management
- This project uses **google-genai** (not google-generativeai which is deprecated as of 2025)
- All dependencies in pyproject.toml are pinned to latest stable versions as of October 2025
