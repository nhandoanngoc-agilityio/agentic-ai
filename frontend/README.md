# Comparison UI

Next.js + CopilotKit frontend for the streaming single-provider run. See the root `README.md`'s
"Streaming comparison UI" section for the full three-process setup — this file covers only what's
specific to this app.

## Setup

```bash
cp .env.local.example .env.local
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Environment variables (`.env.local`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_DEPLOYMENT_URL` | `http://localhost:2024` | Where `LLM_PROVIDER=anthropic langgraph dev` is listening |
| `OPENAI_DEPLOYMENT_URL` | `http://localhost:2025` | Where `LLM_PROVIDER=openai langgraph dev` is listening |

The dropdown's "Anthropic" / "OpenAI" labels are client-side strings — they say which *port*
gets called, not which provider is actually running on it. A `LLM_PROVIDER=` line left in the
repo root's `.env` overrides the one exported on each `langgraph dev` command line, so both ports
can end up on the same provider while the UI still looks right. See the root `README.md`'s
"Streaming comparison UI" section before starting the dev servers.

## Scripts

- `npm run dev` — start the Next.js dev server
- `npm run build` — production build (also the fastest way to typecheck the whole app)
- `npm run test` — run the Vitest component test suite
- `npm run lint` — ESLint
