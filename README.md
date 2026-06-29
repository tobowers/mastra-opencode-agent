# mastra-opencode-agent

Use OpenCode as a [Mastra SDK agent](https://mastra.ai/docs/agents/sdk-agents).

This package wraps the OpenCode JS SDK in a Mastra-compatible agent with `generate()`, `stream()`, `resumeGenerate()`, and `resumeStream()`. OpenCode still owns the runtime, tools, permissions, model selection, and session loop.

## Install

```bash
npm install mastra-opencode-agent @mastra/core @opencode-ai/sdk
```

```bash
pnpm add mastra-opencode-agent @mastra/core @opencode-ai/sdk
```

```bash
bun add mastra-opencode-agent @mastra/core @opencode-ai/sdk
```

## Create An Agent

```ts
import { OpenCodeSDKAgent } from "mastra-opencode-agent";

export const opencodeAgent = new OpenCodeSDKAgent({
  id: "opencode-agent",
  name: "OpenCode Agent",
  description: "Use OpenCode through Mastra.",
  sdkOptions: {
    model: "openrouter/google/gemini-2.5-flash-lite",
    directory: process.cwd(),
    config: {
      model: "openrouter/google/gemini-2.5-flash-lite",
      provider: {
        openrouter: {
          options: {
            apiKey: "{env:OPENROUTER_API_KEY}",
          },
        },
      },
    },
  },
});
```

Register it with Mastra like any other agent:

```ts
import { Mastra } from "@mastra/core";
import { opencodeAgent } from "./agents/opencode-agent";

export const mastra = new Mastra({
  agents: {
    opencodeAgent,
  },
});
```

## Generate

```ts
const result = await opencodeAgent.generate("Explain this repo's test setup.", {
  sdkOptions: {
    tools: {
      write: false,
      edit: false,
    },
  },
});

console.log(result.text);
```

## Stream

```ts
const stream = await opencodeAgent.stream("Inspect this project and summarize it.");

for await (const chunk of stream.textStream) {
  process.stdout.write(chunk);
}
```

## Resume A Session

```ts
const result = await opencodeAgent.resumeGenerate({
  sessionId: "ses_123",
  message: "Continue from the previous result.",
});

console.log(result.text);
```

## Use An Existing OpenCode Client

Pass a client if your app already manages the OpenCode server lifecycle:

```ts
import { createOpencode } from "@opencode-ai/sdk";
import { OpenCodeSDKAgent } from "mastra-opencode-agent";

const opencode = await createOpencode();

const agent = new OpenCodeSDKAgent({
  id: "opencode-agent",
  description: "OpenCode through Mastra.",
  client: opencode.client,
});

// Later, when your app shuts down:
opencode.server.close();
```

Or pass an OpenCode instance/factory and let the wrapper close it:

```ts
const agent = new OpenCodeSDKAgent({
  id: "opencode-agent",
  description: "OpenCode through Mastra.",
  opencode: () => createOpencode({ timeout: 15_000 }),
});

await agent.close();
```

## Options

`sdkOptions` forwards server options to `createOpencode()` and adds prompt/session defaults:

- `hostname`, `port`, `signal`, `timeout`, `config`: OpenCode server options.
- `model`: string like `provider/model` or `{ providerID, modelID }`.
- `directory`: OpenCode directory query for session APIs.
- `sessionId`: reuse an existing OpenCode session.
- `sessionTitle`: title for newly created sessions.
- `agent`: OpenCode agent name.
- `system`: OpenCode system prompt override.
- `tools`: OpenCode tool enablement map.

Per-call `sdkOptions` override constructor defaults.

## Structured Output

Mastra `structuredOutput` is converted to OpenCode's documented JSON schema format and sent as `body.format`:

```ts
const result = await opencodeAgent.generate<{ answer: string }>("Return JSON only.", {
  structuredOutput: {
    schema: {
      type: "object",
      properties: { answer: { type: "string" } },
      required: ["answer"],
    },
  },
});

console.log(result.object.answer);
```

The current published OpenCode SDK type definitions do not include `format`, so this package sends the documented field via a typed extension.

## Testing

Run deterministic tests:

```bash
bun test
```

Run typecheck:

```bash
bun run typecheck
```

Run the optional live test:

```bash
OPENROUTER_API_KEY=... LIVE_OPENCODE_TEST=1 bun test test/live.test.ts --timeout 120000
```

The live test defaults to `openrouter/google/gemini-2.5-flash-lite`. Override with `OPENCODE_MODEL` if needed.

## Notes

- `supportsMemory()` returns `false` because OpenCode owns session state.
- `stream()` uses OpenCode `event.subscribe()` plus `session.promptAsync()` and falls back to `session.messages()` for final text.
- Mastra observability is wired for SDK-agent runs: `AGENT_RUN` and `MODEL_GENERATION` spans are created, `tracingContext` is propagated, and outputs expose `traceId`/`spanId` when tracing is enabled.
- `close()` only closes a server started through this wrapper's `opencode` factory/default path. It does not close a client you pass directly.
