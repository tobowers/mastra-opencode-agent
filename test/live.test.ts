import { describe, expect, it } from "bun:test";

import { OpenCodeSDKAgent } from "../src/index.js";

const runLive = process.env.LIVE_OPENCODE_TEST === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const liveTest = runLive ? it : it.skip;

describe("OpenCodeSDKAgent live", () => {
  liveTest("runs a real OpenCode request through Mastra generate()", async () => {
    const model = process.env.OPENCODE_MODEL ?? "openrouter/google/gemini-2.5-flash-lite";
    const providerID = model.slice(0, model.indexOf("/"));
    const agent = new OpenCodeSDKAgent({
      id: "opencode-live",
      description: "Live OpenCode test",
      sdkOptions: {
        model,
        sessionTitle: "mastra-opencode live test",
        config: {
          model,
          provider: {
            [providerID]: {
              options: {
                apiKey: "{env:OPENROUTER_API_KEY}",
              },
            },
          },
          disabled_providers: [],
        } as never,
      },
    });

    try {
      const result = await agent.generate('Reply with exactly: "mastra-opencode-ok". Do not edit files.', {
        sdkOptions: { tools: { write: false, edit: false, bash: false } },
      });

      expect(providerID.length).toBeGreaterThan(0);
      expect(result.text.toLowerCase()).toContain("mastra-opencode-ok");
    } finally {
      await agent.close();
    }
  });
});
