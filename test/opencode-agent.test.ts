import { describe, expect, it } from "bun:test";
import { Mastra } from "@mastra/core/mastra";
import { SpanType } from "@mastra/core/observability";

import { OpenCodeSDKAgent, __testing } from "../src/index.js";
import type { OpenCodeInstance } from "../src/index.js";

function createSession(overrides: Record<string, unknown> = {}) {
  return {
    id: "session-1",
    projectID: "project-1",
    directory: process.cwd(),
    title: "Test session",
    version: "1.0.0",
    time: { created: Date.now(), updated: Date.now() },
    ...overrides,
  };
}

function createAssistant(overrides: Record<string, unknown> = {}) {
  return {
    id: "assistant-1",
    sessionID: "session-1",
    role: "assistant",
    parentID: "user-1",
    modelID: "gemini-2.5-flash-lite",
    providerID: "google",
    mode: "build",
    path: { cwd: process.cwd(), root: process.cwd() },
    cost: 0.001,
    tokens: { input: 3, output: 5, reasoning: 0, cache: { read: 1, write: 2 } },
    time: { created: Date.now(), completed: Date.now() },
    finish: "stop",
    ...overrides,
  };
}

function createTextPart(text: string, overrides: Record<string, unknown> = {}) {
  return {
    id: "part-1",
    sessionID: "session-1",
    messageID: "assistant-1",
    type: "text",
    text,
    ...overrides,
  };
}

async function* streamEvents(events: unknown[]) {
  for (const event of events) yield event;
}

function createMockClient(events: unknown[] = []) {
  const calls: Array<{ method: string; options: unknown }> = [];
  const session = createSession();
  const assistant = createAssistant();
  const parts = [createTextPart("hello from opencode")];

  const client = {
    session: {
      create: async (options: unknown) => {
        calls.push({ method: "session.create", options });
        return { data: session, error: undefined };
      },
      get: async (options: unknown) => {
        calls.push({ method: "session.get", options });
        return { data: session, error: undefined };
      },
      prompt: async (options: unknown) => {
        calls.push({ method: "session.prompt", options });
        return { data: { info: assistant, parts }, error: undefined };
      },
      promptAsync: async (options: unknown) => {
        calls.push({ method: "session.promptAsync", options });
        return { data: undefined, error: undefined };
      },
      messages: async (options: unknown) => {
        calls.push({ method: "session.messages", options });
        return { data: [{ info: assistant, parts }], error: undefined };
      },
    },
    event: {
      subscribe: async (options: unknown) => {
        calls.push({ method: "event.subscribe", options });
        return { stream: streamEvents(events) };
      },
    },
  };

  return { client: client as never, calls, assistant, session };
}

function createMockSpan(input: { id: string; traceId: string; type?: SpanType; records: Array<{ action: string; span: string; value?: unknown }> }) {
  const span = {
    id: input.id,
    traceId: input.traceId,
    name: input.id,
    type: input.type ?? SpanType.GENERIC,
    startTime: new Date(),
    isEvent: false,
    isInternal: false,
    observabilityInstance: {},
    get isRootSpan() {
      return input.id === "root";
    },
    get isValid() {
      return true;
    },
    get externalTraceId() {
      return input.traceId;
    },
    end: (value?: unknown) => input.records.push({ action: "end", span: input.id, value }),
    error: (value: unknown) => input.records.push({ action: "error", span: input.id, value }),
    update: (value: unknown) => input.records.push({ action: "update", span: input.id, value }),
    createChildSpan: (options: { type: SpanType; name: string }) => {
      input.records.push({ action: "create", span: input.id, value: options });
      const childId = options.type === SpanType.AGENT_RUN ? "agent-span" : "model-span";
      return createMockSpan({ id: childId, traceId: input.traceId, type: options.type, records: input.records });
    },
    createEventSpan: (options: { type: SpanType; name: string }) => {
      input.records.push({ action: "event", span: input.id, value: options });
      return createMockSpan({ id: "event-span", traceId: input.traceId, type: options.type, records: input.records });
    },
    getParentSpanId: () => undefined,
    findParent: () => undefined,
    exportSpan: () => ({
      id: input.id,
      traceId: input.traceId,
      name: input.id,
      type: input.type ?? SpanType.GENERIC,
      startTime: new Date(),
      isRootSpan: input.id === "root",
      isEvent: false,
      isInternal: false,
    }),
    executeInContext: async <T,>(fn: () => Promise<T>) => fn(),
    executeInContextSync: <T,>(fn: () => T) => fn(),
  };
  return span as never;
}

describe("OpenCodeSDKAgent", () => {
  it("maps generate() to an OpenCode session prompt and returns Mastra output", async () => {
    const { client, calls } = createMockClient();
    const agent = new OpenCodeSDKAgent({
      id: "opencode",
      description: "OpenCode through Mastra",
      client,
      sdkOptions: {
        directory: "/repo",
        model: "google/gemini-2.5-flash-lite",
        agent: "build",
        tools: { write: false },
      },
    });

    const result = await agent.generate("say hi", { sdkOptions: { messageId: "user-1" } });

    expect(result.text).toBe("hello from opencode");
    expect(result.response.modelId).toBe("google/gemini-2.5-flash-lite");
    expect(result.usage.inputTokens).toBe(6);
    expect(result.usage.outputTokens).toBe(5);
    expect(calls.map((call) => call.method)).toEqual(["session.create", "session.prompt"]);
    expect(calls[1]?.options).toMatchObject({
      path: { id: "session-1" },
      query: { directory: "/repo" },
      body: {
        messageID: "user-1",
        model: { providerID: "google", modelID: "gemini-2.5-flash-lite" },
        agent: "build",
        tools: { write: false },
        parts: [{ type: "text", text: "say hi" }],
      },
    });
  });

  it("streams OpenCode event deltas and exposes textStream", async () => {
    const assistant = createAssistant();
    const events = [
      { type: "message.updated", properties: { info: assistant } },
      { type: "message.part.updated", properties: { part: createTextPart("hello", { id: "part-a" }), delta: "hello" } },
      { type: "message.part.updated", properties: { part: createTextPart(" world", { id: "part-b" }), delta: " world" } },
      { type: "session.idle", properties: { sessionID: "session-1" } },
    ];
    const { client, calls } = createMockClient(events);
    const agent = new OpenCodeSDKAgent({ id: "opencode", description: "OpenCode", client });

    const stream = await agent.stream("stream please", { sdkOptions: { messageId: "user-1" } });
    const chunks: string[] = [];
    for await (const chunk of stream.textStream) chunks.push(chunk);

    expect(chunks.join("")).toBe("hello world");
    expect(await stream.text).toBe("hello world");
    expect(calls.map((call) => call.method)).toEqual(["session.create", "event.subscribe", "session.promptAsync", "session.messages"]);
  });

  it("falls back to final session messages when stream deltas are missing", async () => {
    const events = [{ type: "session.idle", properties: { sessionID: "session-1" } }];
    const { client } = createMockClient(events);
    const agent = new OpenCodeSDKAgent({ id: "opencode", description: "OpenCode", client });

    const stream = await agent.stream("stream please", { sdkOptions: { messageId: "user-1" } });

    expect(await stream.text).toBe("hello from opencode");
  });

  it("resumes an existing OpenCode session", async () => {
    const { client, calls } = createMockClient();
    const agent = new OpenCodeSDKAgent({ id: "opencode", description: "OpenCode", client });

    await agent.resumeGenerate({ message: "continue", sessionId: "session-1", directory: "/repo" }, { sdkOptions: { messageId: "user-1" } });

    expect(calls.map((call) => call.method)).toEqual(["session.get", "session.prompt"]);
    expect(calls[0]?.options).toMatchObject({ path: { id: "session-1" }, query: { directory: "/repo" } });
  });

  it("supports factory lifecycle and close()", async () => {
    let closed = false;
    const { client } = createMockClient();
    const opencode: OpenCodeInstance = { client, server: { url: "http://localhost:4096", close: () => (closed = true) } };
    const agent = new OpenCodeSDKAgent({ id: "opencode", description: "OpenCode", opencode: async () => opencode });

    await agent.generate("hello", { sdkOptions: { messageId: "user-1" } });
    await agent.close();

    expect(closed).toBe(true);
  });

  it("registers and runs through the Mastra agent registry", async () => {
    const assistant = createAssistant();
    const events = [
      { type: "message.updated", properties: { info: assistant } },
      { type: "message.part.updated", properties: { part: createTextPart("hello", { id: "part-a" }), delta: "hello" } },
      { type: "message.part.updated", properties: { part: createTextPart(" mastra", { id: "part-b" }), delta: " mastra" } },
      { type: "session.idle", properties: { sessionID: "session-1" } },
    ];
    const { client } = createMockClient(events);
    const opencodeAgent = new OpenCodeSDKAgent({ id: "opencode-sdk", description: "OpenCode", client });
    const mastra = new Mastra({ agents: { opencodeAgent } });

    const byName = mastra.getAgent("opencodeAgent");
    const byId = mastra.getAgentById("opencode-sdk");

    expect(byName).toBe(opencodeAgent);
    expect(byId).toBe(opencodeAgent);
    expect(byName.supportsMemory()).toBe(false);

    const result = await byName.generate("say hi", { sdkOptions: { messageId: "user-1" } });
    expect(result.text).toBe("hello from opencode");

    const stream = await byName.stream("stream please", { sdkOptions: { messageId: "user-1" } });
    expect(await stream.text).toBe("hello mastra");
  });

  it("creates Mastra observability spans and propagates tracing context", async () => {
    const records: Array<{ action: string; span: string; value?: unknown }> = [];
    const root = createMockSpan({ id: "root", traceId: "1234567890abcdef1234567890abcdef", records });
    const { client } = createMockClient();
    const agent = new OpenCodeSDKAgent({ id: "opencode", description: "OpenCode", client });

    const result = await agent.generate("trace this", {
      sdkOptions: { messageId: "user-1" },
      tracingContext: { currentSpan: root },
    } as never);

    expect(result.text).toBe("hello from opencode");
    expect(records).toContainEqual(expect.objectContaining({ action: "create", span: "root", value: expect.objectContaining({ type: SpanType.AGENT_RUN }) }));
    expect(records).toContainEqual(
      expect.objectContaining({ action: "create", span: "agent-span", value: expect.objectContaining({ type: SpanType.MODEL_GENERATION }) }),
    );
    expect(records).toContainEqual(expect.objectContaining({ action: "end", span: "model-span" }));
    expect(records).toContainEqual(expect.objectContaining({ action: "end", span: "agent-span" }));
    expect(result.traceId).toBe("1234567890abcdef1234567890abcdef");
    expect(result.spanId).toBe("agent-span");
  });
});

describe("OpenCodeSDKAgent helpers", () => {
  it("parses provider/model strings and merges tool overrides", () => {
    expect(__testing.toModelSelector("google/gemini-2.5-flash-lite")).toEqual({
      providerID: "google",
      modelID: "gemini-2.5-flash-lite",
    });
    expect(
      __testing.mergeRunOptions({ tools: { write: false, bash: false } }, { tools: { bash: true }, sessionId: "session-1" }),
    ).toEqual({ tools: { write: false, bash: true }, sessionId: "session-1" });
  });

  it("omits undefined server options so OpenCode SDK defaults survive", () => {
    expect(__testing.toServerOptions({ model: "google/gemini-2.5-flash-lite", config: { model: "x" } })).toEqual({
      config: { model: "x" },
    });
  });

  it("adds JSON schema format to prompt bodies for OpenCode structured output", () => {
    const body = __testing.createPromptBody("return json", { model: "google/gemini-2.5-flash-lite" }, {
      structuredOutput: {
        schema: {
          type: "object",
          properties: { answer: { type: "string" } },
          required: ["answer"],
        },
      },
    } as never);

    expect(body).toMatchObject({
      format: {
        type: "json_schema",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
      },
    });
  });
});
