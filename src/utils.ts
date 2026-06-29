import { randomUUID } from "node:crypto";
import { ReadableStream } from "node:stream/web";

import type { AgentExecutionOptionsBase, MastraLanguageModel, PublicStructuredOutputOptions } from "@mastra/core/agent";
import { MessageList } from "@mastra/core/agent/message-list";
import type { MessageListInput } from "@mastra/core/agent/message-list";
import { standardSchemaToJSONSchema, toStandardSchema } from "@mastra/core/schema";
import { ChunkFrom, MastraModelOutput } from "@mastra/core/stream";
import type { ChunkType, FullOutput, LanguageModelUsage, ProviderMetadata } from "@mastra/core/stream";
import type { AssistantMessage, Part } from "@opencode-ai/sdk";

export type SDKAgentRunOptions<OUTPUT = unknown> = AgentExecutionOptionsBase<OUTPUT> & {
  signal?: AbortSignal;
  structuredOutput?: OUTPUT extends object ? PublicStructuredOutputOptions<OUTPUT> : never;
  [key: string]: unknown;
};

export type V3Usage = {
  inputTokens: {
    total: number | undefined;
    noCache?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  outputTokens: {
    total: number | undefined;
    text?: number;
    reasoning?: number;
  };
};

export type SDKModelGenerateResult = {
  content: Array<{ type: "text"; text: string }>;
  finishReason: { unified: "stop" | "error" | "abort" | "unknown"; raw: string };
  usage: V3Usage;
  response: {
    id: string;
    modelId: string;
    timestamp: Date;
  };
  providerMetadata?: ProviderMetadata;
  object?: unknown;
};

type MastraModelOutputOptions<OUTPUT = undefined> = ConstructorParameters<typeof MastraModelOutput<OUTPUT>>[0]["options"];

export function createNoopModel({ modelId, provider }: { modelId: string; provider: string }): MastraLanguageModel {
  return {
    modelId,
    provider,
    specificationVersion: "v3",
    supportedUrls: {},
    doGenerate: async () => createNoopStreamResult(),
    doStream: async () => createNoopStreamResult(),
  } as MastraLanguageModel;
}

function createNoopStreamResult() {
  return {
    stream: new ReadableStream<ChunkType>({
      start: (controller) => controller.close(),
    }),
  };
}

export function createMastraOutput<OUTPUT>({
  messages,
  runId,
  modelId,
  provider,
  stream,
  responseText = "",
  options,
}: {
  messages: MessageListInput;
  runId: string;
  modelId: string;
  provider: string;
  stream: ReadableStream<ChunkType<OUTPUT>>;
  responseText?: string;
  options?: Partial<MastraModelOutputOptions<OUTPUT>>;
}): MastraModelOutput<OUTPUT> {
  const messageList = new MessageList();
  messageList.add(messages, "input");
  messageList.add([{ role: "assistant", content: responseText }], "response");

  return new MastraModelOutput({
    model: { modelId, provider, version: "v3" },
    stream,
    messageList,
    messageId: randomUUID(),
    options: { ...options, runId } as MastraModelOutputOptions<OUTPUT>,
  });
}

export async function toFullOutput<OUTPUT>({
  messages,
  runId,
  provider,
  result,
  options,
}: {
  messages: MessageListInput;
  runId: string;
  provider: string;
  result: SDKModelGenerateResult;
  options?: Partial<MastraModelOutputOptions<OUTPUT>>;
}): Promise<FullOutput<OUTPUT>> {
  const text = result.content.map((part) => part.text).join("");
  const stream = createCompletedMastraStream<OUTPUT>({
    runId,
    prompt: promptToText(messages),
    text,
    responseId: result.response.id,
    modelId: result.response.modelId,
    usage: toLanguageModelUsage(result.usage),
    providerMetadata: result.providerMetadata,
    object: result.object,
  });

  return createMastraOutput({
    messages,
    runId,
    modelId: result.response.modelId,
    provider,
    stream,
    responseText: text,
    options,
  }).getFullOutput();
}

export function createCompletedMastraStream<OUTPUT>({
  runId,
  prompt,
  text,
  responseId,
  modelId,
  usage,
  providerMetadata,
  object,
}: {
  runId: string;
  prompt: string;
  text: string;
  responseId: string;
  modelId: string;
  usage: LanguageModelUsage;
  providerMetadata?: ProviderMetadata;
  object?: unknown;
}): ReadableStream<ChunkType<OUTPUT>> {
  return new ReadableStream({
    start(controller) {
      const textId = randomUUID();
      enqueueStartChunks(controller, { runId, prompt, textId, responseId, modelId, providerMetadata });
      if (text) enqueueTextDelta(controller, runId, textId, text, providerMetadata);
      enqueueFinishChunks(controller, { runId, prompt, textId, text, responseId, modelId, usage, providerMetadata, object });
      controller.close();
    },
  });
}

export function enqueueStartChunks<OUTPUT>(
  controller: ReadableStreamDefaultController<ChunkType<OUTPUT>>,
  input: {
    runId: string;
    prompt: string;
    textId: string;
    responseId?: string;
    modelId: string;
    providerMetadata?: ProviderMetadata;
  },
): void {
  controller.enqueue({ type: "start", runId: input.runId, from: ChunkFrom.AGENT, payload: {} });
  controller.enqueue({
    type: "step-start",
    runId: input.runId,
    from: ChunkFrom.AGENT,
    payload: { request: { body: input.prompt } },
  });
  controller.enqueue({
    type: "response-metadata",
    runId: input.runId,
    from: ChunkFrom.AGENT,
    payload: { id: input.responseId, modelId: input.modelId, timestamp: new Date().toISOString() },
  });
  controller.enqueue({
    type: "text-start",
    runId: input.runId,
    from: ChunkFrom.AGENT,
    payload: { id: input.textId, providerMetadata: input.providerMetadata },
  });
}

export function enqueueTextDelta<OUTPUT>(
  controller: ReadableStreamDefaultController<ChunkType<OUTPUT>>,
  runId: string,
  textId: string,
  text: string,
  providerMetadata?: ProviderMetadata,
): void {
  controller.enqueue({ type: "text-delta", runId, from: ChunkFrom.AGENT, payload: { id: textId, text, providerMetadata } });
}

export function enqueueFinishChunks<OUTPUT>(
  controller: ReadableStreamDefaultController<ChunkType<OUTPUT>>,
  input: {
    runId: string;
    prompt: string;
    textId: string;
    text: string;
    responseId?: string;
    modelId: string;
    usage: LanguageModelUsage;
    providerMetadata?: ProviderMetadata;
    object?: unknown;
  },
): void {
  const timestamp = new Date();
  const response = { id: input.responseId, modelId: input.modelId, timestamp };
  const metadata = { providerMetadata: input.providerMetadata, request: { body: input.prompt }, modelId: input.modelId, timestamp };

  controller.enqueue({
    type: "text-end",
    runId: input.runId,
    from: ChunkFrom.AGENT,
    payload: { id: input.textId, providerMetadata: input.providerMetadata },
  });
  if (input.object !== undefined) {
    controller.enqueue({ type: "object-result", runId: input.runId, from: ChunkFrom.AGENT, object: input.object } as ChunkType<OUTPUT>);
  }
  controller.enqueue({
    type: "step-finish",
    runId: input.runId,
    from: ChunkFrom.AGENT,
    payload: {
      id: input.responseId,
      providerMetadata: input.providerMetadata,
      totalUsage: input.usage,
      response,
      stepResult: { reason: "stop", warnings: [] },
      output: { text: input.text, usage: input.usage, steps: [], object: input.object as OUTPUT | undefined },
      metadata,
    },
  });
  controller.enqueue({
    type: "finish",
    runId: input.runId,
    from: ChunkFrom.AGENT,
    payload: {
      stepResult: { reason: "stop", warnings: [] },
      output: { usage: input.usage, steps: [] },
      metadata,
      providerMetadata: input.providerMetadata,
      messages: { all: [], user: [], nonUser: [] },
      response,
    },
  });
}

export function enqueueError<OUTPUT>(controller: ReadableStreamDefaultController<ChunkType<OUTPUT>>, runId: string, error: unknown): void {
  controller.enqueue({ type: "error", runId, from: ChunkFrom.AGENT, payload: { error } });
}

export function toLanguageModelUsage(usage: V3Usage): LanguageModelUsage {
  const inputTokens = usage.inputTokens.total ?? 0;
  const outputTokens = usage.outputTokens.total ?? 0;
  return {
    inputTokens,
    outputTokens,
    totalTokens: inputTokens + outputTokens,
    cachedInputTokens: usage.inputTokens.cacheRead,
    cacheCreationInputTokens: usage.inputTokens.cacheWrite,
    reasoningTokens: usage.outputTokens.reasoning,
  };
}

export function promptToText(prompt: unknown): string {
  if (typeof prompt === "string") return prompt;
  if (Array.isArray(prompt)) return prompt.map(promptToText).filter(Boolean).join("\n");
  const record = toRecord(prompt);
  if (!record) return "";
  if (typeof record.text === "string") return record.text;
  if (typeof record.content === "string") return record.content;
  if (Array.isArray(record.content)) return record.content.map(promptToText).filter(Boolean).join("\n");
  if (typeof record.message === "string") return record.message;
  return "";
}

export function extractText(parts: Part[] | undefined): string {
  return (parts ?? [])
    .map((part) => (part.type === "text" ? part.text : ""))
    .filter(Boolean)
    .join("");
}

export function createUsage(message: AssistantMessage | undefined): V3Usage {
  const tokens = message?.tokens;
  return {
    inputTokens: {
      total: tokens ? sumDefined(tokens.input, tokens.cache.read, tokens.cache.write) : undefined,
      noCache: tokens?.input,
      cacheRead: tokens?.cache.read,
      cacheWrite: tokens?.cache.write,
    },
    outputTokens: {
      total: tokens ? sumDefined(tokens.output, tokens.reasoning) : undefined,
      text: tokens?.output,
      reasoning: tokens?.reasoning,
    },
  };
}

export function createProviderMetadata(provider: string, metadata: Record<string, unknown>): ProviderMetadata {
  return { [provider]: JSON.parse(JSON.stringify(metadata, (_key, value) => (value === undefined ? null : value))) } as ProviderMetadata;
}

export function getStructuredOutputSchema<OUTPUT>(structuredOutput?: PublicStructuredOutputOptions<OUTPUT>): Record<string, unknown> | undefined {
  if (!structuredOutput?.schema) return undefined;
  return standardSchemaToJSONSchema(toStandardSchema(structuredOutput.schema)) as Record<string, unknown>;
}

export async function getStructuredOutputFromValue<OUTPUT>(
  value: unknown,
  structuredOutput?: PublicStructuredOutputOptions<OUTPUT>,
): Promise<OUTPUT | undefined> {
  if (!structuredOutput?.schema || value === undefined) return undefined;
  let parsed = value;
  if (typeof value === "string") {
    try {
      parsed = JSON.parse(value);
    } catch (error) {
      return handleStructuredOutputError(new Error("Structured output must be valid JSON.", { cause: error }), structuredOutput);
    }
  }
  const schema = toStandardSchema(structuredOutput.schema);
  const result = await schema["~standard"].validate(parsed);
  if (!result.issues) return result.value;
  const message = result.issues.map((issue) => `- ${issue.path?.join(".") || "root"}: ${issue.message}`).join("\n");
  return handleStructuredOutputError(new Error(`Structured output validation failed:\n${message}`), structuredOutput);
}

export function getAssistantError(message: AssistantMessage | undefined): Error | undefined {
  const error = message?.error;
  if (!error) return undefined;
  const data = toRecord(error.data);
  return new Error(getString(data, "message") ?? error.name ?? "OpenCode assistant message failed");
}

function handleStructuredOutputError<OUTPUT>(error: Error, structuredOutput: PublicStructuredOutputOptions<OUTPUT>): OUTPUT | undefined {
  if (structuredOutput.errorStrategy === "fallback") return structuredOutput.fallbackValue as OUTPUT;
  if (structuredOutput.errorStrategy === "warn") {
    structuredOutput.logger?.warn(error.message);
    return undefined;
  }
  throw error;
}

export function sumDefined(...values: Array<number | undefined>): number | undefined {
  const defined = values.filter((value): value is number => typeof value === "number");
  if (!defined.length) return undefined;
  return defined.reduce((sum, value) => sum + value, 0);
}

export function toRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

export function getString(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key];
  return typeof value === "string" ? value : undefined;
}
