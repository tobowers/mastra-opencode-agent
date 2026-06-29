import { randomUUID } from "node:crypto";
import { ReadableStream } from "node:stream/web";

import { Agent } from "@mastra/core/agent";
import type { MessageListInput } from "@mastra/core/agent/message-list";
import type { Mastra } from "@mastra/core/mastra";
import { EntityType, executeWithContext, getOrCreateSpan, SpanType } from "@mastra/core/observability";
import type { AnySpan } from "@mastra/core/observability";
import type { ChunkType, FullOutput, MastraModelOutput, ProviderMetadata } from "@mastra/core/stream";
import { createOpencode } from "@opencode-ai/sdk";
import type { AssistantMessage, Event, OpencodeClient, Part, ServerOptions, Session, SessionPromptData } from "@opencode-ai/sdk";

import {
  createMastraOutput,
  createNoopModel,
  createProviderMetadata,
  createUsage,
  enqueueError,
  enqueueFinishChunks,
  enqueueStartChunks,
  enqueueTextDelta,
  extractText,
  getAssistantError,
  getString,
  getStructuredOutputFromValue,
  getStructuredOutputSchema,
  promptToText,
  toFullOutput,
  toLanguageModelUsage,
  toRecord,
} from "./utils.js";
import type { SDKAgentRunOptions, SDKModelGenerateResult } from "./utils.js";

const PROVIDER = "@opencode-ai/sdk";
const MODEL_ID = "opencode";

export type OpenCodeModel = { providerID: string; modelID: string } | string;

export type OpenCodeSDKOptions = ServerOptions & {
  /** Reuse an OpenCode session. If omitted, each run creates a session. */
  sessionId?: string;
  /** Title used when creating a session for a run. */
  sessionTitle?: string;
  /** Directory query sent to OpenCode session APIs. */
  directory?: string;
  /** OpenCode model selector for prompts. String values use `provider/model`. */
  model?: OpenCodeModel;
  /** OpenCode agent name for prompts. */
  agent?: string;
  /** OpenCode system prompt override. */
  system?: string;
  /** OpenCode tool enablement map. */
  tools?: Record<string, boolean>;
};

export type OpenCodeInstance = {
  client: OpencodeClient;
  server?: {
    url: string;
    close(): void;
  };
};

export type OpenCodeFactory = (options?: ServerOptions) => OpenCodeInstance | Promise<OpenCodeInstance>;

type OpenCodeAgentBaseOptions = {
  id: string;
  name?: string;
  description: string;
};

export type OpenCodeAgentOptions = OpenCodeAgentBaseOptions &
  (
    | {
        /** Pre-created OpenCode SDK client. Pass this when you manage the server lifecycle. */
        client: OpencodeClient;
        opencode?: never;
        sdkOptions?: OpenCodeSDKOptions;
      }
    | {
        /** Pre-created OpenCode instance or factory. Useful for custom lifecycle control and tests. */
        opencode: OpenCodeInstance | Promise<OpenCodeInstance> | OpenCodeFactory;
        client?: never;
        sdkOptions?: OpenCodeSDKOptions;
      }
    | {
        client?: never;
        opencode?: never;
        /** Options used to start OpenCode plus default prompt/session options. */
        sdkOptions?: OpenCodeSDKOptions;
      }
  );

export type OpenCodeRunOptions<OUTPUT = unknown> = SDKAgentRunOptions<OUTPUT> & {
  sdkOptions?: Partial<OpenCodeSDKOptions> & {
    /** Use this message id for the OpenCode user message. Defaults to a UUID. */
    messageId?: string;
  };
};

export type OpenCodeSDKAgentResumeData = {
  message: MessageListInput;
  sessionId: string;
  directory?: string;
};

export class OpenCodeSDKAgent extends Agent {
  readonly options: OpenCodeAgentOptions;
  #mastra?: Mastra;
  #instance?: Promise<OpenCodeInstance>;

  constructor(options: OpenCodeAgentOptions) {
    super({
      id: options.id,
      name: options.name ?? options.id,
      description: options.description,
      instructions: "",
      model: createNoopModel({ modelId: getModelId(options.sdkOptions), provider: PROVIDER }),
    });
    this.options = options;
  }

  override __registerMastra(mastra: Mastra): void {
    super.__registerMastra(mastra);
    this.#mastra = mastra;
  }

  supportsMemory(): boolean {
    return false;
  }

  async close(): Promise<void> {
    const instance = await this.#instance;
    instance?.server?.close();
    this.#instance = undefined;
  }

  override async generate<OUTPUT = undefined>(messages: MessageListInput, options?: OpenCodeRunOptions<OUTPUT>): Promise<FullOutput<OUTPUT>> {
    void this.#mastra;
    const runId = options?.runId ?? randomUUID();
    const promptOptions = mergeRunOptions(this.options.sdkOptions, options?.sdkOptions);
    const telemetry = createOpenCodeTelemetry({
      agent: this,
      mastra: this.#mastra,
      messages,
      runId,
      modelId: getModelId(promptOptions),
      method: "generate",
      streaming: false,
      options,
    });
    let result: SDKModelGenerateResult;
    try {
      result = await telemetry.execute(() => this.runOpenCodeGenerate(messages, options));
      telemetry.endGenerate(result);
    } catch (error) {
      telemetry.fail(error);
      throw error;
    }
    return toFullOutput({
      messages,
      runId,
      provider: PROVIDER,
      result,
      options: telemetry.outputOptions(),
    });
  }

  override async stream<OUTPUT = undefined>(messages: MessageListInput, options?: OpenCodeRunOptions<OUTPUT>): Promise<MastraModelOutput<OUTPUT>> {
    void this.#mastra;
    const runId = options?.runId ?? randomUUID();
    const promptOptions = mergeRunOptions(this.options.sdkOptions, options?.sdkOptions);
    const modelId = getModelId(promptOptions);
    const telemetry = createOpenCodeTelemetry({
      agent: this,
      mastra: this.#mastra,
      messages,
      runId,
      modelId,
      method: "stream",
      streaming: true,
      options,
    });
    return createMastraOutput({
      messages,
      runId,
      modelId,
      provider: PROVIDER,
      stream: telemetry.wrapStream(this.runOpenCodeAsMastraStream<OUTPUT>(messages, runId, options)),
      options: telemetry.outputOptions(),
    });
  }

  override async resumeGenerate<OUTPUT = undefined>(
    resumeData: OpenCodeSDKAgentResumeData,
    options?: OpenCodeRunOptions<OUTPUT>,
  ): Promise<FullOutput<OUTPUT>> {
    const data = validateResumeData(resumeData);
    return this.generate(data.message, createResumeRunOptions(data, options));
  }

  override async resumeStream<OUTPUT = undefined>(
    resumeData: OpenCodeSDKAgentResumeData,
    options?: OpenCodeRunOptions<OUTPUT>,
  ): Promise<MastraModelOutput<OUTPUT>> {
    const data = validateResumeData(resumeData);
    return this.stream(data.message, createResumeRunOptions(data, options));
  }

  async runOpenCodeGenerate<OUTPUT>(messages: MessageListInput, options?: OpenCodeRunOptions<OUTPUT>): Promise<SDKModelGenerateResult> {
    const client = await this.resolveClient();
    const promptOptions = mergeRunOptions(this.options.sdkOptions, options?.sdkOptions);
    const session = await resolveSession(client, promptOptions);
    const prompt = promptToText(messages);
    const body = createPromptBody(prompt, promptOptions, options);
    const response = (await unwrapResult(client.session.prompt({ path: { id: session.id }, query: createQuery(promptOptions), body } as never))) as PromptResponse;
    const error = getAssistantError(response.info);
    if (error) throw error;
    const text = extractText(response.parts);
    const object = await getStructuredOutputFromOpenCode(response, text, options?.structuredOutput);
    return createGenerateResult({ info: response.info, parts: response.parts, session, promptOptions, object });
  }

  runOpenCodeAsMastraStream<OUTPUT>(
    messages: MessageListInput,
    runId: string,
    options?: OpenCodeRunOptions<OUTPUT>,
  ): ReadableStream<ChunkType<OUTPUT>> {
    return new ReadableStream({
      start: async (controller) => {
        const client = await this.resolveClient();
        const promptOptions = mergeRunOptions(this.options.sdkOptions, options?.sdkOptions);
        const session = await resolveSession(client, promptOptions);
        const prompt = promptToText(messages);
        const userMessageId = promptOptions.messageId ?? randomUUID();
        const body = createPromptBody(prompt, { ...promptOptions, messageId: userMessageId }, options);
        const textId = randomUUID();
        const seenParts = new Map<string, string>();
        let assistant: AssistantMessage | undefined;
        let assistantMessageId: string | undefined;
        let responseModel = getModelId(promptOptions);
        let responseId = userMessageId;
        let text = "";

        try {
          const events = await client.event.subscribe({ query: createQuery(promptOptions), signal: getAbortSignal(options) } as never);
          enqueueStartChunks(controller, {
            runId,
            prompt,
            textId,
            responseId,
            modelId: responseModel,
            providerMetadata: createOpenCodeProviderMetadata({ session, promptOptions, message: assistant }),
          });
          await unwrapResult(client.session.promptAsync({ path: { id: session.id }, query: createQuery(promptOptions), body } as never));

          for await (const event of events.stream as AsyncIterable<Event>) {
            const outcome = handleOpenCodeEvent(event, { sessionId: session.id, userMessageId, assistantMessageId, seenParts });
            if (outcome.assistant) {
              assistant = outcome.assistant;
              assistantMessageId = assistant.id;
              responseId = assistant.id;
              responseModel = getMessageModelId(assistant, promptOptions);
              const error = getAssistantError(assistant);
              if (error) throw error;
            }
            if (outcome.delta) {
              text += outcome.delta;
              enqueueTextDelta(controller, runId, textId, outcome.delta, createOpenCodeProviderMetadata({ session, promptOptions, message: assistant }));
            }
            if (outcome.done) break;
          }

          const final = await readFinalAssistant(client, session.id, userMessageId, promptOptions);
          assistant ??= final?.info;
          responseId = assistant?.id ?? responseId;
          responseModel = getMessageModelId(assistant, promptOptions);
          const finalText = extractText(final?.parts);
          if (!text && finalText) {
            text = finalText;
            enqueueTextDelta(controller, runId, textId, finalText, createOpenCodeProviderMetadata({ session, promptOptions, message: assistant }));
          } else if (finalText.startsWith(text) && finalText.length > text.length) {
            const delta = finalText.slice(text.length);
            text = finalText;
            enqueueTextDelta(controller, runId, textId, delta, createOpenCodeProviderMetadata({ session, promptOptions, message: assistant }));
          }

          const object = await getStructuredOutputFromOpenCode(final, text, options?.structuredOutput);
          enqueueFinishChunks(controller, {
            runId,
            prompt,
            textId,
            text,
            responseId,
            modelId: responseModel,
            usage: toLanguageModelUsage(createUsage(assistant)),
            providerMetadata: createOpenCodeProviderMetadata({ session, promptOptions, message: assistant }),
            object,
          });
          controller.close();
        } catch (error) {
          enqueueError(controller, runId, error);
          controller.close();
        }
      },
    });
  }

  private async resolveClient(): Promise<OpencodeClient> {
    if ("client" in this.options && this.options.client) return this.options.client;
    this.#instance ??= this.createInstance();
    return (await this.#instance).client;
  }

  private async createInstance(): Promise<OpenCodeInstance> {
    if ("opencode" in this.options && this.options.opencode) {
      const opencode = this.options.opencode;
      return typeof opencode === "function" ? opencode(toServerOptions(this.options.sdkOptions)) : opencode;
    }
    return createOpencode(toServerOptions(this.options.sdkOptions));
  }
}

export type { SDKAgentRunOptions } from "./utils.js";

type PromptOptions = Partial<OpenCodeSDKOptions> & { messageId?: string };
type PromptResponse = { info: AssistantMessage; parts: Part[] };
type MessageResponse = { info: AssistantMessage | { role: string; id: string; parentID?: string }; parts: Part[] };

function createOpenCodeTelemetry<OUTPUT>(input: {
  agent: OpenCodeSDKAgent;
  mastra?: Mastra;
  messages: MessageListInput;
  runId: string;
  modelId: string;
  method: "generate" | "stream";
  streaming: boolean;
  options?: OpenCodeRunOptions<OUTPUT>;
}) {
  const prompt = promptToText(input.messages);
  const agentSpan = getOrCreateSpan({
    type: SpanType.AGENT_RUN,
    name: `agent run: '${input.agent.id}'`,
    entityType: EntityType.AGENT,
    entityId: input.agent.id,
    entityName: input.agent.name,
    input: input.messages,
    attributes: {
      prompt,
      instructions: promptToText(input.options?.instructions),
      maxSteps: input.options?.maxSteps,
    },
    metadata: {
      runId: input.runId,
      sdkAgent: true,
      sdkProvider: PROVIDER,
      sdkMethod: input.method,
    },
    tracingOptions: input.options?.tracingOptions,
    tracingContext: input.options?.tracingContext,
    requestContext: input.options?.requestContext,
    mastra: input.mastra,
  });
  const modelSpan = agentSpan?.createChildSpan({
    type: SpanType.MODEL_GENERATION,
    name: `llm: '${input.modelId}'`,
    input: { messages: input.messages },
    attributes: { model: input.modelId, provider: PROVIDER, streaming: input.streaming },
    metadata: {
      runId: input.runId,
      sdkAgent: true,
      sdkProvider: PROVIDER,
      sdkMethod: input.method,
    },
    requestContext: input.options?.requestContext,
  });
  const tracker = modelSpan && "createTracker" in modelSpan ? modelSpan.createTracker() : undefined;
  let ended = false;

  const end = (result: {
    text?: string;
    usage?: ReturnType<typeof toLanguageModelUsage>;
    providerMetadata?: ProviderMetadata;
    finishReason?: string;
    responseId?: string;
    responseModel?: string;
  }) => {
    if (ended) return;
    ended = true;
    if (tracker) {
      tracker.endGeneration({
        output: { text: result.text ?? "" },
        attributes: { finishReason: result.finishReason ?? "stop", responseId: result.responseId, responseModel: result.responseModel },
        usage: result.usage,
        providerMetadata: result.providerMetadata,
      });
    } else {
      modelSpan?.end({
        output: { text: result.text ?? "" },
        attributes: { finishReason: result.finishReason ?? "stop", responseId: result.responseId, responseModel: result.responseModel },
      });
    }
    agentSpan?.end({ output: { text: result.text ?? "" } });
  };

  const fail = (error: unknown) => {
    if (ended) return;
    ended = true;
    const normalized = error instanceof Error ? error : new Error(String(error));
    if (tracker) tracker.reportGenerationError({ error: normalized });
    else modelSpan?.error({ error: normalized });
    agentSpan?.error({ error: normalized });
  };

  return {
    execute: <T>(fn: () => Promise<T>) => executeWithContext({ span: (modelSpan ?? agentSpan) as AnySpan | undefined, fn }),
    endGenerate(result: SDKModelGenerateResult) {
      end({
        text: result.content.map((part) => part.text).join(""),
        usage: toLanguageModelUsage(result.usage),
        providerMetadata: result.providerMetadata,
        finishReason: result.finishReason.unified,
        responseId: result.response.id,
        responseModel: result.response.modelId,
      });
    },
    fail,
    wrapStream<TOutput>(stream: ReadableStream<ChunkType<TOutput>>): ReadableStream<ChunkType<TOutput>> {
      const tracked = tracker?.wrapStream(stream) ?? stream;
      let text = "";
      return tracked.pipeThrough(
        new TransformStream<ChunkType<TOutput>, ChunkType<TOutput>>({
          transform(chunk, controller) {
            if (chunk.type === "text-delta") text += chunk.payload.text;
            if (chunk.type === "finish") {
              end({
                text,
                usage: chunk.payload.output.usage,
                providerMetadata: chunk.payload.providerMetadata,
                finishReason: chunk.payload.stepResult.reason,
                responseId: chunk.payload.response?.id,
                responseModel: chunk.payload.response?.modelId,
              });
            }
            if (chunk.type === "error") fail(chunk.payload.error);
            controller.enqueue(chunk);
          },
          flush() {
            end({ text });
          },
        }),
      );
    },
    outputOptions() {
      return {
        onFinish: input.options?.onFinish,
        onStepFinish: input.options?.onStepFinish,
        requestContext: input.options?.requestContext,
        tracingContext: agentSpan ? { currentSpan: agentSpan } : input.options?.tracingContext,
      };
    },
  };
}

async function resolveSession(client: OpencodeClient, options: PromptOptions): Promise<Session> {
  if (options.sessionId) {
    return (await unwrapResult(client.session.get({ path: { id: options.sessionId }, query: createQuery(options) } as never))) as Session;
  }
  return (await unwrapResult(client.session.create({ body: { title: options.sessionTitle }, query: createQuery(options) } as never))) as Session;
}

function createPromptBody<OUTPUT>(
  prompt: string,
  options: PromptOptions,
  runOptions?: OpenCodeRunOptions<OUTPUT>,
): NonNullable<SessionPromptData["body"]> & Record<string, unknown> {
  const body: NonNullable<SessionPromptData["body"]> & Record<string, unknown> = {
    messageID: options.messageId,
    parts: [{ type: "text", text: prompt }],
  };
  const model = toModelSelector(options.model);
  if (model) body.model = model;
  if (options.agent) body.agent = options.agent;
  if (options.system ?? runOptions?.instructions) body.system = options.system ?? promptToText(runOptions?.instructions);
  if (options.tools) body.tools = options.tools;
  const schema = getStructuredOutputSchema(runOptions?.structuredOutput);
  if (schema) body.format = { type: "json_schema", schema };
  return body;
}

function createGenerateResult(input: {
  info: AssistantMessage;
  parts: Part[];
  session: Session;
  promptOptions: PromptOptions;
  object?: unknown;
}): SDKModelGenerateResult {
  return {
    content: [{ type: "text", text: extractText(input.parts) }],
    finishReason: {
      unified: input.info.finish === "abort" ? "abort" : input.info.error ? "error" : "stop",
      raw: input.info.finish ?? "stop",
    },
    usage: createUsage(input.info),
    response: {
      id: input.info.id,
      modelId: getMessageModelId(input.info, input.promptOptions),
      timestamp: new Date(input.info.time.completed ?? input.info.time.created),
    },
    providerMetadata: createOpenCodeProviderMetadata({ session: input.session, promptOptions: input.promptOptions, message: input.info }),
    object: input.object,
  };
}

function handleOpenCodeEvent(
  event: Event,
  state: {
    sessionId: string;
    userMessageId: string;
    assistantMessageId?: string;
    seenParts: Map<string, string>;
  },
): { assistant?: AssistantMessage; delta?: string; done?: boolean } {
  if (event.type === "session.idle" && event.properties.sessionID === state.sessionId) return { done: true };
  if (event.type === "session.error" && (!event.properties.sessionID || event.properties.sessionID === state.sessionId)) {
    const error = toRecord(event.properties.error);
    throw new Error(getString(toRecord(error?.data), "message") ?? getString(error, "name") ?? "OpenCode session failed");
  }
  if (event.type === "message.updated") {
    const info = event.properties.info;
    if (info.role === "assistant" && info.sessionID === state.sessionId && info.parentID === state.userMessageId) {
      return { assistant: info };
    }
  }
  if (event.type !== "message.part.updated") return {};
  const { part, delta } = event.properties;
  if (part.sessionID !== state.sessionId || part.type !== "text") return {};
  if (state.assistantMessageId && part.messageID !== state.assistantMessageId) return {};
  if (typeof delta === "string") return { delta };

  const previous = state.seenParts.get(part.id) ?? "";
  const current = part.text;
  state.seenParts.set(part.id, current);
  return current.startsWith(previous) ? { delta: current.slice(previous.length) } : {};
}

async function readFinalAssistant(
  client: OpencodeClient,
  sessionId: string,
  userMessageId: string,
  options: PromptOptions,
): Promise<PromptResponse | undefined> {
  const messages = await unwrapResult(client.session.messages({ path: { id: sessionId }, query: createQuery(options) } as never));
  const matches = (messages as MessageResponse[]).filter(
    (message): message is PromptResponse =>
      message.info.role === "assistant" && "parentID" in message.info && message.info.parentID === userMessageId,
  );
  return matches.at(-1);
}

async function getStructuredOutputFromOpenCode<OUTPUT>(
  response: PromptResponse | undefined,
  text: string,
  structuredOutput?: OpenCodeRunOptions<OUTPUT>["structuredOutput"],
): Promise<OUTPUT | undefined> {
  const info = toRecord(response?.info);
  const structured = toRecord(info?.structured_output) ?? toRecord(info?.structuredOutput);
  return getStructuredOutputFromValue(structured ?? text, structuredOutput);
}

function createOpenCodeProviderMetadata(input: {
  session: Session;
  promptOptions: PromptOptions;
  message?: AssistantMessage;
}): ProviderMetadata {
  const model = toModelSelector(input.promptOptions.model);
  return createProviderMetadata("opencode", {
    sessionId: input.session.id,
    sessionTitle: input.session.title,
    directory: input.promptOptions.directory,
    messageId: input.message?.id,
    providerId: input.message?.providerID ?? model?.providerID,
    modelId: input.message?.modelID ?? model?.modelID,
    mode: input.message?.mode,
    finish: input.message?.finish,
    cost: input.message?.cost,
  });
}

function mergeRunOptions(base?: OpenCodeSDKOptions, override?: Partial<OpenCodeSDKOptions>): PromptOptions {
  return { ...base, ...override, tools: { ...base?.tools, ...override?.tools } };
}

function toServerOptions(options?: OpenCodeSDKOptions): ServerOptions | undefined {
  if (!options) return undefined;
  const serverOptions: ServerOptions = {};
  if (options.hostname !== undefined) serverOptions.hostname = options.hostname;
  if (options.port !== undefined) serverOptions.port = options.port;
  if (options.signal !== undefined) serverOptions.signal = options.signal;
  if (options.timeout !== undefined) serverOptions.timeout = options.timeout;
  if (options.config !== undefined) serverOptions.config = options.config;
  return serverOptions;
}

function createQuery(options: PromptOptions): { directory?: string } | undefined {
  return options.directory ? { directory: options.directory } : undefined;
}

function toModelSelector(model?: OpenCodeModel): { providerID: string; modelID: string } | undefined {
  if (!model) return undefined;
  if (typeof model !== "string") return model;
  const index = model.indexOf("/");
  if (index === -1) return undefined;
  return { providerID: model.slice(0, index), modelID: model.slice(index + 1) };
}

function getModelId(options?: PromptOptions): string {
  const model = toModelSelector(options?.model);
  return model ? `${model.providerID}/${model.modelID}` : MODEL_ID;
}

function getMessageModelId(message: AssistantMessage | undefined, options: PromptOptions): string {
  return message ? `${message.providerID}/${message.modelID}` : getModelId(options);
}

function getAbortSignal(options?: { signal?: AbortSignal; abortSignal?: AbortSignal }): AbortSignal | undefined {
  return options?.signal ?? options?.abortSignal;
}

function validateResumeData(resumeData: OpenCodeSDKAgentResumeData): OpenCodeSDKAgentResumeData {
  if (!toRecord(resumeData) || !("message" in resumeData)) throw new Error("OpenCodeSDKAgent resumeData must include a message.");
  if (typeof resumeData.sessionId !== "string" || !resumeData.sessionId) {
    throw new Error("OpenCodeSDKAgent resumeData.sessionId must be a non-empty string.");
  }
  return resumeData;
}

function createResumeRunOptions<OUTPUT>(resumeData: OpenCodeSDKAgentResumeData, options?: OpenCodeRunOptions<OUTPUT>): OpenCodeRunOptions<OUTPUT> {
  return {
    ...options,
    sdkOptions: {
      ...options?.sdkOptions,
      sessionId: resumeData.sessionId,
      directory: resumeData.directory ?? options?.sdkOptions?.directory,
    },
  };
}

async function unwrapResult<T>(result: Promise<T>): Promise<T extends { data: infer D } ? D : T> {
  const value = await result;
  const record = toRecord(value);
  if (record && "error" in record && record.error) {
    const error = toRecord(record.error);
    throw new Error(getString(toRecord(error?.data), "message") ?? getString(error, "name") ?? "OpenCode request failed");
  }
  if (record && "data" in record) return record.data as T extends { data: infer D } ? D : T;
  return value as T extends { data: infer D } ? D : T;
}

export const __testing = {
  createPromptBody,
  handleOpenCodeEvent,
  mergeRunOptions,
  toServerOptions,
  toModelSelector,
};
