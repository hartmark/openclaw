import { stableStringify } from "@openclaw/normalization-core";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import type { ResponseInput, ResponseOutputItem } from "openai/resources/responses/responses.js";
import { getAiTransportHost, resolveAiTransportHeaderSentinels } from "../host.js";
import { registerSessionResourceCleanup } from "../session-resources.js";
import { sha256Hex } from "./transport-utils.js";

const HTTP_CONTINUATION_IDLE_TTL_MS = 5 * 60 * 1000;
const TURN_HEADERS = new Set(["traceparent", "x-openclaw-turn-id", "x-openclaw-turn-attempt"]);

export type ResponsesContinuationRequest = Record<string, unknown> & {
  input?: ResponseInput;
  previous_response_id?: string;
};
export type ResponsesContinuationState = {
  lastRequest: ResponsesContinuationRequest;
  lastResponseId: string;
  lastResponseItems: ResponseOutputItem[];
};
export type ResponsesContinuationStatus =
  | "continued"
  | "explicit_previous_response_id"
  | "history_changed"
  | "history_shorter"
  | "no_previous_response"
  | "request_changed";

function stableStringifyRoundTrip(value: object): string {
  // Round-trip first so stable key ordering retains JSON's omitted/undefined wire semantics.
  // SAFETY: value is a plain JSON-serializable object, so JSON.stringify always returns a string.
  return stableStringify(JSON.parse(JSON.stringify(value) as string));
}

function jsonValuesEqual(left: object, right: object): boolean {
  return stableStringifyRoundTrip(left) === stableStringifyRoundTrip(right);
}

function requestWithoutInput(request: ResponsesContinuationRequest): ResponsesContinuationRequest {
  const { input: _input, previous_response_id: _previousResponseId, ...rest } = request;
  if (!isRecord(rest.metadata)) {
    return rest;
  }
  const metadata = Object.fromEntries(
    Object.entries(rest.metadata).filter(
      ([key]) => key !== "openclaw_turn_id" && key !== "openclaw_turn_attempt",
    ),
  );
  return { ...rest, metadata };
}

function normalizeAssistantReplayInput(input: readonly unknown[]): unknown[] {
  return input.map((item) => {
    if (!isRecord(item)) {
      return item;
    }
    if (item.type === "reasoning") {
      return { type: "reasoning" };
    }
    if (item.type !== "function_call" && !(item.type === "message" && item.role === "assistant")) {
      return item;
    }
    const { id: _id, status: _status, ...stableItem } = item;
    if (item.type === "message" && Array.isArray(stableItem.content)) {
      stableItem.content = stableItem.content.map((part) => {
        if (!isRecord(part) || part.type !== "output_text") {
          return part;
        }
        const { annotations: _annotations, logprobs: _logprobs, ...stablePart } = part;
        return stablePart;
      });
    }
    return stableItem;
  });
}

function diffFirstMismatch(a: object, b: object): string {
  const aStr = stableStringifyRoundTrip(a);
  const bStr = stableStringifyRoundTrip(b);
  let i = 0;
  const len = Math.min(aStr.length, bStr.length);
  while (i < len && aStr[i] === bStr[i]) i++;
  return (
    `diffAt=${i} aLen=${aStr.length} bLen=${bStr.length} ` +
    `aAround=${JSON.stringify(aStr.slice(Math.max(0, i - 40), i + 120))} ` +
    `bAround=${JSON.stringify(bStr.slice(Math.max(0, i - 40), i + 120))}`
  );
}

function debugHistoryMismatch(
  continuation: ResponsesContinuationState | undefined,
  request: ResponsesContinuationRequest,
): string | undefined {
  if (!continuation) return undefined;
  const currentInput = request.input ?? [];
  const previousInput = continuation.lastRequest.input ?? [];
  const baselineLength = previousInput.length + continuation.lastResponseItems.length;
  const prefixA = normalizeAssistantReplayInput(currentInput.slice(0, previousInput.length));
  const prefixB = normalizeAssistantReplayInput(previousInput);
  if (!jsonValuesEqual(prefixA, prefixB)) {
    return `prefix-mismatch ${diffFirstMismatch(prefixA, prefixB)}`;
  }
  const replyA = normalizeAssistantReplayInput(
    currentInput.slice(previousInput.length, baselineLength),
  );
  const replyB = normalizeAssistantReplayInput(continuation.lastResponseItems);
  if (!jsonValuesEqual(replyA, replyB)) {
    return `reply-mismatch ${diffFirstMismatch(replyA, replyB)}`;
  }
  return "no-mismatch-found";
}

export function resolveResponsesContinuationRequest(
  continuation: ResponsesContinuationState | undefined,
  request: ResponsesContinuationRequest,
): { request: ResponsesContinuationRequest; continuationStatus: ResponsesContinuationStatus } {
  if (!continuation) {
    return { request, continuationStatus: "no_previous_response" };
  }
  if (request.previous_response_id) {
    return { request, continuationStatus: "explicit_previous_response_id" };
  }
  if (
    !jsonValuesEqual(requestWithoutInput(request), requestWithoutInput(continuation.lastRequest))
  ) {
    return { request, continuationStatus: "request_changed" };
  }
  const currentInput = request.input ?? [];
  const previousInput = continuation.lastRequest.input ?? [];
  const baselineLength = previousInput.length + continuation.lastResponseItems.length;
  if (currentInput.length < baselineLength) {
    return { request, continuationStatus: "history_shorter" };
  }
  if (
    !jsonValuesEqual(
      normalizeAssistantReplayInput(currentInput.slice(0, previousInput.length)),
      normalizeAssistantReplayInput(previousInput),
    ) ||
    !jsonValuesEqual(
      normalizeAssistantReplayInput(currentInput.slice(previousInput.length, baselineLength)),
      normalizeAssistantReplayInput(continuation.lastResponseItems),
    )
  ) {
    return { request, continuationStatus: "history_changed" };
  }
  return {
    request: {
      ...request,
      previous_response_id: continuation.lastResponseId,
      input: currentInput.slice(baselineLength),
    },
    continuationStatus: "continued",
  };
}

type HttpContinuationEntry =
  | {
      kind: "ready";
      sessionId: string;
      generation: number;
      state: ResponsesContinuationState;
      idleTimer: ReturnType<typeof setTimeout>;
    }
  | { kind: "claimed"; sessionId: string; generation: number };

const httpContinuationEntries = new Map<string, HttpContinuationEntry>();
let nextHttpContinuationGeneration = 1;

type HttpContinuationIdentity = {
  apiKey: string;
  baseUrl: string;
  headers: Record<string, string>;
};
type ContinuationResponse = { id: string; output: ResponseOutputItem[] };

function connectionIdentity(params: HttpContinuationIdentity): string {
  const headers = Object.entries(resolveAiTransportHeaderSentinels(params.headers) ?? {})
    .map(([name, value]) => [name.toLowerCase(), value] as const)
    .filter(([name]) => !TURN_HEADERS.has(name))
    .toSorted(([a], [b]) => a.localeCompare(b));
  return sha256Hex(
    JSON.stringify([
      getAiTransportHost().resolveSecretSentinel(params.apiKey),
      params.baseUrl,
      headers,
    ]),
  );
}

export function claimOpenAIResponsesHttpContinuation(
  params: HttpContinuationIdentity & {
    sessionId: string;
    request: ResponsesContinuationRequest;
  },
) {
  const key = `${params.sessionId}\0${connectionIdentity(params)}`;
  const previous = httpContinuationEntries.get(key);
  if (previous?.kind === "claimed") {
    return undefined;
  }
  if (previous?.kind === "ready") {
    clearTimeout(previous.idleTimer);
  }
  const generation = nextHttpContinuationGeneration++;
  const claimed = { kind: "claimed", sessionId: params.sessionId, generation } as const;
  httpContinuationEntries.set(key, claimed);
  const previousState = previous?.kind === "ready" ? previous.state : undefined;
  const resolved = resolveResponsesContinuationRequest(previousState, params.request);
  const wireRequest = resolved.request;
  return {
    request: wireRequest,
    debugStatus: resolved.continuationStatus,
    debugMismatch:
      resolved.continuationStatus === "history_changed"
        ? debugHistoryMismatch(previousState, params.request)
        : resolved.continuationStatus === "request_changed"
          ? // SAFETY: resolveResponsesContinuationRequest only returns "request_changed"
            // when `continuation` (previousState) was defined; see its no_previous_response
            // early return above.
            diffFirstMismatch(
              requestWithoutInput(params.request),
              requestWithoutInput(previousState!.lastRequest),
            )
          : undefined,
    commit: (effectiveRequest: ResponsesContinuationRequest, response: ContinuationResponse) => {
      if (httpContinuationEntries.get(key) !== claimed) {
        return;
      }
      const idleTimer = setTimeout(() => {
        const current = httpContinuationEntries.get(key);
        if (current?.kind === "ready" && current.generation === generation) {
          httpContinuationEntries.delete(key);
        }
      }, HTTP_CONTINUATION_IDLE_TTL_MS);
      idleTimer.unref?.();
      const ready = {
        ...claimed,
        kind: "ready",
        state: {
          lastRequest: effectiveRequest,
          lastResponseId: response.id,
          lastResponseItems: response.output,
        },
        idleTimer,
      } satisfies Extract<HttpContinuationEntry, { kind: "ready" }>;
      httpContinuationEntries.set(key, ready);
    },
    release: () => {
      if (httpContinuationEntries.get(key) === claimed) {
        httpContinuationEntries.delete(key);
      }
    },
  };
}

registerSessionResourceCleanup((sessionId) => {
  for (const [key, entry] of httpContinuationEntries) {
    if (!sessionId || entry.sessionId === sessionId) {
      if (entry.kind === "ready") {
        clearTimeout(entry.idleTimer);
      }
      httpContinuationEntries.delete(key);
    }
  }
});
