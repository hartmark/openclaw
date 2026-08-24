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

function scheduleHttpContinuationIdleCleanup(
  key: string,
  generation: number,
): ReturnType<typeof setTimeout> {
  const idleTimer = setTimeout(() => {
    const current = httpContinuationEntries.get(key);
    if (current?.kind === "ready" && current.generation === generation) {
      httpContinuationEntries.delete(key);
    }
  }, HTTP_CONTINUATION_IDLE_TTL_MS);
  idleTimer.unref?.();
  return idleTimer;
}

export function claimOpenAIResponsesHttpContinuation(
  params: HttpContinuationIdentity & {
    sessionId: string;
    request: ResponsesContinuationRequest;
  },
) {
  const key = `${params.sessionId}\0${connectionIdentity(params)}`;
  const previous = httpContinuationEntries.get(key);
  // eslint-disable-next-line no-console
  console.error(
    `[continuation-debug] CLAIM key=${key.replace("\0", "|")} previousKind=${previous?.kind ?? "none"} ` +
      `entriesSize=${httpContinuationEntries.size}`,
  );
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
  // eslint-disable-next-line no-console
  console.error(
    `[continuation-debug] RESOLVE key=${key.replace("\0", "|")} status=${resolved.continuationStatus} ` +
      `hasPreviousState=${previousState !== undefined}`,
  );
  return {
    request: wireRequest,
    commit: (effectiveRequest: ResponsesContinuationRequest, response: ContinuationResponse) => {
      const stillClaimed = httpContinuationEntries.get(key) === claimed;
      // eslint-disable-next-line no-console
      console.error(
        `[continuation-debug] COMMIT key=${key.replace("\0", "|")} stillClaimed=${stillClaimed} responseId=${response?.id}`,
      );
      if (!stillClaimed) {
        return;
      }
      const ready = {
        ...claimed,
        kind: "ready",
        state: {
          lastRequest: effectiveRequest,
          lastResponseId: response.id,
          lastResponseItems: response.output,
        },
        idleTimer: scheduleHttpContinuationIdleCleanup(key, generation),
      } satisfies Extract<HttpContinuationEntry, { kind: "ready" }>;
      httpContinuationEntries.set(key, ready);
    },
    release: () => {
      const stillClaimed = httpContinuationEntries.get(key) === claimed;
      // eslint-disable-next-line no-console
      console.error(
        `[continuation-debug] RELEASE key=${key.replace("\0", "|")} stillClaimed=${stillClaimed} ` +
          `previousKind=${previous?.kind ?? "none"}`,
      );
      if (!stillClaimed) {
        return;
      }
      // This claim never committed a new state (e.g. the request errored
      // before a response came back). Restore whatever was ready before this
      // claim rather than deleting it: a different model/connection sharing
      // this session (client-side model fallback routes every candidate
      // through the same sessionId+connectionIdentity key, since the cache is
      // not model-scoped) must not destroy a still-valid prior continuation
      // baseline just by claiming and failing. Without this, an unrelated
      // model that always errors first (e.g. an exhausted free-tier fallback
      // primary) permanently prevents the model that actually succeeds from
      // ever accumulating continuation state across turns.
      if (previous?.kind === "ready") {
        httpContinuationEntries.set(key, {
          ...previous,
          idleTimer: scheduleHttpContinuationIdleCleanup(key, previous.generation),
        });
        return;
      }
      httpContinuationEntries.delete(key);
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
