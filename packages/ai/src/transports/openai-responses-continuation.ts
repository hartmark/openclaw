import { stableStringify } from "@openclaw/normalization-core";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import type { ResponseInput, ResponseOutputItem } from "openai/resources/responses/responses.js";
import { getAiTransportHost, resolveAiTransportHeaderSentinels } from "../host.js";
import {
  getSessionResourceOwnerId,
  registerSessionResourceCleanup,
  type SessionResourceOwner,
} from "../session-resources.js";
import { parseJsonObjectPreservingUnsafeIntegers } from "./json-unsafe-integers.js";
import {
  canReferenceResponsesReasoningHistory,
  replayResponsesReasoningUpdates,
  type ResponsesConfigurationUpdate,
} from "./openai-responses-reasoning-update.js";
import {
  normalizeOpenAIResponsesFunctionCallId,
  shouldNormalizeOpenAIResponsesToolCallId,
  splitOpenAIFunctionCallPairing,
} from "./openai-responses-tool-call-id-shape.js";
import { sha256Hex } from "./transport-utils.js";

// Fixed, not operator-configurable. A real chat conversation's turns are
// commonly minutes to hours apart, well past the original 5-minute TTL --
// continuation only ever engaged within one multi-round tool-calling turn
// (seconds between rounds), never across separate incoming messages, even
// though sessionId and connection identity are both stable across turns
// (confirmed by tracing the full call chain). Unchanged since #122194
// introduced it; review only ever flagged the in-memory/process-local
// design generally, never the specific value.
const HTTP_CONTINUATION_IDLE_TTL_MS = 90 * 60 * 1000;
// A ready entry retains the full request/response baseline for
// HTTP_CONTINUATION_IDLE_TTL_MS, now 18x longer (5m -> 90m). Without a
// capacity cap, a burst of concurrent sessions/connections could
// grow this process-wide map unbounded for the entire idle window. Claimed
// entries (in-flight, no retained baseline) don't count against the cap --
// they're already bounded by the request they represent.
const MAX_HTTP_CONTINUATION_READY_ENTRIES = 1000;
// A count cap alone bounds cardinality, not memory: a full-context turn near
// a large model's context window can retain a multi-megabyte baseline on its
// own, so 1000 oversized entries could still exhaust process memory well
// before the count cap engages. This aggregate budget is enforced alongside
// the count cap (whichever evicts first), and also bypasses caching a single
// candidate entry that exceeds the whole budget by itself -- evicting every
// other entry still wouldn't make room for it, and the request itself
// already succeeded, so skipping continuation for that one oversized turn
// (falling back to a full-history resend next round) is strictly better than
// either rejecting the response or growing past the budget. 64MB matches the
// existing ANTHROPIC_INLINE_IMAGES_DECODE_SAFETY_BYTES precedent for a
// single-request memory ceiling in this package -- comfortably above even a
// 200K-token context's realistic JSON footprint (well under 4MB).
const MAX_HTTP_CONTINUATION_RETAINED_BYTES = 64 * 1024 * 1024;
const TURN_HEADERS = new Set(["traceparent", "x-openclaw-turn-id", "x-openclaw-turn-attempt"]);

export type ResponsesContinuationRequest = Record<string, unknown> & {
  input?: Array<ResponseInput[number] | ResponsesConfigurationUpdate>;
  previous_response_id?: string;
};
export type ResponsesSteeringContinuationMode = "automatic" | "required-input";
export type ResponsesContinuationState = {
  lastRequest: ResponsesContinuationRequest;
  lastResponseId: string;
  lastResponseItems: ResponseOutputItem[];
  pendingToolCalls?: Array<{ callId: string; itemId?: string }>;
};
type ContinuationResponse = { id: string; output: ResponseOutputItem[] };
export type ResponsesContinuationStatus =
  | "continued"
  | "explicit_previous_response_id"
  | "history_changed"
  | "history_shorter"
  | "no_previous_response"
  | "request_changed";

function jsonValuesEqual(left: object, right: object): boolean {
  // Normalize the left side first to preserve serialization errors and toJSON ordering.
  const leftJson = JSON.stringify(left) as string;
  const normalizedLeft = stableStringify(JSON.parse(leftJson));
  const rightJson = JSON.stringify(right) as string;
  return leftJson === rightJson || normalizedLeft === stableStringify(JSON.parse(rightJson));
}

function requestWithoutInput(request: ResponsesContinuationRequest): ResponsesContinuationRequest {
  // Instructions and tools apply to the current response and remain on every wire request.
  const {
    input: _input,
    previous_response_id: _previousResponseId,
    instructions: _instructions,
    tools: _tools,
    ...rest
  } = request;
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

// Match replay's ID shaping while keeping unrelated ID edits visible to history checks.
function canonicalizeToolCallId(callId: unknown, itemId: unknown): unknown {
  if (typeof callId !== "string") {
    return callId;
  }
  const paired = typeof itemId === "string" && itemId ? `${callId}|${itemId}` : callId;
  // Replay leaves provider-shaped IDs untouched; match that short-circuit.
  if (!shouldNormalizeOpenAIResponsesToolCallId(paired)) {
    return callId;
  }
  return splitOpenAIFunctionCallPairing(normalizeOpenAIResponsesFunctionCallId(paired)).callId;
}

type ResponsesContinuationToolCall = { callId: string; itemId?: string };

function responseToolCalls(items: readonly unknown[]): ResponsesContinuationToolCall[] {
  return items.flatMap((item) => {
    if (!isRecord(item) || item.type !== "function_call" || typeof item.call_id !== "string") {
      return [];
    }
    return [{ callId: item.call_id, ...(typeof item.id === "string" ? { itemId: item.id } : {}) }];
  });
}

function toolCallReplayShapes(call: ResponsesContinuationToolCall): Set<string> {
  return new Set(
    [
      call.callId,
      canonicalizeToolCallId(call.callId, call.itemId),
      canonicalizeToolCallId(call.callId, undefined),
    ].filter((shape): shape is string => typeof shape === "string"),
  );
}

function toolCallOwnerByReplayShape(
  calls: readonly ResponsesContinuationToolCall[],
): Map<string, number | null> {
  const ownerByReplayShape = new Map<string, number | null>();
  for (const [index, call] of calls.entries()) {
    for (const replayShape of toolCallReplayShapes(call)) {
      const owner = ownerByReplayShape.get(replayShape);
      ownerByReplayShape.set(replayShape, owner === undefined || owner === index ? index : null);
    }
  }
  return ownerByReplayShape;
}

// A later response can supersede the one that introduced an unresolved async call.
// Keep its raw ID while the full transcript still has the call but no output.
function pendingResponsesToolCalls(
  calls: readonly ResponsesContinuationToolCall[],
  input: readonly unknown[],
): ResponsesContinuationToolCall[] {
  const ownerByReplayShape = toolCallOwnerByReplayShape(calls);
  const replayedCallIds = new Set<string>();
  const lastCallPosition = new Map<number, number>();
  const lastOutputPosition = new Map<number, number>();
  for (const [inputIndex, item] of input.entries()) {
    if (!isRecord(item) || typeof item.call_id !== "string") {
      continue;
    }
    const owner = ownerByReplayShape.get(item.call_id);
    if (item.type === "function_call") {
      replayedCallIds.add(item.call_id);
      if (owner !== undefined && owner !== null) {
        lastCallPosition.set(owner, inputIndex);
      }
      continue;
    }
    if (item.type !== "function_call_output") {
      continue;
    }
    if (owner !== undefined && owner !== null) {
      lastOutputPosition.set(owner, inputIndex);
    }
  }
  return calls.filter((_call, index) => {
    const callPosition = lastCallPosition.get(index);
    return (
      [...toolCallReplayShapes(_call)].some((shape) => replayedCallIds.has(shape)) &&
      (callPosition === undefined || (lastOutputPosition.get(index) ?? -1) < callPosition)
    );
  });
}

export function recordResponsesContinuationState(
  previous: ResponsesContinuationState | undefined,
  lastRequest: ResponsesContinuationRequest,
  response: ContinuationResponse,
  continued = false,
): ResponsesContinuationState {
  const calls =
    previous && continued
      ? [...(previous.pendingToolCalls ?? []), ...responseToolCalls(previous.lastResponseItems)]
      : responseToolCalls(lastRequest.input ?? []);
  const seenCalls = new Set<string>();
  const uniqueCalls = calls.filter((call) => {
    const key = JSON.stringify([call.callId, call.itemId]);
    if (seenCalls.has(key)) {
      return false;
    }
    seenCalls.add(key);
    return true;
  });
  const pendingToolCalls = pendingResponsesToolCalls(uniqueCalls, lastRequest.input ?? []);
  return {
    lastRequest,
    lastResponseId: response.id,
    lastResponseItems: response.output,
    pendingToolCalls,
  };
}

// Cached output keeps raw IDs; compare with and without the item ID because replay may omit it.
// `fromResponse` also gates provider-output argument normalization below.
function normalizeAssistantReplayInput(
  input: readonly unknown[],
  fromResponse = false,
  ignoreCachedItemIds = false,
): unknown[] {
  return input.map((item) => {
    if (!isRecord(item)) {
      return item;
    }
    if (item.type === "reasoning") {
      return { type: "reasoning" };
    }
    if (
      item.type !== "function_call" &&
      item.type !== "function_call_output" &&
      !(item.type === "message" && item.role === "assistant")
    ) {
      return item;
    }
    const { id: rawId, status: _status, ...stableItem } = item;
    if (item.type === "function_call_output") {
      // Output ID and status affect replayed input; keep them in the prefix comparison.
      if ("id" in item) {
        stableItem.id = item.id;
      }
      if ("status" in item) {
        stableItem.status = item.status;
      }
    }
    if ("call_id" in stableItem) {
      // Only function_call pairs IDs; function_call_output references the bare call ID.
      const itemId =
        item.type === "function_call" && !(fromResponse && ignoreCachedItemIds) ? rawId : undefined;
      stableItem.call_id = canonicalizeToolCallId(stableItem.call_id, itemId);
    }
    if (fromResponse && item.type === "function_call") {
      // Only provider output crosses terminal admission; sent arguments must retain real type edits.
      const args = parseJsonObjectPreservingUnsafeIntegers(stableItem.arguments);
      stableItem.arguments = args ? JSON.stringify(args) : stableItem.arguments;
    }
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

// Compare tool results by their uniquely identified call occurrence, not a lossy ID alias.
function normalizeContinuationHistory(input: readonly unknown[]): unknown[] | undefined {
  const normalized = normalizeAssistantReplayInput(input);
  const calls = responseToolCalls(input);
  const ownerByReplayShape = toolCallOwnerByReplayShape(calls);
  let callIndex = 0;
  for (const [index, item] of input.entries()) {
    const normalizedItem = normalized[index];
    if (!isRecord(item) || !isRecord(normalizedItem)) {
      continue;
    }
    if (item.type === "function_call") {
      if (typeof normalizedItem.call_id !== "string") {
        return undefined;
      }
      const callOwner = ownerByReplayShape.get(normalizedItem.call_id);
      if (callOwner !== callIndex) {
        return undefined;
      }
      normalizedItem.call_id = { callOwner };
      callIndex += 1;
      continue;
    }
    if (item.type !== "function_call_output" || typeof item.call_id !== "string") {
      continue;
    }
    const owner = ownerByReplayShape.get(item.call_id);
    if (owner === null) {
      return undefined;
    }
    normalizedItem.call_id =
      owner === undefined
        ? { unknownCallId: canonicalizeToolCallId(item.call_id, undefined) }
        : { callOwner: owner };
  }
  return normalized;
}

function continuationHistoryMatches(
  previousInput: readonly unknown[],
  currentInput: readonly unknown[],
): boolean {
  for (const [index, previousItem] of previousInput.entries()) {
    const currentItem = currentInput[index];
    if (!isRecord(previousItem) || !isRecord(currentItem)) {
      continue;
    }
    if (previousItem.type === "function_call" && currentItem.type === "function_call") {
      if (
        typeof previousItem.call_id === "string" &&
        typeof currentItem.call_id === "string" &&
        !toolCallReplayShapes({
          callId: previousItem.call_id,
          ...(typeof previousItem.id === "string" ? { itemId: previousItem.id } : {}),
        }).has(currentItem.call_id)
      ) {
        return false;
      }
    }
    if (
      previousItem.type === "function_call_output" &&
      currentItem.type === "function_call_output" &&
      typeof previousItem.call_id === "string" &&
      typeof currentItem.call_id === "string" &&
      currentItem.call_id !== previousItem.call_id &&
      currentItem.call_id !== canonicalizeToolCallId(previousItem.call_id, undefined)
    ) {
      return false;
    }
  }
  const previousCalls = responseToolCalls(previousInput);
  const currentCalls = responseToolCalls(currentInput);
  if (previousCalls.length !== currentCalls.length) {
    return false;
  }
  for (const [index, previousCall] of previousCalls.entries()) {
    const currentCall = currentCalls[index];
    if (!currentCall) {
      return false;
    }
    const previousShapes = toolCallReplayShapes(previousCall);
    if (![...toolCallReplayShapes(currentCall)].some((shape) => previousShapes.has(shape))) {
      return false;
    }
  }
  const normalizedPrevious = normalizeContinuationHistory(previousInput);
  const normalizedCurrent = normalizeContinuationHistory(currentInput);
  return (
    normalizedPrevious !== undefined &&
    normalizedCurrent !== undefined &&
    jsonValuesEqual(normalizedCurrent, normalizedPrevious)
  );
}

export function responsesContinuationRequestFingerprint(
  request: ResponsesContinuationRequest,
): string {
  const serialized = JSON.stringify(requestWithoutInput(request));
  return sha256Hex(stableStringify(JSON.parse(serialized)));
}

export function responsesContinuationPrefixFingerprint(
  input: readonly unknown[],
  output: readonly unknown[] = [],
): string {
  const serialized = JSON.stringify([
    ...normalizeAssistantReplayInput(input),
    ...normalizeAssistantReplayInput(output, true),
  ]);
  return sha256Hex(stableStringify(JSON.parse(serialized)));
}

// Restore accepted replay IDs in the delta to match the provider's cached call.
function restoreRawCallIdsInDelta(
  delta: readonly unknown[],
  cachedResponseItems: readonly unknown[],
  replayedToolRound: readonly unknown[],
  pendingToolCalls: readonly ResponsesContinuationToolCall[],
): unknown[] | undefined {
  const currentResponseCalls = responseToolCalls(cachedResponseItems);
  const replayedCalls = replayedToolRound.filter(
    (item): item is Record<string, unknown> => isRecord(item) && item.type === "function_call",
  );
  if (currentResponseCalls.length !== replayedCalls.length) {
    return undefined;
  }

  const cachedCalls = [...pendingToolCalls, ...currentResponseCalls];
  const currentCallOffset = pendingToolCalls.length;
  const ownerByReplayShape = toolCallOwnerByReplayShape(cachedCalls);

  // Lossy shaping is safe only when a replayed call or output has one cached owner.
  for (const index of currentResponseCalls.keys()) {
    const replayedCallId = replayedCalls[index]?.call_id;
    if (
      typeof replayedCallId !== "string" ||
      ownerByReplayShape.get(replayedCallId) !== currentCallOffset + index
    ) {
      return undefined;
    }
  }

  const restoredDelta: unknown[] = [];
  for (const item of delta) {
    if (
      !isRecord(item) ||
      item.type !== "function_call_output" ||
      typeof item.call_id !== "string"
    ) {
      restoredDelta.push(item);
      continue;
    }
    const owner = ownerByReplayShape.get(item.call_id);
    // Unknown or shared shapes cannot identify a cached call; resend full history to preserve pairing.
    if (owner === undefined || owner === null) {
      return undefined;
    }
    const rawCallId = cachedCalls[owner]?.callId;
    if (!rawCallId) {
      return undefined;
    }
    restoredDelta.push(rawCallId === item.call_id ? item : { ...item, call_id: rawCallId });
  }
  return restoredDelta;
}

export function resolveResponsesContinuationRequest(
  continuation: ResponsesContinuationState | undefined,
  request: ResponsesContinuationRequest,
  steering?: ResponsesSteeringContinuationMode,
): {
  request: ResponsesContinuationRequest;
  fullRequest?: ResponsesContinuationRequest;
  continuationStatus: ResponsesContinuationStatus;
} {
  if (!continuation) {
    return { request, continuationStatus: "no_previous_response" };
  }
  if (request.previous_response_id) {
    return { request, continuationStatus: "explicit_previous_response_id" };
  }
  // Referenced controls remain active even when omitted from the wire delta.
  // Check compatibility whether the caller supplied them or needs rehydration.
  if (!canReferenceResponsesReasoningHistory(continuation.lastRequest, request)) {
    return { request, continuationStatus: "request_changed" };
  }
  const prepared = replayResponsesReasoningUpdates(
    continuation.lastRequest,
    request,
    continuation.lastResponseItems.length,
    steering,
  );
  // Required input creates a new response with current settings. The same
  // history validation below still binds it to the accepted steering's parent.
  if (
    steering !== "required-input" &&
    !jsonValuesEqual(requestWithoutInput(prepared), requestWithoutInput(continuation.lastRequest))
  ) {
    return { request, continuationStatus: "request_changed" };
  }
  const currentInput = prepared.input ?? [];
  const previousInput = continuation.lastRequest.input ?? [];
  const baselineLength = previousInput.length + continuation.lastResponseItems.length;
  if (currentInput.length < baselineLength) {
    return { request, continuationStatus: "history_shorter" };
  }
  const replayedToolRoundInput = currentInput.slice(previousInput.length, baselineLength);
  const replayedToolRound = normalizeAssistantReplayInput(replayedToolRoundInput);
  // Replay may keep or omit function_call.id, so compare both cached forms.
  const historyToolRoundUnchanged =
    jsonValuesEqual(
      replayedToolRound,
      normalizeAssistantReplayInput(continuation.lastResponseItems, true),
    ) ||
    jsonValuesEqual(
      replayedToolRound,
      normalizeAssistantReplayInput(continuation.lastResponseItems, true, true),
    );
  if (
    !continuationHistoryMatches(previousInput, currentInput.slice(0, previousInput.length)) ||
    !historyToolRoundUnchanged
  ) {
    return { request, continuationStatus: "history_changed" };
  }
  const restoredInput = restoreRawCallIdsInDelta(
    currentInput.slice(baselineLength),
    continuation.lastResponseItems,
    replayedToolRoundInput,
    continuation.pendingToolCalls ?? [],
  );
  if (!restoredInput) {
    return { request, continuationStatus: "history_changed" };
  }
  // Restoration only changes output call IDs on items sliced from current input;
  // unknown or ambiguous IDs reject continuation before this cast.
  // SAFETY: shape preserved by restoreRawCallIdsInDelta as documented above.
  const restoredResponseInput = restoredInput as ResponseInput;
  return {
    request: {
      ...prepared,
      previous_response_id: continuation.lastResponseId,
      input: restoredResponseInput,
    },
    ...(prepared !== request ? { fullRequest: prepared } : {}),
    continuationStatus: "continued",
  };
}

type HttpContinuationEntry =
  | {
      kind: "ready";
      sessionId: string;
      owner: SessionResourceOwner;
      state: ResponsesContinuationState;
      idleTimer: ReturnType<typeof setTimeout>;
      readySequence: number;
      retainedBytes: number;
    }
  | { kind: "claimed"; sessionId: string; owner: SessionResourceOwner };

const httpContinuationEntries = new Map<string, HttpContinuationEntry>();
// Ready-only index, insertion-ordered (a Map iterates in insertion order),
// kept in exact lockstep with every ready entry's add/remove in
// httpContinuationEntries. evictReadyEntriesForCapacity reads only this map
// so its cost stays proportional to the ready-entry cap, never to however
// many claimed (in-flight) entries also happen to exist -- a live gateway
// can have many concurrent long-running requests claimed at once, and
// scanning those on every commit would defeat the cap's own purpose of
// bounding synchronous work.
const readyHttpContinuationEntries = new Map<
  string,
  Extract<HttpContinuationEntry, { kind: "ready" }>
>();
// Monotonic counter for ready-entry commit order: Date.now() is not a
// unique completion order (two commits can land in the same millisecond,
// e.g. a reclaimed session key completing alongside another), so an
// eviction based on wall-clock time can pick a newer entry over an older
// one that happens to share a timestamp. A strictly incrementing sequence
// makes "oldest" unambiguous regardless of timing; readyHttpContinuationEntries'
// own insertion order already matches it (both advance exactly at commit
// time), so eviction only needs the map's natural iteration order.
let nextHttpContinuationReadySequence = 1;
// Running total of every ready entry's retainedBytes, kept in lockstep with
// httpContinuationEntries by removeReadyEntry -- the only path that deletes a
// ready entry -- so MAX_HTTP_CONTINUATION_RETAINED_BYTES can be enforced
// without re-summing the map on every commit.
let httpContinuationRetainedBytes = 0;

// Reference-checked removal for a context that only *might* still own this
// entry (a timer firing later, or `release()`): a foreign caller passes the
// exact entry it was handed, and this is a no-op if that entry has since
// been replaced or evicted. removeReadyEntry below is for a caller that
// already knows -- from a synchronous map read of its own -- that the entry
// it holds is still current.
function deleteHttpContinuationIfOwned(key: string, entry: HttpContinuationEntry): void {
  if (httpContinuationEntries.get(key) !== entry) {
    return;
  }
  if (entry.kind === "ready") {
    httpContinuationRetainedBytes -= entry.retainedBytes;
    readyHttpContinuationEntries.delete(key);
  }
  httpContinuationEntries.delete(key);
}

/** Estimates a ready entry's retained memory: the same JSON that gets
 * stringified for the eviction budget it counts against, no separate copy
 * kept around just to size it. */
function estimateRetainedBytes(state: ResponsesContinuationState): number {
  return Buffer.byteLength(JSON.stringify(state), "utf8");
}

// Single removal path for a ready entry the caller already knows is still
// current (reclaim-before-overwrite, capacity/budget eviction, session
// cleanup) -- keeps idleTimer cleanup and the retainedBytes running total
// symmetric with httpContinuationEntries without duplicating either at each
// call site.
function removeReadyEntry(
  key: string,
  entry: Extract<HttpContinuationEntry, { kind: "ready" }>,
): void {
  clearTimeout(entry.idleTimer);
  httpContinuationRetainedBytes -= entry.retainedBytes;
  readyHttpContinuationEntries.delete(key);
  httpContinuationEntries.delete(key);
}

// Deterministic capacity/budget policy: evict the least-recently-committed
// ready entry first (the one least likely to be reused before its own idle
// TTL would have expired it anyway) until both MAX_HTTP_CONTINUATION_READY_ENTRIES
// and MAX_HTTP_CONTINUATION_RETAINED_BYTES (including the incoming
// `pendingBytes` about to be inserted) are satisfied. Reads only
// readyHttpContinuationEntries (never the full httpContinuationEntries map,
// which can also hold arbitrarily many in-flight claimed entries), and its
// insertion order already is oldest-first, so finding the eviction
// candidate is a single first-entry read, not a scan.
function evictReadyEntriesForCapacity(pendingBytes: number): void {
  for (;;) {
    const overCapacity = readyHttpContinuationEntries.size >= MAX_HTTP_CONTINUATION_READY_ENTRIES;
    const overBudget =
      httpContinuationRetainedBytes + pendingBytes > MAX_HTTP_CONTINUATION_RETAINED_BYTES;
    if (!overCapacity && !overBudget) {
      return;
    }
    const oldest = readyHttpContinuationEntries.entries().next();
    if (oldest.done) {
      return;
    }
    const [oldestKey, oldestEntry] = oldest.value;
    removeReadyEntry(oldestKey, oldestEntry);
  }
}

type HttpContinuationIdentity = {
  apiKey: string;
  baseUrl: string;
  headers: Record<string, string>;
};

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
    restoreRequest?: () => ResponsesContinuationRequest;
  },
) {
  const owner = getAiTransportHost();
  const key = `${getSessionResourceOwnerId(owner)}\0${params.sessionId}\0${connectionIdentity(params)}`;
  const previous = httpContinuationEntries.get(key);
  if (previous?.kind === "claimed") {
    return undefined;
  }
  if (previous?.kind === "ready") {
    removeReadyEntry(key, previous);
  }
  const claimed = { kind: "claimed", sessionId: params.sessionId, owner } as const;
  httpContinuationEntries.set(key, claimed);
  try {
    const request =
      previous?.kind === "ready" ? params.request : (params.restoreRequest?.() ?? params.request);
    const resolved = resolveResponsesContinuationRequest(
      previous?.kind === "ready" ? previous.state : undefined,
      request,
    );
    const fullRequest = resolved.fullRequest ?? request;
    return {
      // Unstored HTTP responses cannot be referenced, but their prompt prefix can still be cached.
      request: params.request.store === false ? fullRequest : resolved.request,
      fullRequest,
      commit: (
        effectiveRequest: ResponsesContinuationRequest,
        response: ContinuationResponse,
        dispatchedPreviousResponseId?: string,
      ) => {
        if (httpContinuationEntries.get(key) !== claimed) {
          return;
        }
        const state = recordResponsesContinuationState(
          previous?.kind === "ready" ? previous.state : undefined,
          effectiveRequest,
          response,
          previous?.kind === "ready" &&
            dispatchedPreviousResponseId === previous.state.lastResponseId,
        );
        const retainedBytes = estimateRetainedBytes(state);
        // recordResponsesContinuationState/estimateRetainedBytes just ran
        // JSON.stringify over caller-supplied request/response content, which
        // can synchronously invoke a caller-defined toJSON/getter. That
        // callback could run session cleanup and/or start a replacement claim
        // at this same key mid-serialization -- re-check ownership now, after
        // the one step that can re-enter this module, and before eviction or
        // any write touches shared state, so a stale commit can't overwrite
        // the replacement claim or resurrect state a concurrent cleanup cleared.
        if (httpContinuationEntries.get(key) !== claimed) {
          return;
        }
        if (retainedBytes > MAX_HTTP_CONTINUATION_RETAINED_BYTES) {
          // Evicting every other entry still wouldn't make this one fit --
          // skip caching it. The turn's actual response already completed
          // successfully; only the *next* turn loses continuation and falls
          // back to a full-history resend, same as before this cache existed.
          deleteHttpContinuationIfOwned(key, claimed);
          return;
        }
        evictReadyEntriesForCapacity(retainedBytes);
        const ready = {
          ...claimed,
          kind: "ready",
          state,
          idleTimer: setTimeout(
            () => deleteHttpContinuationIfOwned(key, ready),
            HTTP_CONTINUATION_IDLE_TTL_MS,
          ),
          readySequence: nextHttpContinuationReadySequence++,
          retainedBytes,
        } satisfies Extract<HttpContinuationEntry, { kind: "ready" }>;
        ready.idleTimer.unref?.();
        httpContinuationRetainedBytes += retainedBytes;
        httpContinuationEntries.set(key, ready);
        readyHttpContinuationEntries.set(key, ready);
      },
      release: () => deleteHttpContinuationIfOwned(key, claimed),
    };
  } catch (error) {
    // Preparation failed before the caller received a handle that could release this claim.
    deleteHttpContinuationIfOwned(key, claimed);
    throw error;
  }
}

registerSessionResourceCleanup((sessionId, owner) => {
  for (const [key, entry] of httpContinuationEntries) {
    if ((!owner || entry.owner === owner) && (!sessionId || entry.sessionId === sessionId)) {
      if (entry.kind === "ready") {
        removeReadyEntry(key, entry);
      } else {
        httpContinuationEntries.delete(key);
      }
    }
  }
});
