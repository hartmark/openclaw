import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanupSessionResources } from "../session-resources.js";
import {
  claimOpenAIResponsesHttpContinuation,
  resolveResponsesContinuationRequest,
  type ResponsesContinuationRequest,
  type ResponsesContinuationState,
} from "./openai-responses-continuation.js";

const firstUser = {
  type: "message",
  role: "user",
  content: [{ type: "input_text", text: "first" }],
};
const assistantOutput = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  status: "completed",
  phase: "final_answer",
  content: [
    {
      type: "output_text",
      text: "answer",
      annotations: [
        {
          type: "url_citation",
          url: "https://example.test/source",
          title: "source",
          start_index: 0,
          end_index: 6,
        },
      ],
      logprobs: [{ token: "answer", logprob: -0.1, bytes: [], top_logprobs: [] }],
    },
  ],
};

function continuationState(): ResponsesContinuationState {
  return {
    lastRequest: {
      model: "gpt-5.6-luna",
      store: true,
      max_output_tokens: undefined,
      metadata: { stable: "yes", openclaw_turn_id: "turn-1", openclaw_turn_attempt: "1" },
      input: [firstUser] as never,
    },
    lastResponseId: "resp_1",
    lastResponseItems: [assistantOutput] as never,
  };
}

function nextRequest(phase = "final_answer"): ResponsesContinuationRequest {
  return {
    input: [
      firstUser,
      {
        type: "message",
        role: "assistant",
        phase,
        content: [{ type: "output_text", text: "answer", annotations: [] }],
      },
      { type: "message", role: "user", content: [{ type: "input_text", text: "second" }] },
    ] as never,
    metadata: { openclaw_turn_attempt: "2", openclaw_turn_id: "turn-2", stable: "yes" },
    store: true,
    model: "gpt-5.6-luna",
  };
}

function claim(params: {
  sessionId?: string;
  authorization?: string;
  turn?: string;
  request?: ResponsesContinuationRequest;
}) {
  return claimOpenAIResponsesHttpContinuation({
    sessionId: params.sessionId ?? "session-1",
    apiKey: "api-key",
    baseUrl: "https://api.openai.com/v1",
    headers: {
      Authorization: params.authorization ?? "Bearer tenant-a",
      traceparent: `trace-${params.turn ?? "1"}`,
      "x-openclaw-turn-id": `turn-${params.turn ?? "1"}`,
      "x-openclaw-turn-attempt": params.turn ?? "1",
      "x-stable-route": "route-a",
    },
    request: params.request ?? continuationState().lastRequest,
  });
}

afterEach(() => {
  cleanupSessionResources();
  vi.useRealTimers();
});

describe("OpenAI Responses continuation", () => {
  it("matches JSON wire semantics and provider-only assistant replay metadata", () => {
    const continued = resolveResponsesContinuationRequest(continuationState(), nextRequest());
    expect(continued).toMatchObject({
      continuationStatus: "continued",
      request: {
        previous_response_id: "resp_1",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: "second" }],
          },
        ],
      },
    });

    expect(
      resolveResponsesContinuationRequest(continuationState(), nextRequest("commentary"))
        .continuationStatus,
    ).toBe("history_changed");
    const explicit = { ...nextRequest(), previous_response_id: "resp_explicit" };
    expect(resolveResponsesContinuationRequest(continuationState(), explicit)).toEqual({
      request: explicit,
      continuationStatus: "explicit_previous_response_id",
    });
  });

  it("ignores turn correlation headers but isolates explicit authorization", () => {
    const first = claim({ turn: "1" });
    first?.commit(continuationState().lastRequest, {
      id: "resp_1",
      output: continuationState().lastResponseItems,
    });

    const sameTenant = claim({ turn: "2", request: nextRequest() });
    expect(sameTenant?.request.previous_response_id).toBe("resp_1");
    sameTenant?.commit(nextRequest(), { id: "resp_2", output: [] });

    const rotated = claim({
      turn: "3",
      authorization: "Bearer tenant-b",
      request: nextRequest(),
    });
    expect(rotated?.request.previous_response_id).toBeUndefined();
    rotated?.release();
  });

  it("grants one claim and prevents a concurrent non-owner from overwriting it", () => {
    const owner = claim({});
    expect(claim({})).toBeUndefined();

    owner?.commit(continuationState().lastRequest, {
      id: "resp_owner",
      output: continuationState().lastResponseItems,
    });
    expect(claim({ request: nextRequest() })?.request.previous_response_id).toBe("resp_owner");
  });

  it("prevents cleanup-time claims from resurrecting session state", () => {
    const stale = claim({});
    cleanupSessionResources("session-1");
    stale?.commit(continuationState().lastRequest, {
      id: "resp_stale",
      output: continuationState().lastResponseItems,
    });

    const next = claim({ request: nextRequest() });
    expect(next?.request.previous_response_id).toBeUndefined();
    next?.release();
  });

  it("restores the prior ready state on release instead of destroying it", () => {
    // Mirrors client-side model fallback: every candidate model shares the
    // same sessionId + connection identity (the cache key is not model-
    // scoped), so an unrelated model's claim must not permanently wipe out
    // a different, already-working model's continuation baseline just
    // because that unrelated attempt errors out before committing.
    const first = claim({});
    first?.commit(continuationState().lastRequest, {
      id: "resp_owner",
      output: continuationState().lastResponseItems,
    });

    const failedFallbackAttempt = claim({ request: nextRequest() });
    expect(failedFallbackAttempt?.request.previous_response_id).toBe("resp_owner");
    failedFallbackAttempt?.release();

    const retry = claim({ request: nextRequest() });
    expect(retry?.request.previous_response_id).toBe("resp_owner");
  });

  it("releasing a claim with no prior ready state just clears the slot", () => {
    const first = claim({});
    first?.release();

    const next = claim({ request: nextRequest() });
    expect(next?.request.previous_response_id).toBeUndefined();
    next?.release();
  });

  it("expires completed continuation state after the bounded idle TTL", () => {
    vi.useFakeTimers();
    const first = claim({});
    first?.commit(continuationState().lastRequest, {
      id: "resp_expiring",
      output: continuationState().lastResponseItems,
    });
    vi.advanceTimersByTime(5 * 60 * 1000 + 1);

    const next = claim({ request: nextRequest() });
    expect(next?.request.previous_response_id).toBeUndefined();
    next?.release();
  });

  it("reports no_previous_response for a first turn with no cached state", () => {
    expect(resolveResponsesContinuationRequest(undefined, nextRequest())).toEqual({
      request: nextRequest(),
      continuationStatus: "no_previous_response",
    });
  });

  it("reports request_changed when a non-input request field diverges from the cached baseline", () => {
    const changedModel = { ...nextRequest(), model: "gpt-5.6-terra" };
    expect(
      resolveResponsesContinuationRequest(continuationState(), changedModel).continuationStatus,
    ).toBe("request_changed");

    const changedStore = { ...nextRequest(), store: false };
    expect(
      resolveResponsesContinuationRequest(continuationState(), changedStore).continuationStatus,
    ).toBe("request_changed");
  });

  it("reports history_shorter when the replayed input is shorter than the cached baseline", () => {
    const shorterInput = {
      ...nextRequest(),
      input: [firstUser] as never,
    };
    expect(
      resolveResponsesContinuationRequest(continuationState(), shorterInput).continuationStatus,
    ).toBe("history_shorter");
  });

  it("reports history_changed when the replayed prefix diverges from the cached request input", () => {
    const differentFirstUser = {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "not the original first message" }],
    };
    const divergedPrefix = {
      ...nextRequest(),
      input: [
        differentFirstUser,
        {
          type: "message",
          role: "assistant",
          phase: "final_answer",
          content: [{ type: "output_text", text: "answer", annotations: [] }],
        },
        { type: "message", role: "user", content: [{ type: "input_text", text: "second" }] },
      ] as never,
    };
    expect(
      resolveResponsesContinuationRequest(continuationState(), divergedPrefix).continuationStatus,
    ).toBe("history_changed");
  });

  it("reports history_changed when the replayed assistant turn is a reasoning stub instead of the cached response item", () => {
    // Observed live against a custom OpenAI-Responses-compatible endpoint: the
    // session replayed the prior assistant turn as a bare `{type: "reasoning"}`
    // placeholder while the cached response was a plain assistant text message,
    // so the tail comparison correctly refuses to treat it as a continuation
    // rather than silently reusing the wrong response id.
    const reasoningStubTail = {
      ...nextRequest(),
      input: [
        firstUser,
        { type: "reasoning" },
        { type: "message", role: "user", content: [{ type: "input_text", text: "second" }] },
      ] as never,
    };
    expect(
      resolveResponsesContinuationRequest(continuationState(), reasoningStubTail)
        .continuationStatus,
    ).toBe("history_changed");
  });
});
