// Smoke coverage for session-history sanitization policy wiring.
import type { AgentMessage } from "openclaw/plugin-sdk/agent-core";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createSanitizeSessionHistoryHelpersMock,
  createSanitizeSessionHistoryProviderHookRuntimeMock,
  createSanitizeSessionHistoryProviderRuntimeMock,
  loadSanitizeSessionHistoryWithCleanMocks,
  makeMockSessionManager,
  makeSimpleUserMessages,
  type SanitizeSessionHistoryHarness,
  sanitizeSnapshotChangedOpenAIReasoning,
  sanitizeWithOpenAIResponses,
} from "./embedded-agent-runner.sanitize-session-history.test-harness.js";
import { makeZeroUsageSnapshot } from "./usage.js";

vi.mock("./embedded-agent-helpers.js", async () => await createSanitizeSessionHistoryHelpersMock());

// Provider runtime mocks keep this file focused on high-level policy routing
// while deeper replay-history behavior is covered in the main test suite.
vi.mock(
  "../plugins/provider-runtime.js",
  async () => await createSanitizeSessionHistoryProviderRuntimeMock(),
);
vi.mock(
  "../plugins/provider-hook-runtime.js",
  async () => await createSanitizeSessionHistoryProviderHookRuntimeMock(),
);

let sanitizeSessionHistory: SanitizeSessionHistoryHarness["sanitizeSessionHistory"];
let mockedHelpers: SanitizeSessionHistoryHarness["mockedHelpers"];

describe("sanitizeSessionHistory e2e smoke", () => {
  const mockSessionManager = makeMockSessionManager();
  const mockMessages = makeSimpleUserMessages();

  beforeAll(async () => {
    const harness = await loadSanitizeSessionHistoryWithCleanMocks();
    sanitizeSessionHistory = harness.sanitizeSessionHistory;
    mockedHelpers = harness.mockedHelpers;
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(mockedHelpers.sanitizeSessionMessagesImages).mockImplementation(async (msgs) => msgs);
  });

  it("passes simple user-only history through for google model APIs", async () => {
    const result = await sanitizeSessionHistory({
      messages: mockMessages,
      modelApi: "google-generative-ai",
      provider: "google-vertex",
      sessionManager: mockSessionManager,
      sessionId: "test-session",
    });

    expect(result).toEqual(mockMessages);
  });

  it("passes simple user-only history through for openai-responses", async () => {
    const result = await sanitizeWithOpenAIResponses({
      sanitizeSessionHistory,
      messages: mockMessages,
      sessionManager: mockSessionManager,
    });

    expect(result).toEqual(mockMessages);
  });

  it("preserves an already-valid call_id|fc_id pair for an unowned openai-responses provider (no double-sanitization)", async () => {
    // Custom/self-hosted openai-responses endpoints have no owning runtime
    // plugin, so they fall through to the unowned-provider replay fallback,
    // which sets sanitizeToolCallIds: true. A well-formed call_id|fc_id pair
    // needs no rewriting from the Responses-aware normalizer; running the
    // generic Cloud-Code-Assist-style sanitizer on top of it anyway strips the
    // "|" separator and re-hashes the id into a shape the provider never
    // returned, which silently breaks HTTP continuation's replay-vs-cache
    // comparison for every multi-round tool-calling turn.
    const toolCallId = "call_29b14f4fbcd049c2b37bf0cdb10f7263|fc_29b14f4fbcd049c2b37bf0cdb10f7263";
    const messages = [
      {
        role: "assistant",
        content: [
          // A replayable reasoning signature keeps the call_id|fc_id pairing
          // intact through downgradeOpenAIFunctionCallReasoningPairs, so this
          // test isolates the later, unconditional sanitizeToolCallIds pass.
          {
            type: "thinking",
            thinking: "reasoning",
            thinkingSignature: { id: "rs_test", type: "reasoning" },
          },
          { type: "toolCall", id: toolCallId, name: "noop", arguments: {} },
        ],
      },
      {
        role: "toolResult",
        toolCallId,
        toolName: "noop",
        content: [{ type: "text", text: "ok" }],
        isError: false,
      },
    ] as unknown as AgentMessage[];

    const result = await sanitizeWithOpenAIResponses({
      sanitizeSessionHistory,
      messages,
      sessionManager: mockSessionManager,
    });

    const assistant = result[0] as { content?: Array<{ type?: string; id?: string }> };
    const toolCall = assistant.content?.find((block) => block.type === "toolCall");
    expect(toolCall?.id).toBe(toolCallId);

    const toolResult = result[1] as { toolCallId?: string };
    expect(toolResult.toolCallId).toBe(toolCallId);
  });

  it("downgrades openai reasoning blocks when the model snapshot changed", async () => {
    // Snapshot changes are the public safety boundary: reasoning that was valid
    // for one provider must be replayed as text-only when the model family moves.
    const result = await sanitizeSnapshotChangedOpenAIReasoning({
      sanitizeSessionHistory,
    });

    expect(result).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "answer" }],
        usage: makeZeroUsageSnapshot(),
      },
    ]);
  });
});
