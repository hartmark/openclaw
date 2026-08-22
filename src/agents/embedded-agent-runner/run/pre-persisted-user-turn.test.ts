// Reconciliation tests cover the heartbeat-continuation replay-stability fix
// (docs/plan/heartbeat-continuation-replay-stability.md): orphan-repair's tree
// walk must not misidentify a turn's own just-persisted message as a stale
// prior turn, whether proven via same-process entry id or a durable
// idempotencyKey.
import { describe, expect, it } from "vitest";
import type { AgentMessage } from "../../../../packages/agent-core/src/types.js";
import { reconcilePrePersistedCurrentUserTurn } from "./pre-persisted-user-turn.js";

function userMessage(overrides: Partial<AgentMessage> = {}): AgentMessage {
  return { role: "user", content: "hello", timestamp: 0, ...overrides } as AgentMessage;
}

describe("reconcilePrePersistedCurrentUserTurn", () => {
  describe("entryId path", () => {
    it("reconciles and strips a matching in-memory user tail when entry ids match, without an idempotencyKey", () => {
      const tail = userMessage();
      const activeSession = { agent: { state: { messages: [tail] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage(),
        durableUserTurnMessage: userMessage(),
        userTurnAlreadyPersisted: true,
        currentEntryId: "entry-1",
        candidateEntryId: "entry-1",
      });

      expect(reconciled).toBe(true);
      expect(activeSession.agent.state.messages).toEqual([]);
    });

    it("reconciles without stripping when the in-memory tail is not a user message", () => {
      const tail = { role: "assistant", content: [], timestamp: 0 } as unknown as AgentMessage;
      const activeSession = { agent: { state: { messages: [tail] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage(),
        durableUserTurnMessage: userMessage(),
        userTurnAlreadyPersisted: true,
        currentEntryId: "entry-1",
        candidateEntryId: "entry-1",
      });

      expect(reconciled).toBe(true);
      expect(activeSession.agent.state.messages).toEqual([tail]);
    });

    it("falls through to the idempotencyKey path when entry ids differ", () => {
      const activeSession = { agent: { state: { messages: [] as AgentMessage[] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage(),
        durableUserTurnMessage: userMessage(),
        userTurnAlreadyPersisted: true,
        currentEntryId: "entry-1",
        candidateEntryId: "entry-2",
      });

      expect(reconciled).toBe(false);
    });

    it("falls through to the idempotencyKey path when currentEntryId is unavailable", () => {
      const activeSession = { agent: { state: { messages: [] as AgentMessage[] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage(),
        durableUserTurnMessage: userMessage(),
        userTurnAlreadyPersisted: true,
        candidateEntryId: "entry-2",
      });

      expect(reconciled).toBe(false);
    });
  });

  describe("idempotencyKey path", () => {
    it("rejects when the current turn has no idempotencyKey", () => {
      const activeSession = { agent: { state: { messages: [] as AgentMessage[] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage(),
        durableUserTurnMessage: userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>),
        userTurnAlreadyPersisted: true,
      });

      expect(reconciled).toBe(false);
    });

    it("rejects when not yet persisted and the durable candidate's key does not match", () => {
      const activeSession = { agent: { state: { messages: [] as AgentMessage[] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>),
        durableUserTurnMessage: userMessage({ idempotencyKey: "key-2" } as Partial<AgentMessage>),
        userTurnAlreadyPersisted: false,
      });

      expect(reconciled).toBe(false);
    });

    it("reconciles and strips a matching in-memory user tail (same-process persisted case)", () => {
      const tail = userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>);
      const activeSession = { agent: { state: { messages: [tail] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>),
        durableUserTurnMessage: undefined,
        userTurnAlreadyPersisted: true,
      });

      expect(reconciled).toBe(true);
      expect(activeSession.agent.state.messages).toEqual([]);
    });

    it("reconciles without an in-memory tail match when the durable candidate's key matches (cross-restart case)", () => {
      const activeSession = { agent: { state: { messages: [] as AgentMessage[] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>),
        durableUserTurnMessage: userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>),
        userTurnAlreadyPersisted: false,
      });

      expect(reconciled).toBe(true);
      expect(activeSession.agent.state.messages).toEqual([]);
    });

    it("rejects when persisted but neither the tail nor the durable candidate's key match", () => {
      const tail = userMessage({ idempotencyKey: "other-key" } as Partial<AgentMessage>);
      const activeSession = { agent: { state: { messages: [tail] } } };

      const reconciled = reconcilePrePersistedCurrentUserTurn({
        activeSession,
        currentUserTurnMessage: userMessage({ idempotencyKey: "key-1" } as Partial<AgentMessage>),
        durableUserTurnMessage: userMessage({ idempotencyKey: "key-2" } as Partial<AgentMessage>),
        userTurnAlreadyPersisted: true,
      });

      expect(reconciled).toBe(false);
      expect(activeSession.agent.state.messages).toEqual([tail]);
    });
  });
});
