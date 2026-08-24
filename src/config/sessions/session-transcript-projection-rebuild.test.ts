import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useAutoCleanupTempDirTracker } from "../../../test/helpers/temp-dir.js";
import { STREAM_ERROR_FALLBACK_TEXT } from "../../agents/stream-message-shared.js";
import { HEARTBEAT_PROMPT } from "../../auto-reply/heartbeat.js";
import {
  closeOpenClawAgentDatabasesForTest,
  openOpenClawAgentDatabase,
  runOpenClawAgentWriteTransaction,
} from "../../state/openclaw-agent-db.js";
import { closeOpenClawStateDatabaseForTest } from "../../state/openclaw-state-db.js";
import {
  appendTranscriptEvent,
  appendTranscriptMessage,
  persistSessionTranscriptTurn,
  upsertSessionEntryCore,
} from "./session-accessor.js";
import {
  appendEligibleSessionTranscriptDisplayRowInTransaction,
  prepareSessionTranscriptDisplayRows,
} from "./session-transcript-display.js";
import {
  buildSessionTranscriptProjection,
  type SessionTranscriptProjectionSourceRow,
} from "./session-transcript-projection-rebuild.js";
import { reconcileSessionTranscriptDisplayProjection } from "./session-transcript-reconcile.js";

const SESSION_ID = "projection-session";
const tempDirs = useAutoCleanupTempDirTracker(afterEach);

function row(
  seq: number,
  event: Record<string, unknown>,
  createdAt = seq * 1_000,
): SessionTranscriptProjectionSourceRow {
  return { createdAt, event, seq };
}

function projection(rows: SessionTranscriptProjectionSourceRow[]) {
  return buildSessionTranscriptProjection({
    rows,
    sessionId: SESSION_ID,
    sourceTranscriptUpdatedAt: 42,
  });
}

describe("canonical session transcript projection", () => {
  let env: NodeJS.ProcessEnv;
  const scope = {
    agentId: "main",
    sessionId: SESSION_ID,
    sessionKey: "agent:main:projection-session",
  };

  beforeEach(() => {
    env = {
      ...process.env,
      OPENCLAW_STATE_DIR: tempDirs.make("openclaw-transcript-projection-"),
    };
  });

  afterEach(() => {
    closeOpenClawAgentDatabasesForTest();
    closeOpenClawStateDatabaseForTest();
  });

  function readProjectionSourceRows(): SessionTranscriptProjectionSourceRow[] {
    return openOpenClawAgentDatabase({ agentId: scope.agentId, env })
      .db.prepare(
        "SELECT seq, event_json, created_at FROM transcript_events WHERE session_id = ? ORDER BY seq",
      )
      .all(scope.sessionId)
      .map((entry) => {
        const sourceRow = entry as { created_at: number; event_json: string; seq: number };
        return {
          createdAt: sourceRow.created_at,
          event: JSON.parse(sourceRow.event_json),
          seq: sourceRow.seq,
        };
      });
  }

  function readDisplayRows(sessionId = scope.sessionId) {
    return openOpenClawAgentDatabase({ agentId: scope.agentId, env })
      .db.prepare(
        "SELECT display_ordinal, kind, source_event_seq FROM session_transcript_display_rows WHERE session_id = ? ORDER BY display_ordinal",
      )
      .all(sessionId);
  }

  async function appendEligibleDisplayRowDirectly(event: Record<string, unknown>) {
    const sessionId = `${scope.sessionId}-direct`;
    const sessionKey = `${scope.sessionKey}-direct`;
    await upsertSessionEntryCore(
      { agentId: scope.agentId, env, sessionKey },
      { sessionId, updatedAt: 1 },
    );
    runOpenClawAgentWriteTransaction(
      (database) => {
        database.db
          .prepare(
            "INSERT INTO transcript_events (session_id, seq, event_json, created_at) VALUES (?, 0, ?, 1)",
          )
          .run(sessionId, JSON.stringify(event));
        appendEligibleSessionTranscriptDisplayRowInTransaction(database.db, {
          event,
          seq: 0,
          sessionId,
        });
      },
      { agentId: scope.agentId, env },
    );
    return readDisplayRows(sessionId);
  }

  it("projects one deterministic active branch for both rebuild owners", () => {
    const result = projection([
      row(0, { id: SESSION_ID, type: "session", version: 3 }),
      row(1, {
        id: "root",
        message: { content: "root text", role: "user" },
        parentId: null,
        type: "message",
      }),
      row(2, {
        id: "abandoned",
        message: { content: "abandoned text", role: "assistant" },
        parentId: "root",
        type: "message",
      }),
      row(3, {
        id: "active",
        message: { content: "active text", role: "assistant" },
        parentId: "root",
        type: "message",
      }),
    ]);

    expect(result).toMatchObject({
      activeEventCount: 2,
      activeMessageCount: 2,
      leafEventId: "active",
      sessionId: SESSION_ID,
      sourceIndexedSeq: 3,
      sourceTranscriptUpdatedAt: 42,
    });
    expect(result.activeRows).toEqual([
      { activePosition: 0, eventSeq: 1, messagePosition: 0 },
      { activePosition: 1, eventSeq: 3, messagePosition: 1 },
    ]);
    expect(
      result.displayRows.map(({ displayOrdinal, kind, sourceEventSeq }) => ({
        displayOrdinal,
        kind,
        sourceEventSeq,
      })),
    ).toEqual([
      { displayOrdinal: 0, kind: "user", sourceEventSeq: 1 },
      { displayOrdinal: 1, kind: "assistant", sourceEventSeq: 3 },
    ]);
    expect(result.ftsRows).toEqual([
      { messageId: "root", role: "user", text: "root text", timestamp: 1_000 },
      { messageId: "active", role: "assistant", text: "active text", timestamp: 3_000 },
    ]);
  });

  it("matches incremental append and rebuild after excluding the header and abandoned branch", async () => {
    await persistSessionTranscriptTurn(
      { ...scope, env },
      {
        messages: [
          { eventId: "root", parentId: null, message: { role: "user", content: "root" } },
          {
            eventId: "abandoned",
            parentId: "root",
            message: { role: "assistant", content: "abandoned" },
          },
          {
            eventId: "active",
            parentId: "root",
            message: { role: "assistant", content: "active" },
          },
        ].map((message) => Object.assign(message, { maintainDisplayProjection: true })),
        touchSessionEntry: false,
      },
    );
    await reconcileSessionTranscriptDisplayProjection({ agentId: scope.agentId, env });

    const planned = prepareSessionTranscriptDisplayRows(readProjectionSourceRows()).map(
      ({ displayOrdinal, kind, sourceEventSeq }) => ({
        display_ordinal: displayOrdinal,
        kind,
        source_event_seq: sourceEventSeq,
      }),
    );
    expect(readDisplayRows()).toEqual(planned);
    expect(planned).toEqual([
      { display_ordinal: 0, kind: "user", source_event_seq: 1 },
      { display_ordinal: 1, kind: "assistant", source_event_seq: 3 },
    ]);
  });

  it.each([
    {
      event: { type: "message", message: { role: "user", content: "hello" } },
      expected: "user",
      name: "plain user",
    },
    {
      event: { type: "message", message: { role: "assistant", content: "hello" } },
      expected: "assistant",
      name: "plain assistant",
    },
    {
      event: { type: "compaction" },
      expected: "compaction",
      name: "compaction boundary",
    },
    {
      event: { type: "reset" },
      expected: "reset",
      name: "reset boundary",
    },
    {
      event: { type: "message", message: { role: "user", content: HEARTBEAT_PROMPT } },
      expected: "opaque",
      name: "heartbeat candidate",
    },
    {
      event: {
        type: "message",
        message: {
          role: "assistant",
          content: [{ type: "text", text: STREAM_ERROR_FALLBACK_TEXT }],
          stopReason: "error",
        },
      },
      expected: "opaque",
      name: "stream error candidate",
    },
    {
      event: {
        type: "message",
        message: {
          role: "assistant",
          content: "mirrored message",
          openclawMessageToolMirror: {},
        },
      },
      expected: "opaque",
      name: "message tool candidate",
    },
    {
      event: {
        type: "message",
        message: {
          role: "assistant",
          content: "spoken supplement",
          openclawTtsSupplement: { spokenText: "hello" },
        },
      },
      expected: "opaque",
      name: "TTS supplement candidate",
    },
    {
      event: {
        type: "message",
        message: {
          role: "assistant",
          content: "canvas preview",
          details: { mcpAppPreview: { view: { id: "view-1" } } },
        },
      },
      expected: "opaque",
      name: "canvas candidate",
    },
    {
      event: {
        type: "message",
        message: { role: "toolResult", content: "tool result" },
      },
      expected: "opaque",
      name: "tool record",
    },
    {
      event: { type: "custom", id: "custom-1", parentId: null },
      expected: "opaque",
      name: "unknown canonical entry",
    },
  ])("uses the canonical display classifier for $name rows", async ({ event, expected }) => {
    const persistedEvent = { id: "event-1", parentId: null, ...event };
    if (event.type === "message" && event.message) {
      await appendTranscriptMessage(
        { ...scope, env },
        {
          eventId: "event-1",
          maintainDisplayProjection: true,
          message: event.message,
          parentId: null,
        },
      );
    } else {
      await appendTranscriptEvent({ ...scope, env }, persistedEvent);
    }
    await reconcileSessionTranscriptDisplayProjection({ agentId: scope.agentId, env });

    const directExpected = prepareSessionTranscriptDisplayRows([
      { event: persistedEvent, seq: 0 },
    ]).map(({ displayOrdinal, kind, sourceEventSeq }) => ({
      display_ordinal: displayOrdinal,
      kind,
      source_event_seq: sourceEventSeq,
    }));
    expect(await appendEligibleDisplayRowDirectly(persistedEvent)).toEqual(directExpected);

    const planned = prepareSessionTranscriptDisplayRows(readProjectionSourceRows()).map(
      ({ displayOrdinal, kind, sourceEventSeq }) => ({
        display_ordinal: displayOrdinal,
        kind,
        source_event_seq: sourceEventSeq,
      }),
    );
    expect(readDisplayRows()).toEqual(planned);
    expect(planned).toMatchObject([{ kind: expected }]);
  });

  it("keeps persisted row timestamps for timestamp-less and invalid-timestamp messages", () => {
    const result = projection([
      row(0, { id: SESSION_ID, type: "session", version: 3 }),
      row(
        1,
        {
          id: "old-user",
          message: { content: [{ text: "old content", type: "text" }], role: "user" },
          parentId: null,
          type: "message",
        },
        1_700_000_000_000,
      ),
      row(
        2,
        {
          id: "invalid-timestamp",
          message: { content: "still old", role: "assistant" },
          parentId: "old-user",
          timestamp: "not a date",
          type: "message",
        },
        1_700_000_001_000,
      ),
    ]);

    expect(result.ftsRows.map(({ messageId, timestamp }) => ({ messageId, timestamp }))).toEqual([
      { messageId: "old-user", timestamp: 1_700_000_000_000 },
      { messageId: "invalid-timestamp", timestamp: 1_700_000_001_000 },
    ]);
  });

  it("respects a leaf-control rewind without indexing the abandoned continuation", () => {
    const result = projection([
      row(0, { id: SESSION_ID, type: "session", version: 3 }),
      row(1, {
        id: "root",
        message: { content: "keep", role: "user" },
        parentId: null,
        type: "message",
      }),
      row(2, {
        id: "abandoned",
        message: { content: "remove", role: "assistant" },
        parentId: "root",
        type: "message",
      }),
      row(3, {
        appendParentId: "root",
        id: "rewind",
        parentId: "abandoned",
        targetId: "root",
        type: "leaf",
      }),
    ]);

    expect(result.leafEventId).toBe("root");
    expect(result.activeRows).toEqual([{ activePosition: 0, eventSeq: 1, messagePosition: 0 }]);
    expect(result.ftsRows.map((entry) => entry.messageId)).toEqual(["root"]);
  });

  it("keeps legacy flat-message ordering and searchable identities", () => {
    const result = projection([
      row(0, { id: SESSION_ID, type: "session", version: 1 }),
      row(1, {
        id: "legacy-user",
        message: { content: "first", role: "user" },
        type: "message",
      }),
      row(2, {
        id: "legacy-assistant",
        message: { content: "second", role: "assistant" },
        type: "message",
      }),
    ]);

    expect(result.activeRows).toEqual([
      { activePosition: 0, eventSeq: 1, messagePosition: 0 },
      { activePosition: 1, eventSeq: 2, messagePosition: 1 },
    ]);
    expect(result.ftsRows.map((entry) => entry.messageId)).toEqual([
      "legacy-user",
      "legacy-assistant",
    ]);
    expect(result.sourceIndexedSeq).toBe(2);
  });
});
