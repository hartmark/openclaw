import { randomUUID } from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import { asOptionalRecord as readRecord } from "@openclaw/normalization-core/record-coerce";
import type { Generated } from "kysely";
import { STREAM_ERROR_FALLBACK_TEXT } from "../../agents/stream-message-shared.js";
import { HEARTBEAT_PROMPT } from "../../auto-reply/heartbeat.js";
import {
  executeSqliteQuerySync,
  executeSqliteQueryTakeFirstSync,
  getNodeSqliteKysely,
} from "../../infra/kysely-sync.js";
import { runSqliteDeferredTransactionSync } from "../../infra/sqlite-transaction.js";
import type { DB as OpenClawAgentKyselyDatabase } from "../../state/openclaw-agent-db.generated.js";
import {
  ensureOpenClawAgentDisplayRowSchema,
  SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE,
  SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE,
  validateOpenClawAgentDisplayRowSchema,
} from "../../state/openclaw-agent-display-row-schema.js";
import {
  isCanonicalSessionTranscriptEntry,
  parseSessionTranscriptTreeEntry,
} from "./transcript-tree.js";
import { selectVisibleTranscriptEventEntries } from "./transcript-visible-events.js";

const SESSION_TRANSCRIPT_DISPLAY_ROW_VERSION = 1;
const SESSION_TRANSCRIPT_DISPLAY_PAGE_MAX_ROWS = 200;

type SessionTranscriptDisplayRowKind = "assistant" | "compaction" | "opaque" | "reset" | "user";

type PlannedSessionTranscriptDisplayRow = {
  kind: SessionTranscriptDisplayRowKind;
  sourceEventSeq: number;
};

export type PreparedSessionTranscriptDisplayRow = PlannedSessionTranscriptDisplayRow & {
  displayOrdinal: number;
  revision: number;
  rowId: string;
  rowVersion: number;
};

type SessionTranscriptDisplayState = {
  generation: string;
  indexedSeq: number;
  needsRebuild: boolean;
  rowCount: number;
  updatedAt: number;
};

type SessionTranscriptDisplayReadResult =
  | {
      generation: string;
      kind: "ready";
      nextOrdinal?: number;
      rows: Array<{
        displayOrdinal: number;
        kind: SessionTranscriptDisplayRowKind;
        revision: number;
        rowId: string;
        rowVersion: number;
        sourceEventSeq: number;
      }>;
    }
  | { generation: string | null; kind: "reset" };

type SessionTranscriptDisplayReadParams = {
  expectedGeneration: string;
  fromOrdinal: number | "tail";
  limit: number;
};

type DisplayRowDatabase = Omit<
  Pick<
    OpenClawAgentKyselyDatabase,
    | "session_transcript_display_rows"
    | "session_transcript_display_state"
    | "session_windows"
    | "transcript_events"
  >,
  "session_transcript_display_rows"
> & {
  session_transcript_display_rows: OpenClawAgentKyselyDatabase["session_transcript_display_rows"] & {
    rowid: Generated<number>;
  };
};

type DisplayDeleteChunkResult = {
  hasMore: boolean;
  owned: boolean;
};

function getDisplayKysely(db: DatabaseSync) {
  return getNodeSqliteKysely<DisplayRowDatabase>(db);
}

function createDisplayGeneration(): string {
  return randomUUID().replaceAll("-", "");
}

function createDisplayRowId(): string {
  return randomUUID();
}

function readMessageText(message: Record<string, unknown>): string | undefined {
  if (typeof message.content === "string") {
    return message.content;
  }
  if (!Array.isArray(message.content)) {
    return typeof message.text === "string" ? message.text : undefined;
  }
  const text = message.content.flatMap((block) => {
    const entry = readRecord(block);
    if (!entry) {
      return [];
    }
    return typeof entry.text === "string" &&
      (entry.type === "text" || entry.type === "input_text" || entry.type === "output_text")
      ? [entry.text]
      : [];
  });
  return text.length > 0 ? text.join("\n") : undefined;
}

function hasNonTextContent(message: Record<string, unknown>): boolean {
  return (
    Array.isArray(message.content) &&
    message.content.some((block) => {
      const entry = readRecord(block);
      if (!entry) {
        return true;
      }
      const type = entry.type;
      return type !== "text" && type !== "input_text" && type !== "output_text";
    })
  );
}

function hasCanvasSource(message: Record<string, unknown>): boolean {
  const hasPreview = (value: unknown): boolean =>
    Boolean(
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      ("mcpAppPreview" in value || "mcpApp" in value || "canvas" in value),
    );
  if (hasPreview(message.details)) {
    return true;
  }
  return (
    Array.isArray(message.content) &&
    message.content.some((block) => {
      const entry = readRecord(block);
      return Boolean(entry && (hasPreview(entry.details) || entry.type === "canvas"));
    })
  );
}

function requiresStatefulDisplayProjection(message: Record<string, unknown>): boolean {
  const role = message.role;
  const text = readMessageText(message)?.trim();
  if (
    role === "user" &&
    text &&
    (text === HEARTBEAT_PROMPT || text.startsWith(`${HEARTBEAT_PROMPT}\n`))
  ) {
    return true;
  }
  if (
    role === "assistant" &&
    message.stopReason === "error" &&
    text === STREAM_ERROR_FALLBACK_TEXT
  ) {
    return true;
  }
  return (
    Object.hasOwn(message, "openclawMessageToolMirror") ||
    Object.hasOwn(message, "openclawTtsSupplement") ||
    hasCanvasSource(message) ||
    hasNonTextContent(message)
  );
}

export function hasTranscriptMessage(event: unknown): boolean {
  const record = readRecord(event);
  return record !== undefined && Object.hasOwn(record, "message") && record.message !== undefined;
}

export function shouldProjectActiveEvent(event: unknown): boolean {
  const record = readRecord(event);
  if (!record) {
    return false;
  }
  if (record.type === "session") {
    return false;
  }
  return (
    isCanonicalSessionTranscriptEntry(event) ||
    parseSessionTranscriptTreeEntry(event) !== undefined ||
    hasTranscriptMessage(event)
  );
}

export function isSessionTranscriptDisplayBoundary(event: unknown): boolean {
  const record = readRecord(event);
  if (!record) {
    return false;
  }
  const type = record.type;
  return type === "compaction" || type === "reset";
}

/** Classifies one independently displayable source without applying stateful row semantics. */
function planSessionTranscriptDisplayRow(
  event: unknown,
  sourceEventSeq: number,
): PlannedSessionTranscriptDisplayRow | undefined {
  if (!shouldProjectActiveEvent(event)) {
    return undefined;
  }
  const record = readRecord(event);
  if (!record) {
    return undefined;
  }
  if (record.type === "compaction" || record.type === "reset") {
    return { kind: record.type, sourceEventSeq };
  }
  const message = readRecord(record.message);
  if (!message) {
    return { kind: "opaque", sourceEventSeq };
  }
  const role = message.role;
  if ((role === "user" || role === "assistant") && !requiresStatefulDisplayProjection(message)) {
    return { kind: role, sourceEventSeq };
  }
  return { kind: "opaque", sourceEventSeq };
}

/** Builds display rows from the same canonical visible path used by active transcript reads. */
function buildSessionTranscriptDisplayRows(
  rows: readonly { event: unknown; seq: number }[],
): PlannedSessionTranscriptDisplayRow[] {
  const events = rows.map((row) => row.event);
  return selectVisibleTranscriptEventEntries(events).flatMap((entry) => {
    const source = rows[entry.seq - 1];
    return source ? (planSessionTranscriptDisplayRow(entry.event, source.seq) ?? []) : [];
  });
}

export function prepareSessionTranscriptDisplayRows(
  rows: readonly { event: unknown; seq: number }[],
): PreparedSessionTranscriptDisplayRow[] {
  return buildSessionTranscriptDisplayRows(rows).map((row, displayOrdinal) => ({
    displayOrdinal,
    kind: row.kind,
    revision: 1,
    rowId: createDisplayRowId(),
    rowVersion: SESSION_TRANSCRIPT_DISPLAY_ROW_VERSION,
    sourceEventSeq: row.sourceEventSeq,
  }));
}

export function readSessionTranscriptDisplayState(
  db: DatabaseSync,
  sessionId: string,
): SessionTranscriptDisplayState | undefined {
  ensureOpenClawAgentDisplayRowSchema(db);
  const row = executeSqliteQueryTakeFirstSync(
    db,
    getDisplayKysely(db)
      .selectFrom(SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)
      .select(["generation", "indexed_seq", "needs_rebuild", "row_count", "updated_at"])
      .where("session_id", "=", sessionId),
  );
  return row
    ? {
        generation: row.generation,
        indexedSeq: row.indexed_seq,
        needsRebuild: row.needs_rebuild !== 0,
        rowCount: row.row_count,
        updatedAt: row.updated_at,
      }
    : undefined;
}

function writeDisplayState(
  db: DatabaseSync,
  sessionId: string,
  state: SessionTranscriptDisplayState,
): void {
  executeSqliteQuerySync(
    db,
    getDisplayKysely(db)
      .insertInto(SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)
      .values({
        generation: state.generation,
        indexed_seq: state.indexedSeq,
        needs_rebuild: state.needsRebuild ? 1 : 0,
        row_count: state.rowCount,
        session_id: sessionId,
        updated_at: state.updatedAt,
      })
      .onConflict((conflict) =>
        conflict.column("session_id").doUpdateSet({
          generation: state.generation,
          indexed_seq: state.indexedSeq,
          needs_rebuild: state.needsRebuild ? 1 : 0,
          row_count: state.rowCount,
          updated_at: state.updatedAt,
        }),
      ),
  );
}

/** Rotates one display generation and makes every reader reset until reconcile publishes it. */
export function invalidateSessionTranscriptDisplayInTransaction(
  db: DatabaseSync,
  sessionId: string,
): string {
  ensureOpenClawAgentDisplayRowSchema(db);
  const state = readSessionTranscriptDisplayState(db, sessionId);
  const generation = createDisplayGeneration();
  writeDisplayState(db, sessionId, {
    generation,
    indexedSeq: state?.indexedSeq ?? -1,
    needsRebuild: true,
    rowCount: state?.rowCount ?? 0,
    updatedAt: Date.now(),
  });
  return generation;
}

/** Invalidates an adopted display projection without materializing absent storage. */
export function invalidateExistingSessionTranscriptDisplayInTransaction(
  db: DatabaseSync,
  sessionId: string,
): boolean {
  if (!validateOpenClawAgentDisplayRowSchema(db)) {
    return false;
  }
  const result = executeSqliteQuerySync(
    db,
    getDisplayKysely(db)
      .updateTable(SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)
      .set({
        generation: createDisplayGeneration(),
        needs_rebuild: 1,
        updated_at: Date.now(),
      })
      .where("session_id", "=", sessionId),
  );
  return result.numAffectedRows === 1n;
}

/** Extends one ready display generation after active-path eligibility is already proven. */
export function appendEligibleSessionTranscriptDisplayRowInTransaction(
  db: DatabaseSync,
  params: { event: unknown; seq: number; sessionId: string },
): boolean {
  ensureOpenClawAgentDisplayRowSchema(db);
  const state = readSessionTranscriptDisplayState(db, params.sessionId);
  if (state?.needsRebuild) {
    return true;
  }
  if (state && params.seq !== state.indexedSeq + 1) {
    invalidateSessionTranscriptDisplayInTransaction(db, params.sessionId);
    return true;
  }
  if (!state && params.seq !== 0) {
    invalidateSessionTranscriptDisplayInTransaction(db, params.sessionId);
    return true;
  }
  const generation = state?.generation ?? createDisplayGeneration();
  const planned = planSessionTranscriptDisplayRow(params.event, params.seq);
  const rowCount = state?.rowCount ?? 0;
  if (!state) {
    writeDisplayState(db, params.sessionId, {
      generation,
      indexedSeq: params.seq - 1,
      needsRebuild: false,
      rowCount: 0,
      updatedAt: Date.now(),
    });
  }
  if (planned) {
    executeSqliteQuerySync(
      db,
      getDisplayKysely(db).insertInto(SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE).values({
        display_ordinal: rowCount,
        kind: planned.kind,
        revision: 1,
        row_id: createDisplayRowId(),
        row_version: SESSION_TRANSCRIPT_DISPLAY_ROW_VERSION,
        session_id: params.sessionId,
        source_event_seq: planned.sourceEventSeq,
      }),
    );
  }
  writeDisplayState(db, params.sessionId, {
    generation,
    indexedSeq: params.seq,
    needsRebuild: false,
    rowCount: rowCount + (planned ? 1 : 0),
    updatedAt: Date.now(),
  });
  return false;
}

export function claimSessionTranscriptDisplayInTransaction(
  db: DatabaseSync,
  params: {
    claimId: number;
    generation: string;
    sessionId: string;
  },
): boolean {
  const state = readSessionTranscriptDisplayState(db, params.sessionId);
  if (!state) {
    writeDisplayState(db, params.sessionId, {
      generation: params.generation,
      indexedSeq: -1,
      needsRebuild: true,
      rowCount: 0,
      updatedAt: params.claimId,
    });
    return true;
  }
  if (state.generation !== params.generation) {
    return false;
  }
  writeDisplayState(db, params.sessionId, {
    ...state,
    needsRebuild: true,
    updatedAt: params.claimId,
  });
  return true;
}

function displayClaimIsOwned(
  db: DatabaseSync,
  params: { claimId: number; generation: string; sessionId: string },
): boolean {
  const state = readSessionTranscriptDisplayState(db, params.sessionId);
  return Boolean(
    state?.needsRebuild &&
    state.generation === params.generation &&
    state.updatedAt === params.claimId,
  );
}

export function deleteSessionTranscriptDisplayChunkInTransaction(
  db: DatabaseSync,
  params: {
    claimId: number;
    generation: string;
    maxRows: number;
    sessionId: string;
  },
): DisplayDeleteChunkResult {
  if (!displayClaimIsOwned(db, params)) {
    return { hasMore: false, owned: false };
  }
  const kysely = getDisplayKysely(db);
  const deleted = Number(
    executeSqliteQuerySync(
      db,
      kysely
        .deleteFrom(SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE)
        .where(
          "rowid",
          "in",
          kysely
            .selectFrom(SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE)
            .select("rowid")
            .where("session_id", "=", params.sessionId)
            .limit(params.maxRows),
        ),
    ).numAffectedRows ?? 0n,
  );
  return { hasMore: deleted === params.maxRows, owned: true };
}

export function appendSessionTranscriptDisplayChunkInTransaction(
  db: DatabaseSync,
  params: {
    claimId: number;
    generation: string;
    rows: readonly PreparedSessionTranscriptDisplayRow[];
    sessionId: string;
  },
): boolean {
  if (!displayClaimIsOwned(db, params)) {
    return false;
  }
  if (params.rows.length > 0) {
    executeSqliteQuerySync(
      db,
      getDisplayKysely(db)
        .insertInto(SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE)
        .values(
          params.rows.map((row) => ({
            display_ordinal: row.displayOrdinal,
            kind: row.kind,
            revision: row.revision,
            row_id: row.rowId,
            row_version: row.rowVersion,
            session_id: params.sessionId,
            source_event_seq: row.sourceEventSeq,
          })),
        ),
    );
  }
  return true;
}

export function finalizeSessionTranscriptDisplayInTransaction(
  db: DatabaseSync,
  params: {
    claimId: number;
    generation: string;
    rowCount: number;
    sessionId: string;
    sourceIndexedSeq: number;
  },
): boolean {
  if (!displayClaimIsOwned(db, params)) {
    return false;
  }
  const result = executeSqliteQuerySync(
    db,
    getDisplayKysely(db)
      .updateTable(SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)
      .set({
        indexed_seq: params.sourceIndexedSeq,
        needs_rebuild: 0,
        row_count: params.rowCount,
        updated_at: Date.now(),
      })
      .where("session_id", "=", params.sessionId)
      .where("generation", "=", params.generation)
      .where("needs_rebuild", "!=", 0)
      .where("updated_at", "=", params.claimId),
  );
  return result.numAffectedRows === 1n;
}

function normalizeDisplayPageLimit(limit: number): number {
  if (!Number.isFinite(limit)) {
    return SESSION_TRANSCRIPT_DISPLAY_PAGE_MAX_ROWS;
  }
  return Math.max(1, Math.min(SESSION_TRANSCRIPT_DISPLAY_PAGE_MAX_ROWS, Math.floor(limit)));
}

function parseDisplayRowKind(value: string): SessionTranscriptDisplayRowKind {
  if (
    value === "assistant" ||
    value === "compaction" ||
    value === "opaque" ||
    value === "reset" ||
    value === "user"
  ) {
    return value;
  }
  throw new Error(`Unexpected transcript display-row kind: ${value}`);
}

function readSessionTranscriptDisplayRowsSnapshot(
  db: DatabaseSync,
  sessionId: string,
  params: SessionTranscriptDisplayReadParams,
): SessionTranscriptDisplayReadResult {
  const state = readSessionTranscriptDisplayState(db, sessionId);
  const latest = executeSqliteQueryTakeFirstSync(
    db,
    getDisplayKysely(db)
      .selectFrom("transcript_events")
      .select("seq")
      .where("session_id", "=", sessionId)
      .orderBy("seq", "desc")
      .limit(1),
  );
  const latestSeq = latest?.seq ?? -1;
  if (
    !state ||
    state.generation !== params.expectedGeneration ||
    state.needsRebuild ||
    state.indexedSeq !== latestSeq
  ) {
    return { generation: state?.generation ?? null, kind: "reset" };
  }
  const tail = params.fromOrdinal === "tail";
  const fromOrdinal =
    params.fromOrdinal === "tail" || !Number.isFinite(params.fromOrdinal)
      ? 0
      : Math.max(0, Math.floor(params.fromOrdinal));
  const limit = normalizeDisplayPageLimit(params.limit);
  const kysely = getDisplayKysely(db);
  const pageQuery = kysely
    .selectFrom(SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE)
    .select(["display_ordinal", "kind", "revision", "row_id", "row_version", "source_event_seq"])
    .where("session_id", "=", sessionId);
  const selected = executeSqliteQuerySync(
    db,
    (tail
      ? pageQuery.orderBy("display_ordinal", "desc")
      : pageQuery.where("display_ordinal", ">=", fromOrdinal).orderBy("display_ordinal", "asc")
    ).limit(limit + 1),
  ).rows;
  const hasMore = selected.length > limit;
  const page = selected.slice(0, limit);
  const rows = (tail ? page.toReversed() : page).map((row) => ({
    displayOrdinal: row.display_ordinal,
    kind: parseDisplayRowKind(row.kind),
    revision: row.revision,
    rowId: row.row_id,
    rowVersion: row.row_version,
    sourceEventSeq: row.source_event_seq,
  }));
  return {
    generation: state.generation,
    kind: "ready",
    ...(!tail && hasMore ? { nextOrdinal: fromOrdinal + rows.length } : {}),
    rows,
  };
}

/** Reads one generation-bound page or returns reset without exposing partial projection state. */
export function readSessionTranscriptDisplayRowsInTransaction(
  db: DatabaseSync,
  sessionId: string,
  params: SessionTranscriptDisplayReadParams,
): SessionTranscriptDisplayReadResult {
  ensureOpenClawAgentDisplayRowSchema(db);
  if (db.isTransaction) {
    return readSessionTranscriptDisplayRowsSnapshot(db, sessionId, params);
  }
  return runSqliteDeferredTransactionSync(
    db,
    () => readSessionTranscriptDisplayRowsSnapshot(db, sessionId, params),
    {
      databaseLabel: "agent transcript display projection",
      operationLabel: "sessions.transcript-display.read",
    },
  );
}
