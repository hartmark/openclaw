import type { DatabaseSync } from "node:sqlite";
import { assertSqliteSchemaContains } from "../infra/sqlite-schema-contract.js";
import { runSqliteImmediateTransactionSync } from "../infra/sqlite-transaction.js";
import { OPENCLAW_AGENT_SCHEMA_SQL } from "./openclaw-agent-schema.js";

export const SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE = "session_transcript_display_state";
export const SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE = "session_transcript_display_rows";

const DISPLAY_ROW_SCHEMA_START = `CREATE TABLE IF NOT EXISTS ${SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE} (`;
const DISPLAY_ROW_SCHEMA_END =
  "CREATE VIRTUAL TABLE IF NOT EXISTS session_transcript_fts USING fts5(";
const SQLITE_TABLE_EXISTS_SQL = "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?";
const ENSURED_DATABASES = new WeakSet<DatabaseSync>();
const ABSENT_DATABASES = new WeakSet<DatabaseSync>();

function splitDisplayRowSchema(sql: string): {
  displayRows: string;
  withoutDisplayRows: string;
} {
  const start = sql.indexOf(DISPLAY_ROW_SCHEMA_START);
  const end = sql.indexOf(DISPLAY_ROW_SCHEMA_END, start);
  if (start === -1 || end === -1) {
    throw new Error("OpenClaw agent display-row schema markers are missing.");
  }
  return {
    displayRows: sql.slice(start, end),
    withoutDisplayRows: `${sql.slice(0, start)}${sql.slice(end)}`,
  };
}

const displayRowSchema = splitDisplayRowSchema(OPENCLAW_AGENT_SCHEMA_SQL);

const AGENT_DISPLAY_ROW_SCHEMA_SQL = displayRowSchema.displayRows;
export const AGENT_BASE_SCHEMA_SQL = displayRowSchema.withoutDisplayRows;

function hasDisplayRowTable(db: DatabaseSync, tableName: string): boolean {
  return Boolean(
    // Schema ownership must reject an incomplete lazy group before installing it.
    db.prepare(/* sqlite-allow-raw */ SQLITE_TABLE_EXISTS_SQL).get(tableName),
  );
}

export function validateOpenClawAgentDisplayRowSchema(db: DatabaseSync): boolean {
  if (ENSURED_DATABASES.has(db)) {
    return true;
  }
  if (ABSENT_DATABASES.has(db)) {
    return false;
  }
  const statePresent = hasDisplayRowTable(db, SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE);
  const rowsPresent = hasDisplayRowTable(db, SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE);
  if (!statePresent && !rowsPresent) {
    ABSENT_DATABASES.add(db);
    return false;
  }
  if (!statePresent || !rowsPresent) {
    throw new Error("OpenClaw agent display-row schema is partially present.");
  }
  assertSqliteSchemaContains(db, "OpenClaw agent display-row schema", AGENT_DISPLAY_ROW_SCHEMA_SQL);
  ENSURED_DATABASES.add(db);
  return true;
}

function cacheDisplayRowSchemaAfterTransaction(db: DatabaseSync): void {
  setImmediate(() => {
    if (!db.isOpen || db.isTransaction) {
      return;
    }
    try {
      if (validateOpenClawAgentDisplayRowSchema(db)) {
        ENSURED_DATABASES.add(db);
      }
    } catch {
      // The next feature use must surface external drift synchronously.
    }
  });
}

/** Lazily installs the complete additive display-row group on first projection use. */
export function ensureOpenClawAgentDisplayRowSchema(db: DatabaseSync): void {
  if (ENSURED_DATABASES.has(db)) {
    return;
  }
  const ensure = () => {
    if (!validateOpenClawAgentDisplayRowSchema(db)) {
      db.exec(AGENT_DISPLAY_ROW_SCHEMA_SQL); // sqlite-allow-raw -- Canonical additive DDL only.
      ABSENT_DATABASES.delete(db);
      assertSqliteSchemaContains(
        db,
        "OpenClaw agent display-row schema",
        AGENT_DISPLAY_ROW_SCHEMA_SQL,
      );
    }
  };
  if (db.isTransaction) {
    ensure();
    cacheDisplayRowSchemaAfterTransaction(db);
    return;
  }
  runSqliteImmediateTransactionSync(db, ensure);
  ENSURED_DATABASES.add(db);
}
