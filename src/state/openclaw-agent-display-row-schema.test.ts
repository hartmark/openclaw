import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { useAutoCleanupTempDirTracker } from "../../test/helpers/temp-dir.js";
import { assertSqliteSchemaContains } from "../infra/sqlite-schema-contract.js";
import {
  closeOpenClawAgentDatabasesForTest,
  openOpenClawAgentDatabase,
} from "./openclaw-agent-db.js";
import {
  ensureOpenClawAgentDisplayRowSchema,
  AGENT_BASE_SCHEMA_SQL,
  SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE,
  SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE,
} from "./openclaw-agent-display-row-schema.js";
import { OPENCLAW_AGENT_SCHEMA_SQL } from "./openclaw-agent-schema.js";

const tempDirs = useAutoCleanupTempDirTracker(afterEach);

afterEach(() => {
  closeOpenClawAgentDatabasesForTest();
});

function tableExists(database: DatabaseSync, tableName: string): boolean {
  return Boolean(
    database
      .prepare("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?")
      .get(tableName),
  );
}

function createDisplayDatabase(): DatabaseSync {
  const database = new DatabaseSync(":memory:");
  database.exec("PRAGMA foreign_keys = ON;");
  database.exec(OPENCLAW_AGENT_SCHEMA_SQL);
  database
    .prepare(
      `INSERT INTO session_nodes
         (session_key, current_session_id, entry_json, entry_valid, updated_at)
       VALUES ('agent:main:test', 'session-1', '{}', -1, 1)`,
    )
    .run();
  database
    .prepare(
      `INSERT INTO session_windows
         (session_id, session_key, session_scope, created_at, updated_at)
       VALUES ('session-1', 'agent:main:test', 'conversation', 1, 1)`,
    )
    .run();
  database
    .prepare(
      `INSERT INTO transcript_events (session_id, seq, event_json, created_at)
       VALUES ('session-1', 0, '{"type":"session","id":"session-1"}', 1)`,
    )
    .run();
  return database;
}

function insertDisplayStateAndRow(database: DatabaseSync): void {
  database
    .prepare(
      `INSERT INTO session_transcript_display_state
         (session_id, generation, indexed_seq, row_count, needs_rebuild, updated_at)
       VALUES ('session-1', 'generation-1', 0, 1, 0, 1)`,
    )
    .run();
  database
    .prepare(
      `INSERT INTO session_transcript_display_rows
         (session_id, row_id, row_version, revision, display_ordinal, source_event_seq, kind)
       VALUES ('session-1', 'row-1', 1, 1, 0, 0, 'opaque')`,
    )
    .run();
}

describe("agent display-row schema", () => {
  it("stays absent until first use without changing schema version metadata", () => {
    const stateDir = tempDirs.make("openclaw-display-row-schema-");
    const database = openOpenClawAgentDatabase({
      agentId: "main",
      env: { OPENCLAW_STATE_DIR: stateDir },
    });
    const versionBefore = database.db.prepare("PRAGMA user_version").get();
    const metadataBefore = database.db
      .prepare("SELECT schema_version, updated_at FROM schema_meta WHERE meta_key = 'primary'")
      .get();

    expect(tableExists(database.db, SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)).toBe(false);
    expect(tableExists(database.db, SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE)).toBe(false);

    ensureOpenClawAgentDisplayRowSchema(database.db);
    ensureOpenClawAgentDisplayRowSchema(database.db);

    expect(tableExists(database.db, SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)).toBe(true);
    expect(tableExists(database.db, SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE)).toBe(true);
    expect(
      database.db
        .prepare(
          "SELECT name, strict FROM pragma_table_list WHERE name LIKE 'session_transcript_display_%' ORDER BY name",
        )
        .all(),
    ).toEqual([
      { name: SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE, strict: 1 },
      { name: SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE, strict: 1 },
    ]);
    expect(database.db.prepare("PRAGMA user_version").get()).toEqual(versionBefore);
    expect(
      database.db
        .prepare("SELECT schema_version, updated_at FROM schema_meta WHERE meta_key = 'primary'")
        .get(),
    ).toEqual(metadataBefore);
  });

  it("does not cache a schema ensure rolled back by its caller", () => {
    const database = new DatabaseSync(":memory:");
    try {
      database.exec(AGENT_BASE_SCHEMA_SQL);
      database.exec("BEGIN IMMEDIATE;");
      ensureOpenClawAgentDisplayRowSchema(database);
      database.exec("ROLLBACK;");

      expect(tableExists(database, SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)).toBe(false);
      ensureOpenClawAgentDisplayRowSchema(database);
      expect(tableExists(database, SESSION_TRANSCRIPT_DISPLAY_STATE_TABLE)).toBe(true);
    } finally {
      database.close();
    }
  });

  it.each([
    {
      name: "one missing table",
      damage: `DROP TABLE ${SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE};`,
      message: /partially present/u,
    },
    {
      name: "one missing index",
      damage: "DROP INDEX idx_agent_transcript_display_ordinal;",
      message: /idx_agent_transcript_display_ordinal|schema/u,
    },
    {
      name: "a malformed row table",
      damage: `
        DROP TABLE ${SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE};
        CREATE TABLE ${SESSION_TRANSCRIPT_DISPLAY_ROWS_TABLE} (
          session_id TEXT NOT NULL,
          row_id TEXT NOT NULL,
          PRIMARY KEY (session_id, row_id)
        ) STRICT;
      `,
      message: /session_transcript_display_rows|schema/u,
    },
  ])("rejects $name during physical reopen", ({ damage, message }) => {
    const stateDir = tempDirs.make("openclaw-display-row-reopen-");
    const options = { agentId: "main", env: { OPENCLAW_STATE_DIR: stateDir } };
    const database = openOpenClawAgentDatabase(options);
    ensureOpenClawAgentDisplayRowSchema(database.db);
    database.db.exec(damage);
    closeOpenClawAgentDatabasesForTest();

    expect(() => openOpenClawAgentDatabase(options)).toThrow(message);
  });

  it("keeps a complete populated group compatible with the prior schema contract", () => {
    const database = createDisplayDatabase();
    try {
      insertDisplayStateAndRow(database);
      expect(() =>
        assertSqliteSchemaContains(database, "previous agent schema", AGENT_BASE_SCHEMA_SQL),
      ).not.toThrow();
      expect(database.prepare("SELECT row_id FROM session_transcript_display_rows").get()).toEqual({
        row_id: "row-1",
      });
    } finally {
      database.close();
    }
  });

  it.each([
    {
      deleteSql: "DELETE FROM transcript_events WHERE session_id = 'session-1' AND seq = 0",
      expectedStateRows: 1,
      name: "source event",
    },
    {
      deleteSql: "DELETE FROM session_transcript_display_state WHERE session_id = 'session-1'",
      expectedStateRows: 0,
      name: "display state",
    },
    {
      deleteSql: "DELETE FROM session_windows WHERE session_id = 'session-1'",
      expectedStateRows: 0,
      name: "session",
    },
  ])("cascades display rows when deleting the $name owner", ({ deleteSql, expectedStateRows }) => {
    const database = createDisplayDatabase();
    try {
      insertDisplayStateAndRow(database);
      database.exec(deleteSql);

      expect(
        database.prepare("SELECT COUNT(*) AS count FROM session_transcript_display_rows").get(),
      ).toEqual({ count: 0 });
      expect(
        database.prepare("SELECT COUNT(*) AS count FROM session_transcript_display_state").get(),
      ).toEqual({ count: expectedStateRows });
      expect(database.prepare("PRAGMA foreign_key_check").all()).toEqual([]);
    } finally {
      database.close();
    }
  });
});
