import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { OpenClawConfig } from "openclaw/plugin-sdk/memory-core-host-engine-foundation";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { closeAllMemorySearchManagers, getMemorySearchManager } from "./index.js";
import type { MemoryIndexManager } from "./manager.js";
import "./test-runtime-mocks.js";

const batchEmbedSpy = vi.fn(async ({ chunks }) => chunks.map(() => Array(1536).fill(0.1)));

vi.mock("./embeddings.js", () => ({
  createEmbeddingProvider: async () => ({
    requestedProvider: "openai",
    provider: {
      id: "openai",
      model: "text-embedding-3-small",
      embedQuery: async () => Array(1536).fill(0.1),
      embedBatch: async (texts: string[]) => texts.map(() => Array(1536).fill(0.1)),
    },
    runtime: {
      id: "openai",
      batchEmbed: batchEmbedSpy,
    },
  }),
  resolveEmbeddingProviderFallbackModel: () => "fts-only",
}));

describe("memory manager multi-file batching", () => {
  let fixtureRoot = "";
  let workspaceDir = "";
  let indexPath = "";
  let manager: MemoryIndexManager | null = null;

  beforeAll(async () => {
    fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-mem-batching-"));
  });

  beforeEach(async () => {
    workspaceDir = path.join(fixtureRoot, `test-run`);
    await fs.mkdir(path.join(workspaceDir, "memory"), { recursive: true });
    indexPath = path.join(workspaceDir, "index.sqlite");
    batchEmbedSpy.mockClear();
  });

  afterEach(async () => {
    if (manager) {
      await manager.close();
      manager = null;
    }
    await closeAllMemorySearchManagers();
    await fs.rm(workspaceDir, { recursive: true, force: true });
  });

  afterAll(async () => {
    if (fixtureRoot) {
      await fs.rm(fixtureRoot, { recursive: true, force: true });
    }
  });

  async function createManager(batchEnabled: boolean): Promise<MemoryIndexManager> {
    const cfg = {
      memory: { backend: "builtin" },
      agents: {
        defaults: {
          workspace: workspaceDir,
          memorySearch: {
            provider: "openai",
            model: "text-embedding-3-small",
            store: { path: indexPath },
            remote: {
              batch: { enabled: batchEnabled, wait: true },
            },
            cache: { enabled: false },
            sync: { watch: false, onSessionStart: false, onSearch: false },
          },
        },
        list: [{ id: "main", default: true }],
      },
    } as unknown as OpenClawConfig;
    const result = await getMemorySearchManager({ cfg, agentId: "main" });
    if (!result.manager) {
      throw new Error(result.error ?? "manager missing");
    }
    manager = result.manager as unknown as MemoryIndexManager;
    return manager;
  }

  it("calls batchEmbed once for multiple dirty files (reproduction)", async () => {
    await fs.writeFile(path.join(workspaceDir, "memory", "file1.md"), "content 1");
    await fs.writeFile(path.join(workspaceDir, "memory", "file2.md"), "content 2");
    await fs.writeFile(path.join(workspaceDir, "memory", "file3.md"), "content 3");

    const memoryManager = await createManager(true);
    await memoryManager.sync({ force: true });

    // Current behavior: expected 3, actual will be 3 (test fails if it expects 1)
    // We want it to be 1.
    expect(batchEmbedSpy).toHaveBeenCalledTimes(1);
  });
});
