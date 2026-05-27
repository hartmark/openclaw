import fs from "node:fs/promises";
import path from "node:path";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { resolveSessionTranscriptsDirForAgent } from "../../../../src/config/sessions/paths.js";
import { MemoryIndexManager, closeMemoryIndexManagersForAgent } from "./manager.js";
import "./test-runtime-mocks.js";

describe("MemoryManager Batch Repro 2", () => {
  const workspaceDir = path.join(process.cwd(), "test-workspace-repro-2");
  const memoryDir = path.join(workspaceDir, "memory");
  const dbPath = path.join(workspaceDir, "index.sqlite");

  let batchEmbedCalls = 0;
  let batchEmbedChunkCounts: number[] = [];

  const mockProvider = {
    id: "openai",
    model: "mistral/mistral-embed",
    embedBatch: vi.fn(),
    batchEmbed: async (params: any) => {
      batchEmbedCalls++;
      batchEmbedChunkCounts.push(params.chunks.length);
      // Return dummy embeddings
      return new Array(params.chunks.length).fill(0).map(() => new Array(1536).fill(0));
    },
    close: vi.fn(),
  };

  beforeEach(async () => {
    await fs.rm(workspaceDir, { recursive: true, force: true });
    await fs.mkdir(memoryDir, { recursive: true });
    batchEmbedCalls = 0;
    batchEmbedChunkCounts = [];
  });

  afterEach(async () => {
    await closeMemoryIndexManagersForAgent("main");
  });

  it("batches exactly ONE time for many files", async () => {
    const fileCount = 50;
    // Create MEMORY.md
    await fs.writeFile(path.join(workspaceDir, "MEMORY.md"), "# MEMORY.md\nContent");

    // Create many memory files
    for (let i = 0; i < fileCount; i++) {
      await fs.writeFile(path.join(memoryDir, `file-${i}.md`), `# File ${i}\nContent ${i}`);
    }

    const settings = {
      store: {
        path: dbPath,
        vector: { enabled: true },
        fts: { tokenizer: "porter" },
        indexingConcurrency: 16,
      },
      sources: ["memory"],
      cache: { enabled: false, maxEntries: 1000 },
      multimodal: { enabled: false, modalities: [], maxFileBytes: 1024 * 1024 },
      chunking: { tokens: 500, overlap: 50 },
      query: { hybrid: { enabled: true } },
      sync: { watch: false, debounceMs: 0, onSessionStart: false, onDelta: false },
      remote: {
        provider: "openai",
        model: "mistral/mistral-embed",
        apiKey: "sk-mock",
        batch: { enabled: true, wait: true, concurrency: 1 },
      },
    };

    const dummyCfg = {
      agents: {
        defaults: { memorySearch: settings },
        list: [{ id: "main", memorySearch: settings }],
      },
      plugins: { entries: {} },
      provider: { openai: { baseUrl: "https://api.openai.com/v1" } },
    };

    const manager = new MemoryIndexManager({
      cacheKey: "test-2",
      cfg: dummyCfg as any,
      agentId: "main",
      workspaceDir,
      settings: settings as any,
      purpose: "cli",
    });

    // Inject mocks
    (manager as any).provider = mockProvider;
    (manager as any).providerRuntime = mockProvider;
    (manager as any).providerInitialized = true;
    (manager as any).batch = (manager as any).resolveBatchConfig();

    await manager.sync({ reason: "test", force: true });

    // EXPECTATION: Exactly 1 call to batchEmbed
    expect(batchEmbedCalls).toBe(1);
    expect(batchEmbedChunkCounts[0]).toBe(fileCount + 1); // files + MEMORY.md
  });
});
