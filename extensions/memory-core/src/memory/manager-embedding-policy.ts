import { formatErrorMessage } from "openclaw/plugin-sdk/error-runtime";
import { type EmbeddingInput } from "openclaw/plugin-sdk/memory-core-host-engine-embeddings";

type MemoryEmbeddingChunk = {
  text: string;
  embeddingInput?: EmbeddingInput;
};

export function filterNonEmptyMemoryChunks<T extends MemoryEmbeddingChunk>(chunks: T[]): T[] {
  return chunks.filter((chunk) => chunk.text.trim().length > 0);
}

export function buildMemoryEmbeddingBatches<T extends MemoryEmbeddingChunk>(
  chunks: T[],
  maxChunks: number,
): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < chunks.length; i += maxChunks) {
    batches.push(chunks.slice(i, i + maxChunks));
  }
  return batches;
}

export function isRetryableMemoryEmbeddingError(message: string): boolean {
  return /(rate[_ ]limit|too many requests|429|resource has been exhausted|5\d\d|cloudflare|tokens per day|fetch failed|other side closed|ECONNRESET|ECONNREFUSED|ETIMEDOUT|EPIPE|UND_ERR_|socket hang up|network error|read ECONN|timed out)/i.test(
    message,
  );
}

export function isStructuredInputTooLargeMemoryEmbeddingError(message: string): boolean {
  return /(413|payload too large|request too large|input too large|too many tokens|input limit|request size)/i.test(
    message,
  );
}

export function resolveMemoryEmbeddingRetryDelay(
  delayMs: number,
  randomValue: number,
  maxDelayMs: number,
): number {
  return Math.min(maxDelayMs, Math.round(delayMs * (1 + randomValue * 0.2)));
}

export async function runMemoryEmbeddingRetryLoop<T>(params: {
  run: () => Promise<T>;
  isRetryable: (message: string) => boolean;
  waitForRetry: (delayMs: number) => Promise<void>;
  maxAttempts: number;
  baseDelayMs: number;
}): Promise<T> {
  let attempt = 0;
  let delayMs = params.baseDelayMs;
  while (true) {
    try {
      return await params.run();
    } catch (err) {
      const message = formatErrorMessage(err);
      if (!params.isRetryable(message) || attempt >= params.maxAttempts) {
        throw err;
      }
      await params.waitForRetry(delayMs);
      delayMs *= 2;
      attempt += 1;
    }
  }
}

export function buildTextEmbeddingInputs(chunks: MemoryEmbeddingChunk[]): EmbeddingInput[] {
  return chunks.map((chunk) => chunk.embeddingInput ?? { text: chunk.text });
}
