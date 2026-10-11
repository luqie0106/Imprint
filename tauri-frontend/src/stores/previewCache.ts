/**
 * A small in-memory cache shared by the Enhance and Ricoh preview pages.
 * Object URLs remain owned by each page; this store only retains response
 * blobs and headers so one page cannot revoke another page's preview URL.
 */

export const PREVIEW_CACHE_LIMIT_BYTES = 64 * 1024 * 1024;
export const PREVIEW_CACHE_LIMIT_ENTRIES = 32;

export interface CachedPreviewResponse {
  blob: Blob;
  headers: Array<[string, string]>;
}

interface CacheEntry extends CachedPreviewResponse {
  byteLength: number;
}

const entries = new Map<string, CacheEntry>();
let totalBytes = 0;

function normalizeForJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(item => normalizeForJson(item));
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(Object.keys(record)
      .filter(key => record[key] !== undefined)
      .sort()
      .map(key => [key, normalizeForJson(record[key])]));
  }
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

function stableSerialize(value: unknown): string {
  return JSON.stringify(normalizeForJson(value)) ?? "null";
}

/**
 * Request generations and client IDs identify cancellation ownership only;
 * all remaining request fields participate in the preview identity.
 */
export function previewCacheKey(baseUrl: string, body: Record<string, unknown>): string {
  const {
    preview_client_id: _clientId,
    preview_generation: _generation,
    ...renderRequest
  } = body;
  return stableSerialize({ baseUrl, renderRequest });
}

export function getCachedPreview(
  baseUrl: string,
  body: Record<string, unknown>,
): CachedPreviewResponse | undefined {
  const key = previewCacheKey(baseUrl, body);
  const entry = entries.get(key);
  if (!entry) return undefined;

  // Reinsertion makes Map iteration order the LRU order.
  entries.delete(key);
  entries.set(key, entry);
  return { blob: entry.blob, headers: entry.headers.map(header => [...header] as [string, string]) };
}

export function hasCachedPreview(baseUrl: string, body: Record<string, unknown>): boolean {
  const key = previewCacheKey(baseUrl, body);
  const entry = entries.get(key);
  if (!entry) return false;
  entries.delete(key);
  entries.set(key, entry);
  return true;
}

export function cachePreviewResponse(
  baseUrl: string,
  body: Record<string, unknown>,
  response: Pick<Response, "ok" | "status" | "headers">,
  blob: Blob,
): boolean {
  if (!response.ok || response.status === 204) return false;

  const key = previewCacheKey(baseUrl, body);
  const existing = entries.get(key);
  if (existing) {
    totalBytes -= existing.byteLength;
    entries.delete(key);
  }

  const byteLength = blob.size;
  if (byteLength > PREVIEW_CACHE_LIMIT_BYTES || PREVIEW_CACHE_LIMIT_BYTES <= 0) return false;

  const headers: Array<[string, string]> = [];
  response.headers.forEach((value, name) => headers.push([name, value]));
  entries.set(key, { blob, headers, byteLength });
  totalBytes += byteLength;

  while ((totalBytes > PREVIEW_CACHE_LIMIT_BYTES || entries.size > PREVIEW_CACHE_LIMIT_ENTRIES) && entries.size > 0) {
    const oldestKey = entries.keys().next().value as string | undefined;
    if (oldestKey === undefined) break;
    const oldest = entries.get(oldestKey);
    entries.delete(oldestKey);
    if (oldest) totalBytes -= oldest.byteLength;
  }
  return entries.has(key);
}

export function clearPreviewCache(): void {
  entries.clear();
  totalBytes = 0;
  decodedImages.clear();
  decodedBytes = 0;
  histogramPaths = new WeakMap();
}

export function previewCacheStats(): { entries: number; bytes: number } {
  return { entries: entries.size, bytes: totalBytes };
}

// Keep decoded images bounded separately from compressed JPEG responses.
const previewBlobs = new Map<string, Blob>();
const decodedImages = new Map<Blob, HTMLImageElement>();
const decodedLimitBytes = 256 * 1024 * 1024;
let decodedBytes = 0;

export function createPreviewUrl(blob: Blob): string {
  const url = URL.createObjectURL(blob);
  previewBlobs.set(url, blob);
  return url;
}

export function releasePreviewUrl(url: string): void {
  previewBlobs.delete(url);
  if (url) URL.revokeObjectURL(url);
}

export function getDecodedPreview(url: string): HTMLImageElement | undefined {
  const blob = previewBlobs.get(url);
  const image = blob ? decodedImages.get(blob) : undefined;
  if (blob && image) {
    decodedImages.delete(blob);
    decodedImages.set(blob, image);
  }
  return image;
}

export function rememberDecodedPreview(url: string, image: HTMLImageElement): void {
  const blob = previewBlobs.get(url);
  const bytes = image.naturalWidth * image.naturalHeight * 4;
  if (!blob || !bytes || bytes > decodedLimitBytes) return;
  const previous = decodedImages.get(blob);
  if (previous) decodedBytes -= previous.naturalWidth * previous.naturalHeight * 4;
  decodedImages.delete(blob);
  decodedImages.set(blob, image);
  decodedBytes += bytes;
  while (decodedBytes > decodedLimitBytes || decodedImages.size > 4) {
    const oldest = decodedImages.keys().next().value;
    if (!oldest) break;
    const oldImage = decodedImages.get(oldest)!;
    decodedBytes -= oldImage.naturalWidth * oldImage.naturalHeight * 4;
    decodedImages.delete(oldest);
  }
}

// Histograms follow the cached response Blob, even when pages create new URLs.
// Weak keys release these small arrays when their preview response is evicted.
let histogramPaths = new WeakMap<Blob, string[]>();

export function getPreviewHistogram(url: string): string[] | undefined {
  const blob = previewBlobs.get(url);
  return blob ? histogramPaths.get(blob) : undefined;
}

export function rememberPreviewHistogram(url: string, paths: string[]): void {
  const blob = previewBlobs.get(url);
  if (blob) histogramPaths.set(blob, paths);
}
