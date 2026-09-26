import type { AnalyzeOptions, AnalyzeResponse, ModelStatus } from "./types";

/**
 * API client. In production the frontend is served by the FastAPI app itself
 * (same origin). In dev (Vite on :5173) it talks to the local backend on :8000
 * via the proxy configured in vite.config.ts.
 */
const BASE_URL =
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body - keep the generic message */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  return handle(await fetch(`${BASE_URL}/api/health`));
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  return handle(await fetch(`${BASE_URL}/api/model/status`));
}

export async function analyzeImage(
  file: File,
  options: AnalyzeOptions,
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("pixels_per_micrometer", String(options.pixelsPerMicrometer));
  if (options.minParticleAreaPx != null) {
    form.append("min_particle_area_px", String(options.minParticleAreaPx));
  }
  if (options.thresholdInvert != null) {
    form.append("threshold_invert", String(options.thresholdInvert));
  }
  return handle(
    await fetch(`${BASE_URL}/api/analyze`, { method: "POST", body: form }),
  );
}
