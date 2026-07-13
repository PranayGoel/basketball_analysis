/**
 * Thin fetch wrapper shared by every api/* module.
 *
 * In dev, VITE_API_URL is unset and the Vite proxy (vite.config.ts) forwards
 * /api/* to localhost:8000. In production, set VITE_API_URL to the backend
 * origin (e.g. https://api.courtvision.app) so the frontend can be deployed
 * separately (e.g. Vercel) from the backend (e.g. Railway / Render / Fly.io).
 */
// Strip any trailing slash from VITE_API_URL so setting it to e.g.
// "https://api.courtvision.app/" doesn't produce double slashes, and
// setting it to ".../api" doesn't produce a doubled "/api/api" prefix.
const API_BASE = `${(import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "")}/api`;

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function parseErrorBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  try {
    if (contentType.includes("application/json")) {
      return await response.json();
    }
    return await response.text();
  } catch {
    return null;
  }
}

function extractErrorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const detail = (body as Record<string, unknown>).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{ loc, msg, type }, ...]
      const first = detail[0] as Record<string, unknown> | undefined;
      if (first && typeof first.msg === "string") return first.msg;
    }
  }
  if (typeof body === "string" && body.trim().length > 0) return body;
  return `Request failed with status ${status}`;
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const body = await parseErrorBody(response);
    throw new ApiError(response.status, extractErrorMessage(response.status, body), body);
  }

  // 204 No Content (DELETE) has no body to parse.
  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export const apiClient = {
  get<T>(path: string): Promise<T> {
    return request<T>(path, { method: "GET" });
  },

  post<T>(path: string, body?: unknown): Promise<T> {
    return request<T>(path, {
      method: "POST",
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  },

  postForm<T>(path: string, formData: FormData): Promise<T> {
    // Do NOT set Content-Type manually here -- the browser needs to set it
    // (including the multipart boundary) itself for FormData bodies.
    return request<T>(path, { method: "POST", body: formData });
  },

  delete<T = void>(path: string): Promise<T> {
    return request<T>(path, { method: "DELETE" });
  },
};

/** Builds the direct (non-fetched) URL for the video stream endpoint, for use as a <video src>. */
export function videoStreamUrl(videoId: string): string {
  return `${API_BASE}/videos/${videoId}/stream`;
}

/** Builds the SSE endpoint URL for a job's progress stream. */
export function jobStreamUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/stream`;
}
