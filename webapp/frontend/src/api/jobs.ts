import { apiClient } from "./client";
import type { JobStatusResponse } from "./types";

export function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  return apiClient.get<JobStatusResponse>(`/jobs/${jobId}`);
}
