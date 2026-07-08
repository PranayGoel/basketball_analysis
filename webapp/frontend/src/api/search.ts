import { apiClient } from "./client";
import type { SearchResponse } from "./types";

export function searchLibrary(query: string): Promise<SearchResponse> {
  return apiClient.post<SearchResponse>("/search", { query });
}
