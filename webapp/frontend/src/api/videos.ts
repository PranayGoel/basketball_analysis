import { apiClient } from "./client";
import type {
  UploadVideoResponse,
  VideoDetail,
  VideoListFilters,
  VideoListResponse,
} from "./types";

export function uploadVideo(file: File): Promise<UploadVideoResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiClient.postForm<UploadVideoResponse>("/videos", formData);
}

function buildQueryString(filters: VideoListFilters): string {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.min_possession_split !== undefined) {
    params.set("min_possession_split", String(filters.min_possession_split));
  }
  if (filters.sort_by) params.set("sort_by", filters.sort_by);
  if (filters.sort_order) params.set("sort_order", filters.sort_order);
  if (filters.page !== undefined) params.set("page", String(filters.page));
  if (filters.page_size !== undefined) {
    params.set("page_size", String(filters.page_size));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

export function fetchVideos(
  filters: VideoListFilters = {},
): Promise<VideoListResponse> {
  return apiClient.get<VideoListResponse>(`/videos${buildQueryString(filters)}`);
}

export function fetchVideo(id: string): Promise<VideoDetail> {
  return apiClient.get<VideoDetail>(`/videos/${id}`);
}

export function deleteVideo(id: string): Promise<void> {
  return apiClient.delete<void>(`/videos/${id}`);
}
