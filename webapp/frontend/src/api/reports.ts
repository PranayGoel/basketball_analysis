import { apiClient } from "./client";
import type {
  GameReport,
  NarrativeResponse,
  QaResponse,
  TimelineEventsResponse,
} from "./types";

export function fetchReport(videoId: string): Promise<GameReport> {
  return apiClient.get<GameReport>(`/videos/${videoId}/report`);
}

export function fetchNarrative(videoId: string): Promise<NarrativeResponse> {
  return apiClient.get<NarrativeResponse>(`/videos/${videoId}/narrative`);
}

export function askQuestion(
  videoId: string,
  question: string,
): Promise<QaResponse> {
  return apiClient.post<QaResponse>(`/videos/${videoId}/qa`, { question });
}

export function fetchEvents(
  videoId: string,
): Promise<TimelineEventsResponse> {
  return apiClient.get<TimelineEventsResponse>(`/videos/${videoId}/events`);
}
