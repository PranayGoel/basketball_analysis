/**
 * TypeScript types mirroring the backend API contract exactly
 * (webapp/backend, base path /api). These are hand-kept in sync with the
 * FastAPI backend's Pydantic models -- there is no shared schema package
 * between the two, so any backend contract change must be mirrored here
 * manually.
 */

export type VideoStatus = "queued" | "processing" | "done" | "failed";

export interface VideoSummary {
  id: string;
  filename: string;
  uploaded_at: string; // ISO datetime
  duration_sec: number | null;
  status: VideoStatus;
  team_a_possession_pct: number | null;
  team_b_possession_pct: number | null;
  player_count: number | null;
  total_passes: number | null;
  total_interceptions: number | null;
  max_player_speed_kmh: number | null;
  has_violations: boolean;
}

export interface VideoDetail extends VideoSummary {
  error_message: string | null;
}

export interface VideoListResponse {
  items: VideoSummary[];
  total: number;
  page: number;
  page_size: number;
}

export type VideoSortBy =
  | "uploaded_at"
  | "total_passes"
  | "max_player_speed_kmh";
export type SortOrder = "asc" | "desc";

export interface VideoListFilters {
  status?: VideoStatus;
  min_possession_split?: number;
  sort_by?: VideoSortBy;
  sort_order?: SortOrder;
  page?: number;
  page_size?: number;
}

export interface UploadVideoResponse {
  video_id: string;
  job_id: string;
  status: "queued";
}

export type JobStatus = "queued" | "processing" | "done" | "failed";

export interface JobStatusResponse {
  job_id: string;
  video_id: string;
  status: JobStatus;
  stage: string | null;
  stage_index: number | null;
  total_stages: number | null;
  error_message: string | null;
}

/**
 * Per-player stats as emitted by the CV pipeline's own game_report.py,
 * passed through verbatim by GET /api/videos/{id}/report.
 */
export interface PlayerStats {
  label: string;
  team: number | null;
  total_distance_m: number;
  avg_speed_kmh: number;
  max_speed_kmh: number;
}

export type ViolationType = "double_dribble" | "traveling";

export interface ViolationRecord {
  violation_type: ViolationType;
  player_id: number;
  start_frame: number;
  end_frame: number;
  confidence: "heuristic";
}

export interface GameReport {
  players: Record<string, PlayerStats>;
  team_possession: {
    team_1_pct: number;
    team_2_pct: number;
    undecided_pct: number;
  };
  events: {
    passes: { team_1: number; team_2: number };
    interceptions: { team_1: number; team_2: number };
  };
  num_frames: number;
  violations?: ViolationRecord[];
}

export interface NarrativeResponse {
  narrative: string;
  cached: boolean;
}

export interface QaResponse {
  answer: string;
}

export interface SearchResponse {
  answer: string;
  matched_video_ids: string[];
}

/**
 * NOTE: the /events endpoint currently only ever returns "violation" events,
 * because the CV pipeline's report only carries aggregate pass/interception
 * counts, not a timestamped per-event list. event_type is still typed as a
 * string union (not a hardcoded literal) so this type doesn't need to change
 * the day the backend starts emitting "pass" | "interception" events too.
 */
export type TimelineEventType = "violation" | (string & {});

export interface TimelineEventDetail {
  violation_type: string;
  end_frame: number;
  confidence: string;
}

export interface TimelineEvent {
  event_type: TimelineEventType;
  frame_num: number;
  timestamp_sec: number;
  player_id: number;
  detail: TimelineEventDetail;
}

export interface TimelineEventsResponse {
  events: TimelineEvent[];
}
