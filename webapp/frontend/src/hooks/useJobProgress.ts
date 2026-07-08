import { useEffect, useState } from "react";
import { jobStreamUrl } from "@/api/client";
import type { JobStatusResponse } from "@/api/types";

export interface JobProgressState {
  status: "idle" | "queued" | "processing" | "done" | "failed";
  stage: string | null;
  stageIndex: number | null;
  totalStages: number | null;
  errorMessage: string | null;
}

const IDLE_STATE: JobProgressState = {
  status: "idle",
  stage: null,
  stageIndex: null,
  totalStages: null,
  errorMessage: null,
};

function toProgressState(payload: JobStatusResponse): JobProgressState {
  return {
    status: payload.status,
    stage: payload.stage,
    stageIndex: payload.stage_index,
    totalStages: payload.total_stages,
    errorMessage: payload.error_message,
  };
}

/**
 * Subscribes to a job's SSE progress stream (GET /api/jobs/{id}/stream).
 *
 * This is deliberately NOT a TanStack Query hook: SSE is a push-based,
 * long-lived connection rather than a request/response the Query cache can
 * model well (there's no "refetch", the server pushes updates on its own
 * schedule, and we need explicit connection lifecycle management tied to
 * jobId identity). A plain useEffect + EventSource + useState is the more
 * honest fit here.
 */
export function useJobProgress(jobId: string | null): JobProgressState {
  const [state, setState] = useState<JobProgressState>(IDLE_STATE);

  useEffect(() => {
    if (!jobId) {
      setState(IDLE_STATE);
      return;
    }

    // Reset to a "queued" placeholder immediately so the UI doesn't show
    // stale state from a previous jobId while the new connection opens.
    setState({ ...IDLE_STATE, status: "queued" });

    const eventSource = new EventSource(jobStreamUrl(jobId));

    const handleStage = (event: MessageEvent<string>) => {
      try {
        const payload = JSON.parse(event.data) as JobStatusResponse;
        setState(toProgressState(payload));
      } catch {
        // Malformed SSE payload -- ignore this frame rather than crashing
        // the whole progress UI; the next "stage"/"done" event will recover.
      }
    };

    const handleTerminal = (event: MessageEvent<string>) => {
      handleStage(event);
      eventSource.close();
    };

    eventSource.addEventListener("stage", handleStage);
    eventSource.addEventListener("done", handleTerminal);
    eventSource.addEventListener("error", handleTerminal);

    // Also handle the EventSource's own connection-level error (e.g. server
    // unreachable), distinct from a named "error" SSE event above.
    eventSource.onerror = () => {
      setState((prev) =>
        prev.status === "done" || prev.status === "failed"
          ? prev
          : {
              ...prev,
              status: "failed",
              errorMessage: prev.errorMessage ?? "Lost connection to the job progress stream.",
            },
      );
    };

    return () => {
      eventSource.close();
    };
  }, [jobId]);

  return state;
}
