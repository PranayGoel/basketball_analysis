import { useEffect, useRef } from "react";
import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { useJobProgress } from "@/hooks/useJobProgress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface JobProgressCardProps {
  jobId: string;
  onComplete?: () => void;
}

/** Live progress UI driven by the useJobProgress SSE hook. */
export function JobProgressCard({ jobId, onComplete }: JobProgressCardProps) {
  const progress = useJobProgress(jobId);

  const percent =
    progress.stageIndex !== null && progress.totalStages
      ? Math.min(100, Math.round(((progress.stageIndex + 1) / progress.totalStages) * 100))
      : progress.status === "done"
        ? 100
        : 0;

  // Fire onComplete exactly once when the job finishes successfully. This
  // must run in an effect, not the render body: calling a parent's
  // state-setting callback directly during render risks an infinite
  // render loop (parent setState -> re-render -> this renders again ->
  // calls onComplete again). The ref guard additionally protects against
  // StrictMode's intentional double-invoke of effects in development.
  const hasFiredCompleteRef = useRef(false);
  useEffect(() => {
    if (progress.status === "done" && !hasFiredCompleteRef.current) {
      hasFiredCompleteRef.current = true;
      onComplete?.();
    }
  }, [progress.status, onComplete]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-base">
          <span>Analysis progress</span>
          <StatusBadge status={progress.status} />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={
              progress.status === "failed"
                ? "h-full rounded-full bg-destructive transition-all duration-500"
                : "h-full rounded-full bg-primary transition-all duration-500"
            }
            style={{ width: `${percent}%` }}
          />
        </div>

        <div className="flex items-center gap-2 text-sm">
          {progress.status === "failed" ? (
            <AlertTriangle className="h-4 w-4 shrink-0 text-destructive" />
          ) : progress.status === "done" ? (
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
          ) : (
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
          )}
          <span className="text-muted-foreground">
            {describeStage(progress)}
          </span>
        </div>

        {progress.status === "failed" && progress.errorMessage && (
          <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {progress.errorMessage}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function describeStage(progress: ReturnType<typeof useJobProgress>): string {
  if (progress.status === "failed") return progress.errorMessage ?? "Analysis failed.";
  if (progress.status === "done") return "Analysis complete.";
  if (progress.status === "idle") return "Waiting to start...";
  if (progress.stage) {
    const position =
      progress.stageIndex !== null && progress.totalStages
        ? ` (${progress.stageIndex + 1}/${progress.totalStages})`
        : "";
    return `${progress.stage}${position}`;
  }
  return progress.status === "queued" ? "Queued for processing..." : "Processing...";
}

function StatusBadge({ status }: { status: ReturnType<typeof useJobProgress>["status"] }) {
  switch (status) {
    case "done":
      return <Badge variant="success">Done</Badge>;
    case "failed":
      return <Badge variant="destructive">Failed</Badge>;
    case "processing":
      return <Badge variant="default">Processing</Badge>;
    case "queued":
      return <Badge variant="secondary">Queued</Badge>;
    default:
      return <Badge variant="outline">Idle</Badge>;
  }
}
