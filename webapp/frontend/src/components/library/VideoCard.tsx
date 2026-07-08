import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Clock, Users, Zap } from "lucide-react";
import type { VideoSummary } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn, formatDateTime, formatDuration, formatPercent } from "@/lib/utils";

export interface VideoCardProps {
  video: VideoSummary;
  /** True when this card matched the current NL library search. */
  highlighted?: boolean;
}

export function VideoCard({ video, highlighted = false }: VideoCardProps) {
  return (
    <Link to={`/videos/${video.id}`} className="block h-full">
      <Card
        className={cn(
          "h-full transition-all hover:border-primary/50 hover:shadow-md",
          highlighted && "ring-2 ring-primary ring-offset-2 ring-offset-background",
        )}
      >
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="truncate text-sm font-medium" title={video.filename}>
              {video.filename}
            </CardTitle>
            <StatusBadge status={video.status} />
          </div>
          <p className="text-xs text-muted-foreground">{formatDateTime(video.uploaded_at)}</p>
        </CardHeader>
        <CardContent className="space-y-3">
          {video.status === "done" ? (
            <>
              <PossessionSummary
                teamA={video.team_a_possession_pct}
                teamB={video.team_b_possession_pct}
              />
              <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                <Stat icon={<Clock className="h-3.5 w-3.5" />} label={formatDuration(video.duration_sec ?? 0)} />
                <Stat icon={<Users className="h-3.5 w-3.5" />} label={`${video.player_count ?? "-"} players`} />
                <Stat
                  icon={<Zap className="h-3.5 w-3.5" />}
                  label={video.max_player_speed_kmh !== null ? `${video.max_player_speed_kmh.toFixed(1)} km/h` : "-"}
                />
              </div>
              {video.has_violations && (
                <div className="flex items-center gap-1.5 text-xs font-medium text-violation">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  Rule violations detected
                </div>
              )}
            </>
          ) : video.status === "failed" ? (
            <p className="text-xs text-destructive">Analysis failed. Open for details.</p>
          ) : (
            <p className="text-xs text-muted-foreground">Analysis {video.status}&hellip;</p>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}

function StatusBadge({ status }: { status: VideoSummary["status"] }) {
  switch (status) {
    case "done":
      return <Badge variant="success">Done</Badge>;
    case "failed":
      return <Badge variant="destructive">Failed</Badge>;
    case "processing":
      return <Badge>Processing</Badge>;
    default:
      return <Badge variant="secondary">Queued</Badge>;
  }
}

function Stat({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <span className="flex items-center gap-1">
      {icon}
      {label}
    </span>
  );
}

function PossessionSummary({
  teamA,
  teamB,
}: {
  teamA: number | null;
  teamB: number | null;
}) {
  if (teamA === null || teamB === null) {
    return <p className="text-xs text-muted-foreground">Possession data unavailable</p>;
  }
  return (
    <div className="space-y-1">
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-team-a" style={{ width: `${teamA}%` }} />
        <div className="h-full bg-team-b" style={{ width: `${teamB}%` }} />
      </div>
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>Team A {formatPercent(teamA)}</span>
        <span>Team B {formatPercent(teamB)}</span>
      </div>
    </div>
  );
}
