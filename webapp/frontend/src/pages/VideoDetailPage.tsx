import { useCallback, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Loader2, Trash2 } from "lucide-react";
import { deleteVideo, fetchVideo } from "@/api/videos";
import { fetchEvents, fetchReport } from "@/api/reports";
import {
  SmartVideoPlayer,
  type SmartVideoPlayerHandle,
} from "@/components/player/SmartVideoPlayer";
import { EventTimeline } from "@/components/player/EventTimeline";
import { GameNarrativeCard } from "@/components/report/GameNarrativeCard";
import { GameQAPanel } from "@/components/chat/GameQAPanel";
import { PlayerStatsTable } from "@/components/report/PlayerStatsTable";
import { PossessionBar } from "@/components/report/PossessionBar";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { formatDuration } from "@/lib/utils";

export default function VideoDetailPage() {
  // useParams can technically return undefined for an unmatched param, but
  // the route is declared as "/videos/:id" so id is always present here in
  // practice. Falling back to "" (rather than an early return before hooks
  // run) keeps every hook below unconditional, which the Rules of Hooks
  // require -- React needs the exact same hooks to run, in the same order,
  // on every render of this component. The `enabled: Boolean(id)` guards on
  // the queries below are what actually prevent a request from firing with
  // an empty id.
  const { id = "" } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const playerRef = useRef<SmartVideoPlayerHandle>(null);

  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [playerDurationSec, setPlayerDurationSec] = useState<number | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  const videoQuery = useQuery({
    queryKey: ["video", id],
    queryFn: () => fetchVideo(id),
    enabled: Boolean(id),
    // Poll while the video is still processing so the UI notices when it
    // finishes even without a job_id on hand (the video summary itself
    // doesn't include one -- only the initial upload response does).
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "processing" ? 3000 : false;
    },
  });

  const isProcessing =
    videoQuery.data?.status === "queued" || videoQuery.data?.status === "processing";

  const reportQuery = useQuery({
    queryKey: ["report", id],
    queryFn: () => fetchReport(id),
    enabled: Boolean(id) && videoQuery.data?.status === "done",
  });

  const eventsQuery = useQuery({
    queryKey: ["events", id],
    queryFn: () => fetchEvents(id),
    enabled: Boolean(id) && videoQuery.data?.status === "done",
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteVideo(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      navigate("/library");
    },
  });

  const handleSeek = useCallback((timeSec: number) => {
    playerRef.current?.seek(timeSec);
  }, []);

  if (!id) {
    return <p className="text-destructive">Missing video id.</p>;
  }

  // isPending (not isLoading) deliberately -- see LibraryPage.tsx's comment:
  // a query "paused" (fetchStatus) rather than actively "fetching" is
  // isPending but not isLoading, and would otherwise fall through to the
  // isError-or-no-data branch below and show a misleading "Failed to load"
  // for a video that hasn't actually failed, just hasn't resolved yet.
  if (videoQuery.isPending) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-1/3" />
        <Skeleton className="aspect-video w-full" />
      </div>
    );
  }

  if (videoQuery.isError || !videoQuery.data) {
    return (
      <Card className="border-destructive/50">
        <CardContent className="py-8 text-center text-destructive">
          Failed to load this video.{" "}
          {videoQuery.error instanceof Error ? videoQuery.error.message : ""}
        </CardContent>
      </Card>
    );
  }

  const video = videoQuery.data;

  // approximate fps derivation: the report/events contract has no explicit
  // fps field, so timestamp_sec on each TimelineEvent is already computed
  // by the backend from an assumed fps (see api/types.ts TimelineEvent
  // doc-comment). Here we independently derive the *player's* duration from
  // the true <video> element metadata (via onDurationChange) rather than
  // trusting the possibly-null video.duration_sec from the summary --
  // this is the more accurate source once the element has loaded, and
  // degrades gracefully to duration_sec/0 while metadata is still loading.
  const durationSec = playerDurationSec ?? video.duration_sec ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{video.filename}</h1>
          <p className="text-sm text-muted-foreground">
            Uploaded {new Date(video.uploaded_at).toLocaleString()}
          </p>
        </div>
        <Button variant="outline" onClick={() => setIsDeleteDialogOpen(true)}>
          <Trash2 className="h-4 w-4" />
          Delete
        </Button>
      </div>

      {video.status === "failed" && (
        <Card className="border-destructive/50">
          <CardContent className="flex items-start gap-3 py-6">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
            <div>
              <p className="font-medium text-destructive">Analysis failed</p>
              <p className="text-sm text-muted-foreground">
                {video.error_message ?? "An unknown error occurred while processing this video."}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {isProcessing && (
        <Card>
          <CardContent className="flex items-center gap-3 py-10">
            <Loader2 className="h-5 w-5 shrink-0 animate-spin text-primary" />
            <div>
              <p className="font-medium">Analysis in progress&hellip;</p>
              <p className="text-sm text-muted-foreground">
                This page will update automatically once processing completes.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {video.status === "done" && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <SmartVideoPlayer
              ref={playerRef}
              videoId={id}
              onTimeUpdate={setCurrentTimeSec}
              onDurationChange={setPlayerDurationSec}
            />
            {durationSec > 0 && (
              <EventTimeline
                durationSec={durationSec}
                currentTimeSec={currentTimeSec}
                events={eventsQuery.data?.events ?? []}
                onSeek={handleSeek}
              />
            )}

            {reportQuery.data && <PlayerStatsTable players={reportQuery.data.players} />}
          </div>

          <div className="space-y-4">
            <GameNarrativeCard videoId={id} />
            <GameQAPanel videoId={id} />
            {reportQuery.data && (
              <PossessionBar
                teamOnePct={reportQuery.data.team_possession.team_1_pct}
                teamTwoPct={reportQuery.data.team_possession.team_2_pct}
                undecidedPct={reportQuery.data.team_possession.undecided_pct}
              />
            )}
          </div>
        </div>
      )}

      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent onClose={() => setIsDeleteDialogOpen(false)}>
          <DialogHeader>
            <DialogTitle>Delete this video?</DialogTitle>
            <DialogDescription>
              This will permanently remove &ldquo;{video.filename}&rdquo; ({formatDuration(durationSec)}) and
              its analysis. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteMutation.mutate()}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
