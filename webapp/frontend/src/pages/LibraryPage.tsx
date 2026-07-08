import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Inbox, UploadCloud } from "lucide-react";
import { useVideoLibrary } from "@/hooks/useVideoLibrary";
import { LibraryFilters, type LibraryFilterState } from "@/components/library/LibraryFilters";
import { LibrarySearchBar } from "@/components/library/LibrarySearchBar";
import { VideoCard } from "@/components/library/VideoCard";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import type { VideoListFilters } from "@/api/types";

const DEFAULT_FILTERS: LibraryFilterState = {
  status: "all",
  sortBy: "uploaded_at",
  sortOrder: "desc",
};

export default function LibraryPage() {
  const [filters, setFilters] = useState<LibraryFilterState>(DEFAULT_FILTERS);
  const [matchedVideoIds, setMatchedVideoIds] = useState<string[] | null>(null);

  const queryFilters: VideoListFilters = useMemo(
    () => ({
      status: filters.status === "all" ? undefined : filters.status,
      sort_by: filters.sortBy,
      sort_order: filters.sortOrder,
    }),
    [filters],
  );

  // isPending (not isLoading) deliberately: isLoading in React Query v5 means
  // "isPending AND actively fetching" -- a query stuck in a "paused"
  // fetchStatus (e.g. the browser's/query client's online-manager considers
  // the connection offline, however transiently or spuriously) is isPending
  // but NOT isLoading, so an isLoading-only check leaves the whole library
  // page blank (no skeleton, no error, no content) instead of showing
  // *something*. isPending covers "no data yet" regardless of why.
  const { data, isPending, isError, error } = useVideoLibrary(queryFilters);
  const matchedSet = useMemo(() => new Set(matchedVideoIds ?? []), [matchedVideoIds]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Video library</h1>
          <p className="text-muted-foreground">Every game you&apos;ve analyzed, in one place.</p>
        </div>
        <Link to="/upload">
          <Button>
            <UploadCloud className="h-4 w-4" />
            Upload a video
          </Button>
        </Link>
      </div>

      <LibrarySearchBar onResults={setMatchedVideoIds} />
      <LibraryFilters value={filters} onChange={setFilters} />

      {isPending && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-52 w-full" />
          ))}
        </div>
      )}

      {isError && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          Failed to load videos: {error instanceof Error ? error.message : "Unknown error"}
        </div>
      )}

      {!isPending && !isError && data && data.items.length === 0 && (
        <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border py-16 text-center">
          <Inbox className="h-10 w-10 text-muted-foreground" />
          <div>
            <p className="font-medium">No videos yet</p>
            <p className="text-sm text-muted-foreground">Upload your first game film to get started.</p>
          </div>
          <Link to="/upload">
            <Button variant="secondary" className="mt-2">
              Upload a video
            </Button>
          </Link>
        </div>
      )}

      {!isPending && !isError && data && data.items.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.items.map((video) => (
            <VideoCard key={video.id} video={video} highlighted={matchedSet.has(video.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
