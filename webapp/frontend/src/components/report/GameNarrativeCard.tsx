import { useQuery } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import { fetchNarrative } from "@/api/reports";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

export interface GameNarrativeCardProps {
  videoId: string;
}

/** LLM-generated game recap, fetched on demand from GET /videos/{id}/narrative. */
export function GameNarrativeCard({ videoId }: GameNarrativeCardProps) {
  // isPending (not isLoading) deliberately -- see LibraryPage.tsx's comment
  // on the same fix: isLoading in React Query v5 is isPending && isFetching,
  // so a query stuck in a "paused" fetchStatus renders neither the skeleton
  // nor the error branch below, leaving this card silently blank forever.
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["narrative", videoId],
    queryFn: () => fetchNarrative(videoId),
    // The most common failure here (503: no LLM provider configured on the
    // server) is permanent, not transient -- retrying can't make an env var
    // appear. Retrying it anyway also risks the query getting stuck in a
    // "paused" fetchStatus waiting on an online-transition that may never
    // fire (observed in practice), leaving the card blank indefinitely
    // instead of showing the actionable error message below.
    retry: false,
  });

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          Game recap
        </CardTitle>
        {data?.cached && <Badge variant="secondary">Cached</Badge>}
      </CardHeader>
      <CardContent>
        {isPending && (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        )}
        {isError && (
          <p className="text-sm text-destructive">
            Couldn&apos;t generate a recap: {error instanceof Error ? error.message : "Unknown error"}
          </p>
        )}
        {data && !isPending && !isError && (
          <p className="whitespace-pre-line text-sm leading-relaxed text-foreground/90">
            {data.narrative}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
