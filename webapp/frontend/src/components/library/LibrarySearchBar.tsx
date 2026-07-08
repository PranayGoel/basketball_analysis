import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { Search, X } from "lucide-react";
import { searchLibrary } from "@/api/search";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export interface LibrarySearchBarProps {
  /** Called with the matched video IDs whenever a search resolves, and with
   * null when the search is cleared -- lets the parent grid highlight/reset. */
  onResults: (matchedVideoIds: string[] | null) => void;
}

/**
 * Natural-language search input for the library. On submit, calls
 * POST /api/search and surfaces both the free-text `answer` and the
 * `matched_video_ids` (via onResults) so the parent LibraryPage can
 * highlight matching VideoCards in the grid -- the search is meant to
 * visibly narrow the grid the user is already looking at, not just show
 * disconnected text.
 */
export function LibrarySearchBar({ onResults }: LibrarySearchBarProps) {
  const [query, setQuery] = useState("");

  const mutation = useMutation({
    mutationFn: searchLibrary,
    onSuccess: (data) => onResults(data.matched_video_ids),
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    mutation.mutate(trimmed);
  };

  const handleClear = () => {
    setQuery("");
    mutation.reset();
    onResults(null);
  };

  return (
    <div className="space-y-2">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about your library, e.g. &ldquo;games with more than 3 turnovers&rdquo;"
            className="pl-9 pr-9"
            disabled={mutation.isPending}
          />
          {query && (
            <button
              type="button"
              onClick={handleClear}
              aria-label="Clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-sm p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <Button type="submit" disabled={mutation.isPending || !query.trim()}>
          {mutation.isPending ? "Searching..." : "Search"}
        </Button>
      </form>

      {mutation.isError && (
        <p className="text-sm text-destructive">
          Search failed: {mutation.error instanceof Error ? mutation.error.message : "Unknown error"}
        </p>
      )}

      {mutation.isSuccess && (
        <div className="rounded-md border border-border bg-accent/40 p-3 text-sm">
          <p>{mutation.data.answer}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {mutation.data.matched_video_ids.length > 0
              ? `${mutation.data.matched_video_ids.length} matching video(s) highlighted below.`
              : "No videos matched this query."}
          </p>
        </div>
      )}
    </div>
  );
}
