import { useQuery } from "@tanstack/react-query";
import { fetchVideos } from "@/api/videos";
import type { VideoListFilters } from "@/api/types";

/**
 * TanStack Query wrapper around GET /api/videos, keyed by the full filter
 * object so changing any filter (status, sort, page) triggers a refetch
 * and is independently cached.
 */
export function useVideoLibrary(filters: VideoListFilters) {
  return useQuery({
    queryKey: ["videos", filters],
    queryFn: () => fetchVideos(filters),
    placeholderData: (previousData) => previousData, // avoid layout flash when paging/filtering
  });
}
