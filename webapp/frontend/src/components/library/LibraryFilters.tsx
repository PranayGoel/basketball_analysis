import type { SortOrder, VideoSortBy, VideoStatus } from "@/api/types";
import { cn } from "@/lib/utils";

export interface LibraryFilterState {
  status: VideoStatus | "all";
  sortBy: VideoSortBy;
  sortOrder: SortOrder;
}

export interface LibraryFiltersProps {
  value: LibraryFilterState;
  onChange: (value: LibraryFilterState) => void;
}

const STATUS_OPTIONS: Array<{ value: VideoStatus | "all"; label: string }> = [
  { value: "all", label: "All statuses" },
  { value: "queued", label: "Queued" },
  { value: "processing", label: "Processing" },
  { value: "done", label: "Done" },
  { value: "failed", label: "Failed" },
];

const SORT_OPTIONS: Array<{ value: VideoSortBy; label: string }> = [
  { value: "uploaded_at", label: "Upload date" },
  { value: "total_passes", label: "Total passes" },
  { value: "max_player_speed_kmh", label: "Max player speed" },
];

const selectClass =
  "h-9 rounded-md border border-input bg-transparent px-3 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";

/** Status + sort filter controls for the library grid. */
export function LibraryFilters({ value, onChange }: LibraryFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-sm text-muted-foreground">
        Status
        <select
          className={cn(selectClass, "min-w-[9rem]")}
          value={value.status}
          onChange={(e) =>
            onChange({ ...value, status: e.target.value as VideoStatus | "all" })
          }
        >
          {STATUS_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm text-muted-foreground">
        Sort by
        <select
          className={cn(selectClass, "min-w-[10rem]")}
          value={value.sortBy}
          onChange={(e) => onChange({ ...value, sortBy: e.target.value as VideoSortBy })}
        >
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>

      <button
        type="button"
        onClick={() =>
          onChange({ ...value, sortOrder: value.sortOrder === "asc" ? "desc" : "asc" })
        }
        className="h-9 rounded-md border border-input bg-transparent px-3 text-sm shadow-sm transition-colors hover:bg-accent"
        aria-label={`Sort order: ${value.sortOrder === "asc" ? "ascending" : "descending"}`}
      >
        {value.sortOrder === "asc" ? "↑ Asc" : "↓ Desc"}
      </button>
    </div>
  );
}
