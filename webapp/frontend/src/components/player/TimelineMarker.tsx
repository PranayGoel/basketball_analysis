import type { TimelineEvent, TimelineEventType } from "@/api/types";
import { Tooltip } from "@/components/ui/tooltip";
import { formatDuration } from "@/lib/utils";

/**
 * Color lookup keyed by event_type, not a single hardcoded color. Today the
 * backend only ever emits "violation" events, but this map is written to be
 * trivially extended the day "pass"/"interception" events land -- adding a
 * new key here (and to EVENT_LABELS below) is the only change needed.
 */
const EVENT_COLORS: Record<string, string> = {
  violation: "bg-violation border-violation",
};
const DEFAULT_COLOR = "bg-primary border-primary";

const EVENT_LABELS: Record<string, string> = {
  violation: "Violation",
};

function colorForEventType(eventType: TimelineEventType): string {
  return EVENT_COLORS[eventType] ?? DEFAULT_COLOR;
}

function labelForEventType(eventType: TimelineEventType): string {
  return EVENT_LABELS[eventType] ?? eventType;
}

export interface TimelineMarkerProps {
  event: TimelineEvent;
  /** Position along the track, 0-100. */
  positionPercent: number;
  onSeek: (timeSec: number) => void;
}

export function TimelineMarker({ event, positionPercent, onSeek }: TimelineMarkerProps) {
  const clampedPercent = Math.min(100, Math.max(0, positionPercent));

  return (
    <Tooltip
      side="top"
      content={
        <div className="space-y-0.5">
          <p className="font-medium">{labelForEventType(event.event_type)}</p>
          <p className="text-muted-foreground">
            Player #{event.player_id} &middot; {formatDuration(event.timestamp_sec)}
          </p>
          {event.detail.violation_type && (
            <p className="text-muted-foreground">{event.detail.violation_type.replace(/_/g, " ")}</p>
          )}
        </div>
      }
    >
      <button
        type="button"
        onClick={() => onSeek(event.timestamp_sec)}
        aria-label={`${labelForEventType(event.event_type)} at ${formatDuration(event.timestamp_sec)}, seek to this moment`}
        className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 ${colorForEventType(
          event.event_type,
        )} shadow-sm transition-transform hover:scale-125 focus-visible:scale-125 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background`}
        style={{ left: `${clampedPercent}%` }}
      />
    </Tooltip>
  );
}
