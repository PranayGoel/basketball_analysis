import { useRef, type MouseEvent } from "react";
import type { TimelineEvent } from "@/api/types";
import { TimelineMarker } from "./TimelineMarker";
import { formatDuration } from "@/lib/utils";

export interface EventTimelineProps {
  durationSec: number;
  currentTimeSec: number;
  events: TimelineEvent[];
  onSeek: (timeSec: number) => void;
}

/**
 * Custom horizontal scrubber for a video's duration, with clickable
 * event-marker ticks layered on top.
 *
 * This is a plain styled <div> track with absolutely-positioned children
 * rather than a native <input type="range"> -- a range input can't host
 * arbitrary positioned child elements (the event markers) cleanly, since
 * its thumb/track are rendered by the browser's own UA styling and don't
 * expose a slot for extra DOM content at arbitrary points along the track.
 */
export function EventTimeline({ durationSec, currentTimeSec, events, onSeek }: EventTimelineProps) {
  const trackRef = useRef<HTMLDivElement>(null);

  // Guard against durationSec being 0/NaN (e.g. metadata not loaded yet) --
  // percentages would divide by zero otherwise.
  const safeDuration = durationSec > 0 ? durationSec : 1;
  const playheadPercent = Math.min(100, Math.max(0, (currentTimeSec / safeDuration) * 100));

  const handleTrackClick = (e: MouseEvent<HTMLDivElement>) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const clickRatio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    onSeek(clickRatio * safeDuration);
  };

  return (
    <div className="space-y-1.5">
      <div
        ref={trackRef}
        role="slider"
        aria-label="Video timeline"
        aria-valuemin={0}
        aria-valuemax={durationSec}
        aria-valuenow={currentTimeSec}
        tabIndex={0}
        onClick={handleTrackClick}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") onSeek(Math.min(safeDuration, currentTimeSec + 5));
          if (e.key === "ArrowLeft") onSeek(Math.max(0, currentTimeSec - 5));
        }}
        className="relative h-8 w-full cursor-pointer rounded-md bg-muted"
      >
        {/* Base track fill up to the current playback position. */}
        <div
          className="pointer-events-none absolute inset-y-0 left-0 rounded-md bg-primary/20"
          style={{ width: `${playheadPercent}%` }}
        />

        {/* Current playback position indicator -- a thin vertical line, so
            the timeline also functions as a progress indicator and not
            just an event-marker strip. */}
        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-foreground"
          style={{ left: `${playheadPercent}%` }}
        />

        {/* Event marker ticks. Renders nothing extra when events is empty --
            the base track above still renders fine on its own. */}
        {events.map((event, index) => (
          <TimelineMarker
            key={`${event.event_type}-${event.frame_num}-${index}`}
            event={event}
            positionPercent={(event.timestamp_sec / safeDuration) * 100}
            onSeek={onSeek}
          />
        ))}
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{formatDuration(currentTimeSec)}</span>
        <span>{formatDuration(durationSec)}</span>
      </div>
    </div>
  );
}
