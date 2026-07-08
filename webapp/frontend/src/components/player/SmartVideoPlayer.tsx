import { forwardRef, useImperativeHandle, useRef } from "react";
import { videoStreamUrl } from "@/api/client";

export interface SmartVideoPlayerHandle {
  /** Imperatively seeks the underlying <video> to the given time. */
  seek: (timeSec: number) => void;
}

export interface SmartVideoPlayerProps {
  videoId: string;
  /** Fired on every native timeupdate tick so a parent (and siblings via the
   * parent) can keep a play-position indicator in sync, e.g. EventTimeline. */
  onTimeUpdate?: (currentTimeSec: number) => void;
  /** Fired once metadata loads, with the video's true duration in seconds. */
  onDurationChange?: (durationSec: number) => void;
  className?: string;
}

/**
 * Thin wrapper around a plain <video> element. Exposes an imperative
 * `seek(timeSec)` method via forwardRef + useImperativeHandle so a parent
 * page can wire a sibling component's onSeek callback to this player
 * without lifting playback state into the parent (VideoDetailPage holds
 * the ref and passes `(t) => playerRef.current?.seek(t)` down to
 * EventTimeline's onSeek prop).
 */
export const SmartVideoPlayer = forwardRef<SmartVideoPlayerHandle, SmartVideoPlayerProps>(
  function SmartVideoPlayer({ videoId, onTimeUpdate, onDurationChange, className }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);

    useImperativeHandle(
      ref,
      () => ({
        seek(timeSec: number) {
          const video = videoRef.current;
          if (!video) return;
          video.currentTime = timeSec;
          // If the user clicked a timeline marker while paused, honor that
          // intent by resuming playback -- clicking a marker implies "take
          // me there and keep going," not "pause here."
          void video.play().catch(() => {
            // Autoplay can be rejected by the browser (e.g. no prior user
            // gesture in this exact call stack in some browsers); the seek
            // itself still succeeded, so this is safe to swallow.
          });
        },
      }),
      [],
    );

    return (
      <video
        ref={videoRef}
        src={videoStreamUrl(videoId)}
        controls
        className={className ?? "aspect-video w-full rounded-lg border border-border bg-black"}
        onTimeUpdate={(e) => onTimeUpdate?.(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => onDurationChange?.(e.currentTarget.duration)}
      >
        Your browser does not support the video tag.
      </video>
    );
  },
);
