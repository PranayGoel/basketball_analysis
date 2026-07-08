import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Standard shadcn/ui className merge helper: combines clsx's conditional
 * class logic with tailwind-merge's conflict resolution (e.g. "p-2 p-4"
 * resolves to just "p-4" instead of emitting both).
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Formats a duration in seconds as M:SS (or H:MM:SS for videos over an
 * hour). Used by the video player and timeline.
 */
export function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "0:00";
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return hours > 0
    ? `${hours}:${pad(minutes)}:${pad(seconds)}`
    : `${minutes}:${pad(seconds)}`;
}

/** Formats a percentage (0-100) to at most one decimal place, e.g. 54.3%. */
export function formatPercent(value: number): string {
  return `${Math.round(value * 10) / 10}%`;
}

/** Formats an ISO datetime string as a short, human-readable date/time. */
export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
