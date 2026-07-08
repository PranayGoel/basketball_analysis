import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Minimal CSS-only tooltip (no Radix dependency, per the hand-written
 * shadcn-fallback approach used across src/components/ui). Shows on
 * hover/focus of its child trigger; positioned above the trigger by
 * default. Good enough for short, single-line hints (e.g. timeline event
 * details) -- not intended as a full floating-ui replacement with
 * collision detection.
 */
export interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactElement;
  side?: "top" | "bottom";
  className?: string;
}

export function Tooltip({ content, children, side = "top", className }: TooltipProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          className={cn(
            "pointer-events-none absolute z-50 whitespace-nowrap rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground shadow-md",
            "left-1/2 -translate-x-1/2",
            side === "top" ? "bottom-full mb-2" : "top-full mt-2",
            "animate-in fade-in-0 zoom-in-95 duration-100",
            className,
          )}
        >
          {content}
        </span>
      )}
    </span>
  );
}
