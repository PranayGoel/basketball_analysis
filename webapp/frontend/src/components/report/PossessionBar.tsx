import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatPercent } from "@/lib/utils";

export interface PossessionBarProps {
  teamOnePct: number;
  teamTwoPct: number;
  undecidedPct: number;
}

/** A labeled, segmented possession-split bar -- doesn't need a full charting library for a single stacked stat. */
export function PossessionBar({ teamOnePct, teamTwoPct, undecidedPct }: PossessionBarProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Team possession</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex h-6 w-full overflow-hidden rounded-full">
          <div
            className="flex items-center justify-center bg-team-a text-[11px] font-medium text-team-a-foreground"
            style={{ width: `${teamOnePct}%` }}
          >
            {teamOnePct >= 12 && formatPercent(teamOnePct)}
          </div>
          <div
            className="flex items-center justify-center bg-team-b text-[11px] font-medium text-team-b-foreground"
            style={{ width: `${teamTwoPct}%` }}
          >
            {teamTwoPct >= 12 && formatPercent(teamTwoPct)}
          </div>
          {undecidedPct > 0 && (
            <div
              className="flex items-center justify-center bg-muted text-[11px] font-medium text-muted-foreground"
              style={{ width: `${undecidedPct}%` }}
            >
              {undecidedPct >= 12 && formatPercent(undecidedPct)}
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <Legend swatchClass="bg-team-a" label={`Team A: ${formatPercent(teamOnePct)}`} />
          <Legend swatchClass="bg-team-b" label={`Team B: ${formatPercent(teamTwoPct)}`} />
          {undecidedPct > 0 && (
            <Legend swatchClass="bg-muted" label={`Undecided: ${formatPercent(undecidedPct)}`} />
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Legend({ swatchClass, label }: { swatchClass: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`h-2.5 w-2.5 rounded-full ${swatchClass}`} />
      {label}
    </span>
  );
}
