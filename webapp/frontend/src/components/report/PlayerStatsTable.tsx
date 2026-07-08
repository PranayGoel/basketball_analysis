import type { PlayerStats } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export interface PlayerStatsTableProps {
  players: Record<string, PlayerStats>;
}

/** Sortable-by-eye stats table for every tracked player in the game report. */
export function PlayerStatsTable({ players }: PlayerStatsTableProps) {
  const entries = Object.entries(players).sort(
    ([, a], [, b]) => b.total_distance_m - a.total_distance_m,
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Player statistics</CardTitle>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <p className="text-sm text-muted-foreground">No player data available for this video.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th className="pb-2 pr-4 font-medium">Player</th>
                  <th className="pb-2 pr-4 font-medium">Team</th>
                  <th className="pb-2 pr-4 text-right font-medium">Distance (m)</th>
                  <th className="pb-2 pr-4 text-right font-medium">Avg speed (km/h)</th>
                  <th className="pb-2 text-right font-medium">Max speed (km/h)</th>
                </tr>
              </thead>
              <tbody>
                {entries.map(([playerId, stats]) => (
                  <tr key={playerId} className="border-b border-border/50 last:border-0">
                    <td className="py-2 pr-4 font-medium">{stats.label}</td>
                    <td className="py-2 pr-4">
                      <TeamBadge team={stats.team} />
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {stats.total_distance_m.toFixed(1)}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">{stats.avg_speed_kmh.toFixed(1)}</td>
                    <td className="py-2 text-right tabular-nums">{stats.max_speed_kmh.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function TeamBadge({ team }: { team: number | null }) {
  if (team === null) return <Badge variant="outline">Unassigned</Badge>;
  return (
    <Badge
      className={cn(
        "border-transparent",
        team === 1 ? "bg-team-a text-team-a-foreground" : "bg-team-b text-team-b-foreground",
      )}
    >
      Team {team === 1 ? "A" : "B"}
    </Badge>
  );
}
