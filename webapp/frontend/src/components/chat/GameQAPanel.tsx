import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { MessageCircleQuestion, Send } from "lucide-react";
import { askQuestion } from "@/api/reports";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export interface GameQAPanelProps {
  videoId: string;
}

/**
 * Simple chat-style Q&A UI for asking free-text questions about a single
 * game's report. Deliberately keeps only the single most-recent
 * question/answer pair in state rather than a full conversation history --
 * the backend's /qa endpoint is stateless per-call (no thread/session id in
 * the contract), so maintaining a growing local transcript would imply a
 * multi-turn conversational memory the backend doesn't actually provide.
 */
export function GameQAPanel({ videoId }: GameQAPanelProps) {
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: (q: string) => askQuestion(videoId, q),
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;
    setLastQuestion(trimmed);
    mutation.mutate(trimmed);
    setQuestion("");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageCircleQuestion className="h-4 w-4 text-primary" />
          Ask about this game
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. Who had the most turnovers?"
            disabled={mutation.isPending}
          />
          <Button type="submit" size="icon" disabled={mutation.isPending || !question.trim()}>
            <Send className="h-4 w-4" />
            <span className="sr-only">Ask</span>
          </Button>
        </form>

        {lastQuestion && (
          <div className="space-y-2 rounded-md border border-border bg-accent/30 p-3 text-sm">
            <p className="font-medium">{lastQuestion}</p>
            {mutation.isPending && (
              <div className="space-y-1.5">
                <Skeleton className="h-3.5 w-full" />
                <Skeleton className="h-3.5 w-4/5" />
              </div>
            )}
            {mutation.isError && (
              <p className="text-destructive">
                {mutation.error instanceof Error ? mutation.error.message : "Something went wrong."}
              </p>
            )}
            {mutation.isSuccess && (
              <p className="whitespace-pre-line text-foreground/90">{mutation.data.answer}</p>
            )}
          </div>
        )}

        {!lastQuestion && (
          <p className="text-sm text-muted-foreground">
            Ask a question about possession, passes, violations, or player performance.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
