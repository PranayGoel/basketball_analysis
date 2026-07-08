import { Link } from "react-router-dom";
import { CircleOff } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
      <CircleOff className="h-12 w-12 text-muted-foreground" />
      <div>
        <h1 className="text-xl font-semibold">Page not found</h1>
        <p className="text-muted-foreground">The page you&apos;re looking for doesn&apos;t exist.</p>
      </div>
      <Link to="/library">
        <Button variant="secondary">Back to library</Button>
      </Link>
    </div>
  );
}
