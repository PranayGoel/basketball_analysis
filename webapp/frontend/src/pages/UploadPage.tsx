import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight, FileVideo } from "lucide-react";
import { uploadVideo } from "@/api/videos";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { JobProgressCard } from "@/components/upload/JobProgressCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isComplete, setIsComplete] = useState(false);

  const mutation = useMutation({
    mutationFn: uploadVideo,
    onSuccess: () => setIsComplete(false),
  });

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    setIsComplete(false);
    mutation.mutate(file);
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Upload game film</h1>
        <p className="text-muted-foreground">
          Upload a basketball game clip to get player tracking, possession analysis, and rule-violation
          detection.
        </p>
      </div>

      {!mutation.isPending && !mutation.isSuccess && (
        <UploadDropzone onFileSelected={handleFileSelected} disabled={mutation.isPending} />
      )}

      {mutation.isPending && (
        <Card>
          <CardContent className="flex items-center gap-3 py-6">
            <FileVideo className="h-5 w-5 shrink-0 text-primary" />
            <div>
              <p className="text-sm font-medium">Uploading {selectedFile?.name}&hellip;</p>
              <p className="text-xs text-muted-foreground">This may take a moment for large files.</p>
            </div>
          </CardContent>
        </Card>
      )}

      {mutation.isError && (
        <Card className="border-destructive/50">
          <CardContent className="py-6 text-sm text-destructive">
            Upload failed: {mutation.error instanceof Error ? mutation.error.message : "Unknown error"}
          </CardContent>
        </Card>
      )}

      {mutation.isSuccess && (
        <>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">{selectedFile?.name}</CardTitle>
            </CardHeader>
          </Card>
          <JobProgressCard jobId={mutation.data.job_id} onComplete={() => setIsComplete(true)} />
          {isComplete && (
            <Link to={`/videos/${mutation.data.video_id}`}>
              <Button className="w-full" size="lg">
                View results
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          )}
        </>
      )}
    </div>
  );
}
