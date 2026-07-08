import * as React from "react";
import { FileVideo, UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

export interface UploadDropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
  accept?: string;
}

/** Drag-and-drop + click-to-browse file input for video uploads. */
export function UploadDropzone({
  onFileSelected,
  disabled = false,
  accept = "video/*",
}: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFileSelected(file);
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={cn(
        "flex min-h-[280px] cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-10 text-center transition-colors",
        isDragging
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-accent/40",
        disabled && "cursor-not-allowed opacity-50 hover:border-border hover:bg-transparent",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
        {isDragging ? <FileVideo className="h-8 w-8" /> : <UploadCloud className="h-8 w-8" />}
      </div>
      <div className="space-y-1">
        <p className="text-base font-medium">
          {isDragging ? "Drop it right here" : "Drag & drop your game film"}
        </p>
        <p className="text-sm text-muted-foreground">
          or <span className="font-medium text-primary">click to browse</span> &middot; MP4, MOV, AVI
        </p>
      </div>
    </div>
  );
}
