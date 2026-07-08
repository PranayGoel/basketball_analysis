import { Navigate, Route, Routes } from "react-router-dom";
import { UploadCloud, Library } from "lucide-react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import UploadPage from "@/pages/UploadPage";
import LibraryPage from "@/pages/LibraryPage";
import VideoDetailPage from "@/pages/VideoDetailPage";
import NotFoundPage from "@/pages/NotFoundPage";

function NavBar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    cn(
      "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
      isActive
        ? "bg-primary/15 text-primary"
        : "text-muted-foreground hover:bg-accent hover:text-foreground",
    );

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="container flex h-14 items-center justify-between">
        <NavLink to="/library" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <circle cx="12" cy="12" r="10" fillOpacity="0.15" />
              <path
                d="M12 2a10 10 0 0 0-7.07 17.07M12 2a10 10 0 0 1 7.07 17.07M12 2v20M2 12h20"
                stroke="currentColor"
                strokeWidth="1.4"
                fill="none"
              />
            </svg>
          </span>
          <span>CourtVision</span>
        </NavLink>
        <nav className="flex items-center gap-1">
          <NavLink to="/library" className={linkClass}>
            <Library className="h-4 w-4" />
            Library
          </NavLink>
          <NavLink to="/upload" className={linkClass}>
            <UploadCloud className="h-4 w-4" />
            Upload
          </NavLink>
        </nav>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="container py-8">
        <Routes>
          <Route path="/" element={<Navigate to="/library" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/videos/:id" element={<VideoDetailPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </main>
    </div>
  );
}
