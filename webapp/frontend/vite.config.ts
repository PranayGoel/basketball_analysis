import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      // Proxies /api/* to the FastAPI backend during `npm run dev` so that
      // fetch("/api/videos") and new EventSource("/api/jobs/x/stream") work
      // without any CORS configuration on the frontend side. The backend
      // still needs its own CORS middleware for direct (non-proxied) access.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
