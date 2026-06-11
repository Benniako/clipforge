import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the API + generated media live on the FastAPI backend (port 8000);
// proxy them so the SPA can call same-origin paths. In prod the backend serves
// the built SPA from `dist/`, so these same paths resolve directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, ws: true },
      "/media": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
