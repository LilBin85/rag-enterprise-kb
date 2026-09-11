import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/v1": {
        target: mode.startsWith("e2e_") ? process.env.RAG_KB_E2E_API_URL || "http://localhost:8000" : "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
}));
