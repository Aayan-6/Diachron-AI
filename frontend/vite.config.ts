import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const BACKEND = process.env.DIACHRON_BACKEND ?? process.env.SATQUERY_BACKEND ?? 'http://127.0.0.1:8000';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    // maplibre-gl loads its own module worker at runtime. Vite's dependency
    // pre-bundling rewrites the entry but does not emit the worker chunk
    // alongside it, so the worker request 404s, the style never finishes
    // loading, and the map renders as an empty canvas with no error. Serving
    // maplibre unbundled keeps the worker URL resolvable.
    exclude: ['maplibre-gl'],
  },
  server: {
    // Bind on all interfaces: the backend advertises its LAN IP and enables
    // permissive CORS specifically so a second laptop can drive this UI.
    host: true,
    proxy: {
      // The client talks to same-origin /api/v1 and /static, so both dev and
      // LAN access work without hard-coding a backend host in the bundle.
      '/api': { target: BACKEND, changeOrigin: true },
      '/static': { target: BACKEND, changeOrigin: true },
      // Proxy Bhuvan WMS to bypass strict CORS policies on the browser
      '/bhuvan-wms': {
        target: 'https://bhuvan-vec1.nrsc.gov.in',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/bhuvan-wms/, '/bhuvan/gwc/service/wms/')
      },
    },
  },
  preview: {
    host: true,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/static': { target: BACKEND, changeOrigin: true },
    },
  },
});
