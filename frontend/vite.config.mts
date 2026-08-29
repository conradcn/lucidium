import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Content-Security-Policy for the renderer. ``default-src 'none'`` means
// nothing loads unless it's listed below:
//   * script-src 'self'      — only the bundle vite emits; no inline, no eval.
//   * style-src 'unsafe-inline' — React inline ``style={{...}}`` props and the
//     animation styles the screens set imperatively.
//   * img/media-src          — bundled art plus generated assets served over
//     the confined ``lucidium-asset:`` scheme (see electron/assetPath.ts);
//     ``data:``/``blob:`` cover vite-inlined small assets and object URLs.
//   * connect-src            — the loopback WebSocket to the Python backend,
//     bound to an ephemeral port.
// Injected at BUILD time only: the vite dev server needs an inline preamble
// script and its own HMR socket, and dev already runs against a dev-server
// origin rather than the packaged file:// page that ships to players.
const CONTENT_SECURITY_POLICY = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: lucidium-asset:",
  "media-src 'self' data: blob: lucidium-asset:",
  "font-src 'self' data:",
  "connect-src ws://127.0.0.1:* ws://localhost:* lucidium-asset: data:",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-src 'none'",
  "object-src 'none'",
].join("; ");

function cspPlugin(): Plugin {
  return {
    name: "lucidium-csp",
    apply: "build",
    transformIndexHtml(html) {
      return {
        html,
        tags: [
          {
            tag: "meta",
            injectTo: "head-prepend",
            attrs: {
              "http-equiv": "Content-Security-Policy",
              content: CONTENT_SECURITY_POLICY,
            },
          },
        ],
      };
    },
  };
}

export default defineConfig({
  // Relative paths so Electron can load dist/index.html via file://
  // without absolute /assets references resolving to drive root.
  base: "./",
  plugins: [react(), cspPlugin()],
  resolve: {
    alias: {
      // ``import.meta.dirname``, not ``__dirname``: this file is ESM
      // (``.mts``), where ``__dirname`` only exists because Vite's
      // legacy config loader shims it. The native loader — already the
      // default in a future Vite major — does not, and warns today.
      // Requires Node >= 20.11; package.json floors us well above that.
      "@": path.resolve(import.meta.dirname, "src"),
      "@shared": path.resolve(import.meta.dirname, "src/shared"),
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  server: {
    // ``release/`` is the electron-builder output (win-unpacked + the
    // 7z archive). It contains thousands of bundled HTML files
    // (chromium licences, torch / numba template skeletons) that
    // vite would otherwise try to live-reload every time
    // ``package.ps1`` rewrites them, eventually crashing esbuild
    // with "The service is no longer running". ``dist/`` is the
    // vite build output itself — same reasoning.
    watch: {
      ignored: ["**/release/**", "**/dist/**"],
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}", "src/**/*.test.{ts,tsx}"],
  },
});
