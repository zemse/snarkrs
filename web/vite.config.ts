import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// The artifact bucket is a different origin, so every fetch of a zkey is a CORS request.
// In production that is the bucket's problem and it is solved by the rule in s3-cors.json.
// In dev it is solved here instead, by proxying /s3/* through the dev server, so a checkout
// with no AWS access can still run the whole benchmark end to end.
const BUCKET =
  (globalThis as { process?: { env?: Record<string, string> } }).process?.env?.G16_BUCKET ??
  'gpu-snark-bench';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/s3': {
        target: `https://${BUCKET}.s3.amazonaws.com`,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/s3/, '')
      }
    }
  }
});
