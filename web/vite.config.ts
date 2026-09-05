import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';

// The artifact bucket is a different origin, so every fetch of a zkey is a CORS request.
// In production that is the bucket's problem and it is solved by the rule in s3-cors.json.
// In dev it is solved here instead, by proxying /s3/* through the dev server, so a checkout
// with no AWS access can still run the whole benchmark end to end.
const BUCKET =
  (globalThis as { process?: { env?: Record<string, string> } }).process?.env?.G16_BUCKET ??
  'gpu-snark-bench';

/// Lets a browser that cannot be driven from a terminal report its own results.
///
/// Safari refuses WebDriver until "Allow remote automation" is switched on by hand in its
/// Developer settings, and a Safari-only failure is the kind that has to be reproduced rather
/// than reasoned about. With `?report=1` the page POSTs its finished rows here and they land
/// in the dev server's output. Dev only: it is a plugin with `apply: 'serve'`, so nothing
/// like it exists in a build.
const reportSink: Plugin = {
  name: 'g16-report-sink',
  apply: 'serve',
  configureServer(server) {
    server.middlewares.use('/__report', (req, res) => {
      // Typed locally rather than pulling in @types/node for one handler: this project has
      // no other Node-typed code and the two events used here are the whole surface.
      const stream = req as unknown as {
        on(event: 'data' | 'end', cb: (chunk?: unknown) => void): void;
      };
      const chunks: unknown[] = [];
      stream.on('data', (c) => chunks.push(c));
      stream.on('end', () => {
        console.log(`\n=== report ===\n${chunks.join('')}\n=== end report ===\n`);
        res.statusCode = 204;
        res.end();
      });
    });
  }
};

export default defineConfig({
  plugins: [sveltekit(), reportSink],
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
