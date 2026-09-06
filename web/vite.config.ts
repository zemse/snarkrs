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

/// A self-signed certificate, if one has been generated, so a phone on the same Wi-Fi can
/// reach this machine over **https**.
///
/// WebGPU needs a secure context and only `localhost` is exempt, so `http://192.168.x.x`
/// hands the page a browser with no `navigator.gpu` at all: the site reports "unsupported"
/// and nothing is testable. Serving the LAN address over TLS is the whole difference between
/// iterating against a real iPhone in seconds and redeploying for every one-line change.
/// The certificate is trusted by nothing, so Safari shows an interstitial once per device;
/// accepting it still leaves the origin `https:` and therefore secure.
///
/// Absent by default and entirely optional. Generate with:
///
///     ip=$(ipconfig getifaddr en0)
///     mkdir -p .certs && openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 365 \
///       -keyout .certs/key.pem -out .certs/cert.pem -subj "/CN=$ip" \
///       -addext "subjectAltName=IP:$ip,DNS:localhost,IP:127.0.0.1"
///
/// `readFileSync` is typed here rather than by adding @types/node, for the same reason the
/// request stream is typed inside `reportSink`: one function is the whole surface. The
/// specifier is a variable so it stays out of the module graph and out of type resolution.
async function selfSigned() {
  try {
    const spec = 'node:fs';
    const fs = (await import(/* @vite-ignore */ spec)) as {
      readFileSync(path: string): Uint8Array;
    };
    return { key: fs.readFileSync('.certs/key.pem'), cert: fs.readFileSync('.certs/cert.pem') };
  } catch {
    return undefined;
  }
}

export default defineConfig(async () => {
  const https = await selfSigned();
  // `host` only when there is a certificate to serve with. Exposing the server on the
  // network over plain http would be worse than not exposing it at all: the page would
  // load and then report that this browser has no WebGPU.
  const lan = { host: https ? true : undefined, https };
  return {
    plugins: [sveltekit(), reportSink],
    server: {
      ...lan,
      proxy: {
        '/s3': {
          target: `https://${BUCKET}.s3.amazonaws.com`,
          changeOrigin: true,
          rewrite: (p: string) => p.replace(/^\/s3/, '')
        }
      }
    },
    preview: lan
  };
});
