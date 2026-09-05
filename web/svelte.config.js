import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// adapter-static, not adapter-auto. The whole page is one prerendered HTML file plus a wasm
// bundle, and there is no server side to it: every byte the benchmark touches is either in
// static/ or fetched from S3 at runtime. Vercel, Netlify, Cloudflare Pages and GitHub Pages
// all take the resulting build/ directory as-is, and none of them needs a Node runtime for it.
export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({ pages: 'build', assets: 'build', fallback: 'index.html', precompress: false })
  }
};
