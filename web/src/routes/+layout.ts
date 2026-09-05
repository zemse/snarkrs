// One page, prerendered to static HTML. `ssr` is off because everything on it touches
// `navigator.gpu`, `Worker` and `performance`, none of which exist in a Node prerender.
export const prerender = true;
export const ssr = false;
