/// Whether this browser can run the test at all, decided before the button is enabled.
///
/// Three separate things can be missing and they fail at different moments, so all three are
/// checked up front rather than being discovered one at a time by a visitor who has already
/// waited for a 100 MB download:
///
/// 1. **`navigator.gpu` absent.** No WebGPU. Firefox before 141, Safari before 26, Chrome on
///    Linux without the flag.
/// 2. **`requestAdapter()` resolves to null.** The API exists and there is no usable GPU
///    behind it: a blocklisted driver, a headless container, a VM. This is the case a page
///    that only checks `navigator.gpu` reports as "supported" and then dies on.
/// 3. **The adapter cannot meet the limits.** The prover needs a 128 MB storage binding, and
///    an adapter that offers less cannot run the larger circuits whatever else is true. The
///    spec floor is exactly 128 MB so this should always pass, which is the reason to check
///    it: if it ever fails, saying so beats a shader that fails to create.
///
/// The adapter's *actual* storage-binding size is returned on success, not just compared
/// against the floor. The prover opens its device at the `auto` profile, which raises that
/// one limit to whatever the adapter grants, so this number is what decides how far up the
/// circuit ladder this machine can go. See `circuits.ts`.
///
/// A secure context is a fourth requirement and is not checked here, because a browser
/// outside one does not expose `navigator.gpu` at all and so fails the first test. It is
/// worth knowing when reading a bug report: WebGPU needs HTTPS in production, with
/// `localhost` and `127.0.0.1` exempt.

export type Support =
  | {
      ok: true;
      vendor: string;
      architecture: string;
      device: string;
      /// What this adapter says it can bind in one storage buffer, which is what decides
      /// whether the largest circuits are in reach. Read here rather than after the device
      /// opens, because the page has to size the ladder and quote a download total before
      /// the worker exists. The device may still grant less; the runner reconciles.
      maxStorageBinding: number;
      maxBufferSize: number;
    }
  | { ok: false; reason: string; detail: string };

import { FLOOR_STORAGE_BINDING } from './circuits';

/// The floor every WebGPU implementation guarantees, and what the backend is built against.
const NEEDED_STORAGE_BINDING = FLOOR_STORAGE_BINDING;

export async function checkSupport(): Promise<Support> {
  if (!navigator.gpu) {
    return {
      ok: false,
      reason: 'This browser does not have WebGPU',
      detail:
        'navigator.gpu is not defined. WebGPU needs Chrome or Edge 113+, Chrome 121+ on ' +
        'Android, Safari 26+, or Firefox 141+. It also needs a secure context, so an HTTPS ' +
        'page or localhost.'
    };
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e: unknown) {
    return {
      ok: false,
      reason: 'Asking for a GPU adapter threw',
      detail: String((e as Error)?.message ?? e)
    };
  }

  if (!adapter) {
    return {
      ok: false,
      reason: 'WebGPU is present but there is no usable GPU behind it',
      detail:
        'requestAdapter() returned null. That usually means a blocklisted or missing ' +
        'graphics driver, a virtual machine, or a headless browser. Chrome reports the ' +
        'specific reason at chrome://gpu.'
    };
  }

  const have = adapter.limits.maxStorageBufferBindingSize;
  if (have < NEEDED_STORAGE_BINDING) {
    return {
      ok: false,
      reason: 'This GPU offers less than the WebGPU floor',
      detail:
        `maxStorageBufferBindingSize is ${have} bytes, below the ${NEEDED_STORAGE_BINDING} ` +
        'the specification guarantees and the prover is built against.'
    };
  }

  const i = adapter.info ?? {};
  return {
    ok: true,
    vendor: i.vendor ?? '',
    architecture: i.architecture ?? '',
    device: i.device ?? '',
    maxStorageBinding: have,
    maxBufferSize: adapter.limits.maxBufferSize
  };
}
