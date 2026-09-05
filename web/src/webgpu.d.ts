// The slice of the WebGPU IDL this page actually touches.
//
// Hand-written rather than pulling in `@webgpu/types`, because everything below is used from
// four call sites and none of it is the compute API: the prover itself is Rust, and wgpu
// owns every device, pipeline and buffer on the other side of the wasm boundary. What
// TypeScript needs to know about here is only what the page reads to decide whether to run
// at all and to label a result with the GPU that produced it.

interface GPUAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

interface GPUSupportedLimitsLike {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxStorageBuffersPerShaderStage: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
}

interface GPUAdapterLike {
  readonly info?: GPUAdapterInfoLike;
  readonly limits: GPUSupportedLimitsLike;
  readonly features: ReadonlySet<string>;
}

interface GPULike {
  requestAdapter(options?: {
    powerPreference?: 'low-power' | 'high-performance';
  }): Promise<GPUAdapterLike | null>;
}

interface Navigator {
  // Optional, and that is the point: a browser without WebGPU simply does not have it, and
  // one that has it can still hand back a null adapter.
  readonly gpu?: GPULike;
}

interface WorkerNavigator {
  readonly gpu?: GPULike;
}
