//! The `cdylib` that `wasm-pack` links. It contains no logic and it is not supposed to.
//!
//! The browser prover is [`g16_wgpu::wasm`]. This crate exists only because a crate cannot be
//! both the `rlib` that `cargo test --workspace --release` builds twice and the `cdylib` that
//! `wasm-pack` needs: see this file's `Cargo.toml` and `g16-wgpu/tests/manifest.rs`, which
//! together record the build failure that made it necessary.
//!
//! # Why a `pub use` and not an empty file
//!
//! `#[wasm_bindgen]` puts its glue in custom sections that the linker merges, and it marks
//! every exported function with `#[export_name]`, so once the `rlib` is linked the exports
//! survive. What is not guaranteed is that the `rlib` is linked at all: an unused dependency
//! contributes nothing and `wasm-bindgen` then finds no sections and emits an empty module,
//! which fails as "g16_wgpu.js exports nothing" rather than as a link error. The re-export
//! below is what makes the dependency used.
//!
//! Build with `../../../webgpu-trial/build.sh`, which pins the `wasm-bindgen-cli` version to
//! the `wasm-bindgen` in `Cargo.lock`. A skew between them fails quietly at
//! `requestAdapter()`, not at build time.

#![cfg(target_arch = "wasm32")]

pub use g16_wgpu::wasm::*;
