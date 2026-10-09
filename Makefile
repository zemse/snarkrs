# Release pipeline. .github/workflows/release.yml only calls these targets, so a release
# built here and one built in CI run the same commands with the same settings.
#
# A release is two commands on a machine logged in to crates.io (`cargo login`):
#
#   make bump VERSION=0.2.0         set every crate to 0.2.0; review and commit
#   make release                    publish the crates, then push the tag CI builds from
#
# The pieces, each runnable alone:
#
#   make release-build TARGET=...   build and package the snarkrs binary for one target
#   make release-matrix             the CI build matrix, as JSON
#   make check-tag TAG=v0.2.0       fail unless the tag matches the workspace version
#   make publish-dry-run            package and verify every crate, upload nothing
#   make github-release TAG=...     create the GitHub release from target/dist
#   make publish-crates             publish every crate not yet on crates.io
#
# GNU Make 3.81 is what macOS ships, so nothing here needs a newer one.

SHELL := bash

# The one toolchain every release binary is built with. A different rustc is a different
# binary, so this is pinned rather than read from whatever `stable` is on the day.
RUST_TOOLCHAIN ?= 1.99.0
CARGO := cargo +$(RUST_TOOLCHAIN)
PYTHON ?= python3

# Every target gets the same features. `metal` compiles to nothing off macOS and `cuda`
# dlopens its libraries only when `--backend cuda` is picked, so neither costs a machine
# without the hardware anything but binary size. `cpu`, `wgpu` and `witness-wasm` are the
# package defaults.
FEATURES := metal,cuda

# Target, then the GitHub runner that builds it natively. Linux builds on 22.04 so the
# binary needs glibc 2.35, not the newer one a later image would link against.
RELEASE_TARGETS := \
	aarch64-apple-darwin:macos-26 \
	x86_64-unknown-linux-gnu:ubuntu-22.04 \
	aarch64-unknown-linux-gnu:ubuntu-22.04-arm \
	x86_64-pc-windows-msvc:windows-2025-vs2026

DIST := target/dist
# A target dir of its own: the remapping flags below differ from a normal build's, and
# sharing target/ would rebuild everything each time you switched between the two.
BUILD_DIR := target/release-build
EXE := $(if $(findstring windows,$(TARGET)),.exe,)
ARCHIVE := snarkrs-$(TARGET)
CARGO_HOME ?= $(HOME)/.cargo
UA := snarkrs-release (https://github.com/zemse/snarkrs)

# The commit time, not the build time, for every timestamp that ends up in the archive.
SOURCE_DATE_EPOCH := $(shell git log -1 --format=%ct)
export SOURCE_DATE_EPOCH

# Checkout and registry paths are machine specific, so they are rewritten to fixed ones
# rather than baked into panic messages. With the rust-src component installed, rustc points
# std's paths at the local copy instead of /rustc/<commit>, so that is mapped back too, or a
# build here and one on a minimal CI toolchain would differ. `=` rather than `:=` because the
# toolchain may not be installed until the recipe runs.
RELEASE_RUSTFLAGS = --remap-path-prefix=$(CURDIR)=/snarkrs --remap-path-prefix=$(CARGO_HOME)=/cargo \
	--remap-path-prefix=$$(rustc +$(RUST_TOOLCHAIN) --print sysroot)/lib/rustlib/src/rust=/rustc/$$(rustc +$(RUST_TOOLCHAIN) -vV | sed -n 's/^commit-hash: //p')

# The workspace version, as a shell command. Inline rather than a recursive $(MAKE), whose
# "Entering directory" lines Make 3.81 prints even under -s. Plain `cargo`, because reading
# a manifest needs no particular toolchain and the CI plan job never installs the pinned one.
VERSION_CMD := cargo metadata --format-version 1 --no-deps --offline | $(PYTHON) -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "snarkrs"))'
comma := ,

.PHONY: release bump version release-build release-matrix check-tag publish-dry-run github-release publish-crates

# Crates go first and the tag last, so a tag on GitHub always means the crates are out.
# Safe to run again after a failure: published crates are skipped and the tag is only
# created once.
release:
	@test -z "$$(git status --porcelain)" || { echo "commit or stash your changes first"; exit 1; }
	@test "$$(git branch --show-current)" = main || { echo "release from main"; exit 1; }
	rustup toolchain install $(RUST_TOOLCHAIN) --profile minimal --no-self-update
	$(MAKE) publish-dry-run
	git push origin main
	$(MAKE) publish-crates
	@v=$$($(VERSION_CMD)); \
	git rev-parse -q --verify "refs/tags/v$$v" >/dev/null || git tag -a "v$$v" -m "v$$v"; \
	git push origin "v$$v"

bump:
	@test -n "$(VERSION)" || { echo "usage: make bump VERSION=x.y.z"; exit 2; }
	@old=$$($(VERSION_CMD)); \
	perl -pi -e 's/^version = "\Q'"$$old"'\E"$$/version = "$(VERSION)"/; s/^(snarkrs[\w-]* *= \{ path = "[^"]+", version = )"\Q'"$$old"'\E"/$$1"$(VERSION)"/' Cargo.toml; \
	$(CARGO) update --workspace --offline; \
	echo "$$old -> $$($(VERSION_CMD)). Commit, then: git tag v$(VERSION) && git push origin main v$(VERSION)"

version:
	@$(VERSION_CMD)

release-matrix:
	@echo '{"include":[$(subst } {,}$(comma){,$(foreach t,$(RELEASE_TARGETS),{"target":"$(word 1,$(subst :, ,$(t)))"$(comma)"runner":"$(word 2,$(subst :, ,$(t)))"}))]}'

check-tag:
	@test -n "$(TAG)" || { echo "usage: make check-tag TAG=vX.Y.Z"; exit 2; }
	@v=$$($(VERSION_CMD)); test "$(TAG)" = "v$$v" || { echo "tag $(TAG) does not match workspace version $$v"; exit 1; }

release-build:
	@test -n "$(TARGET)" || { echo "usage: make release-build TARGET=<triple>"; exit 2; }
	rustup toolchain install $(RUST_TOOLCHAIN) --profile minimal --target $(TARGET) --no-self-update
	RUSTFLAGS="$(RELEASE_RUSTFLAGS)" CARGO_TARGET_DIR=$(BUILD_DIR) \
		$(CARGO) build --release --locked -p snarkrs --bin snarkrs --target $(TARGET) --features $(FEATURES)
	mkdir -p $(DIST)
	$(PYTHON) -c "$$PACKAGE_PY" $(DIST)/$(ARCHIVE).tar.xz $(ARCHIVE) \
		$(BUILD_DIR)/$(TARGET)/release/snarkrs$(EXE) LICENSE-MIT LICENSE-APACHE README.md
	@cat $(DIST)/$(ARCHIVE).tar.xz.sha256

# Packs the binary as snarkrs-<target>/snarkrs inside snarkrs-<target>.tar.xz, the layout
# cargo-binstall looks for without any configuration. Owners, modes and mtimes are fixed so
# the same binary always gives the same archive; macOS tar cannot do that, Python can.
define PACKAGE_PY
import hashlib, os, sys, tarfile
out, root, binary, *docs = sys.argv[1:]
epoch = int(os.environ["SOURCE_DATE_EPOCH"])
def entry(name, mode, size=0, kind=tarfile.REGTYPE):
    ti = tarfile.TarInfo(f"{root}/{name}" if name else root)
    ti.type, ti.mode, ti.size, ti.mtime = kind, mode, size, epoch
    return ti
with tarfile.open(out, "w:xz", format=tarfile.USTAR_FORMAT) as tar:
    tar.addfile(entry("", 0o755, kind=tarfile.DIRTYPE))
    for path, mode in [(binary, 0o755)] + [(d, 0o644) for d in docs]:
        with open(path, "rb") as f:
            tar.addfile(entry(os.path.basename(path), mode, os.path.getsize(path)), f)
digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
open(out + ".sha256", "w", newline="\n").write(f"{digest}  {os.path.basename(out)}\n")
endef
export PACKAGE_PY

publish-dry-run:
	$(CARGO) publish --workspace --locked --dry-run

github-release:
	@test -n "$(TAG)" || { echo "usage: make github-release TAG=vX.Y.Z"; exit 2; }
	@if gh release view $(TAG) >/dev/null 2>&1; then \
		gh release upload $(TAG) $(DIST)/*.tar.xz $(DIST)/*.sha256 --clobber; \
	else \
		gh release create $(TAG) $(DIST)/*.tar.xz $(DIST)/*.sha256 --verify-tag --title $(TAG) --generate-notes; \
	fi

# Skips crates whose version is already on crates.io, so a run that failed halfway can
# simply be run again.
publish-crates:
	@v=$$($(VERSION_CMD)); skip=""; \
	for p in $$($(CARGO) metadata --format-version 1 --no-deps --offline | $(PYTHON) -c 'import json,sys; print(" ".join(p["name"] for p in json.load(sys.stdin)["packages"] if p["publish"] != []))'); do \
		code=$$(curl -s -o /dev/null -w '%{http_code}' -A "$(UA)" https://crates.io/api/v1/crates/$$p/$$v); \
		if [ "$$code" = 200 ]; then echo "$$p $$v is already on crates.io"; skip="$$skip --exclude $$p"; fi; \
	done; \
	$(CARGO) publish --workspace --locked $$skip
