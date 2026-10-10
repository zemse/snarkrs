#!/usr/bin/env bash
# Build snarkrs, verify_time and field_ops once per arkworks checkout, for bench/scripts/arkab.py.
#
#   bench/scripts/arkab-build.sh KIND OUT_DIR NAME=ALGEBRA_DIR...
#
# KIND is mac (aarch64-apple-darwin, cpu + metal), x86 (x86_64 linux, cpu, the Makefile's
# +adx,+bmi2) or cuda (x86_64 linux, cpu + cuda, same flags). Every variant builds from the
# same source, flags and toolchain; only the [patch.crates-io] paths differ (ark-patch.sh),
# each into a target dir of its own. Writes OUT_DIR/snarkrs-NAME, verify_time-NAME,
# field_ops-NAME and OUT_DIR/build.json. Leaves the patch block out of Cargo.toml at the end.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
kind=$1 out=$2
shift 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$repo"
trap 'bench/scripts/ark-patch.sh --remove' EXIT

case $kind in
  mac)
    target=aarch64-apple-darwin build=(cargo build) features=(--features metal) flags= ;;
  x86)
    target=x86_64-unknown-linux-gnu build=(cargo zigbuild --target x86_64-unknown-linux-gnu)
    features=(--no-default-features --features cpu) flags="-C target-feature=+adx,+bmi2" ;;
  cuda)
    target=x86_64-unknown-linux-gnu build=(cargo zigbuild --target x86_64-unknown-linux-gnu)
    features=(--no-default-features --features cpu,cuda) flags="-C target-feature=+adx,+bmi2" ;;
  *) echo "KIND is mac, x86 or cuda" >&2; exit 2 ;;
esac

manifest="{\"kind\": \"$kind\", \"target\": \"$target\", \"rustflags\": \"$flags\", \"source\": \"$(git rev-parse HEAD)\", \"rustc\": \"$(rustc --version)\", \"variants\": {"
sep=
for spec in "$@"; do
  name=${spec%%=*} dir=${spec#*=}
  bench/scripts/ark-patch.sh "$dir"
  tdir=$repo/target/arkab/build-$kind-$name
  echo "==> $name: $dir"
  RUSTFLAGS="$flags" CARGO_TARGET_DIR=$tdir "${build[@]}" --release -p snarkrs "${features[@]}"
  RUSTFLAGS="$flags" CARGO_TARGET_DIR=$tdir "${build[@]}" --release -p snarkrs-groth16 \
    --example verify_time --example field_ops
  rel=$tdir/release
  [[ $kind == mac ]] || rel=$tdir/$target/release
  cp "$rel/snarkrs" "$out/snarkrs-$name"
  cp "$rel/examples/verify_time" "$out/verify_time-$name"
  cp "$rel/examples/field_ops" "$out/field_ops-$name"
  dups=$(cargo tree -d -e normal --target "$target" 2>/dev/null | grep -c '^ark-' || true)
  commit=$(git -C "$dir" rev-parse HEAD)
  manifest+="$sep\"$name\": {\"algebra\": \"$dir\", \"commit\": \"$commit\", \"duplicate_ark_crates\": $dups"
  for f in snarkrs verify_time field_ops; do
    manifest+=", \"$f\": \"$(shasum -a 256 "$out/$f-$name" | cut -d' ' -f1)\""
  done
  manifest+="}"
  sep=", "
done
echo "$manifest}}" | python3 -m json.tool > "$out/build.json"
cat "$out/build.json"
