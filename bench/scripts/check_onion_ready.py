#!/usr/bin/env python3
"""Reject incomplete or changed local onion fixtures before renting a GPU."""
import argparse
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("onion_generator", Path(__file__).with_name("gen-onion-artifacts.py"))
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def check_ready(directory):
    directory = Path(directory)
    state = json.loads((directory / "metadata.json").read_text())
    shipping = json.loads((directory / "ship-metadata.json").read_text())
    if state.get("ready") is not True or shipping.get("ready") is not True:
        raise ValueError(f"{directory.name}: local verification is not complete")
    power = state["config"]["requested_power"]
    if state.get("domain_power") != power or state.get("domain") != 1 << power:
        raise ValueError(f"{directory.name}: compiled domain does not match requested power")
    if shipping.get("domain") != state["domain"]:
        raise ValueError(f"{directory.name}: shipping domain does not match")
    for stage in generator.STAGES:
        if not generator.complete(state, stage, directory):
            raise ValueError(f"{directory.name}: missing or changed {stage} outputs")
    outputs = state.get("shipping_outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(generator.SHIP_FILES):
        raise ValueError(f"{directory.name}: missing shipping content digests; revalidation required")
    for name, info in outputs.items():
        if generator.output_fingerprint(directory / name) != info:
            raise ValueError(f"{directory.name}: changed shipping output {name}")
    return state


def write_bundle_manifest(directory, destination):
    directory = Path(directory)
    state = check_ready(directory)
    lines = [f"{state['shipping_outputs'][name]['sha256']}  {name}\n"
             for name in generator.SHIP_FILES]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text("".join(lines))
    tmp.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        state = check_ready(args.directory)
        if args.manifest:
            write_bundle_manifest(args.directory, args.manifest)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
    print(f"ready: {args.directory} (domain 2^{state['domain_power']})")
