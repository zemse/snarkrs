#!/usr/bin/env python3
"""Compare a generated proof's public signals with the fixture reference."""
import json
from pathlib import Path
import sys


def public_values(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list):
        raise ValueError("public signals must be a JSON array")
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in values):
        raise ValueError("public signals must contain integer values")
    return [int(value) for value in values]


def matches_reference(vkey, public):
    reference = Path(vkey).with_name("public.json")
    return public_values(reference) == public_values(public)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_reference_public.py VKEY GENERATED_PUBLIC")
    try:
        if not matches_reference(sys.argv[1], sys.argv[2]):
            raise ValueError("public signals differ from the fixture reference")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
