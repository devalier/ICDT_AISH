"""``python3 -m harness`` — the Working Group's command line.

Commands:
  validate [--strict]   check every pack file; --strict also fails on warnings
  show                  print what is currently defined, for review
"""

from __future__ import annotations

import argparse
import sys

from .loader import load, packs_dir
from .validate import run


def _cmd_validate(args: argparse.Namespace) -> int:
    problems, ok = run(strict=args.strict)
    for problem in problems:
        print(problem, file=sys.stderr if problem.severity == "error" else sys.stdout)

    errors = sum(1 for p in problems if p.severity == "error")
    warnings = len(problems) - errors
    if ok:
        print(f"OK — packs valid ({warnings} warning(s)).")
        return 0
    print(f"FAILED — {errors} error(s), {warnings} warning(s).", file=sys.stderr)
    return 1


def _cmd_show(_args: argparse.Namespace) -> int:
    packs = load()
    print(f"packs: {packs_dir()}\n")

    print("Models")
    for model in packs.models.get("models") or []:
        print(
            f"  {model.get('id'):<20} {model.get('deployment'):<8} "
            f"egress={model.get('egress_class')}"
        )

    print("\nSuites")
    for path, suite in packs.suites.items():
        restricted = " [restricted]" if suite.get("restricted") else ""
        print(
            f"  {suite.get('id'):<20} {suite.get('family'):<11} v{suite.get('version')}"
            f"  scorer={suite.get('scorer')}  criteria={suite.get('criteria')}{restricted}"
        )
        print(f"    {path}  basis: {suite.get('basis')}")

    print("\nBias attributes")
    for attribute in packs.attributes.get("attributes") or []:
        marker = "*" if attribute.get("eu_specific") else " "
        print(
            f"  {marker} {attribute.get('id'):<22} "
            f"{len(attribute.get('variants') or [])} variants  "
            f"basis: {', '.join(attribute.get('basis') or [])}"
        )
    print("  (* EU-specific axis not covered by generic benchmarks)")

    print("\nCriteria")
    for name, body in (packs.criteria.get("criteria") or {}).items():
        metrics = ", ".join((body.get("metrics") or {}).keys())
        print(f"  {name:<22} {metrics}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate", help="check the pack files")
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors (use for the scored-run gate)",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    show_parser = sub.add_parser("show", help="print the current definitions")
    show_parser.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
