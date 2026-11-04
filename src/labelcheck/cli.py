"""Command line interface: ``labelcheck yolo ...`` and ``labelcheck coco ...``."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from . import __version__
from .checks import CODES, Config, check_dataset
from .loaders import load_coco, load_yolo
from .report import write_html, write_json


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--html", metavar="FILE", help="write a self-contained HTML report with thumbnails")
    p.add_argument("--json", metavar="FILE", help="write the full machine-readable report")
    p.add_argument("--min-side", type=float, default=2.0, help="boxes with a shorter side (px) below this are flagged (default 2)")
    p.add_argument("--dup-iou", type=float, default=0.95, help="IoU at or above this counts as a duplicate (default 0.95)")
    p.add_argument("--max-thumbs", type=int, default=12, help="thumbnails per problem type in the HTML report (default 12)")
    p.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default="error",
        help="exit with status 1 when problems of this severity or worse exist (default: error)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="only print the totals")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="labelcheck",
        description="Audit object-detection / segmentation labels and find the mistakes before training does.",
    )
    p.add_argument("--version", action="version", version=f"labelcheck {__version__}")
    sub = p.add_subparsers(dest="format", required=True)

    y = sub.add_parser("yolo", help="YOLO txt labels (boxes or polygons)")
    y.add_argument("root", help="dataset folder containing images/ and labels/ (or split folders that do)")
    y.add_argument("--images", help="images folder, if the layout is not the standard one")
    y.add_argument("--labels", help="labels folder, if the layout is not the standard one")
    y.add_argument("--names", help="data.yaml or classes.txt (auto-detected in the dataset folder)")
    _common(y)

    c = sub.add_parser("coco", help="COCO json (bbox and/or segmentation)")
    c.add_argument("annotations", help="path to the COCO annotations .json file")
    c.add_argument("--images", help="folder with the image files (enables file existence, size and thumbnails)")
    _common(c)
    return p


def _print_summary(result, quiet: bool) -> None:
    s = result.stats
    print(f"{s['kind'].upper()} dataset: {s['images']} images, {s['annotations']} annotations")
    print(
        f"  {s['errors']} errors, {s['warnings']} warnings, {s['infos']} info "
        f"({s['images_with_problems']} images with errors or warnings)"
    )
    if quiet or not result.issues:
        return
    groups = defaultdict(list)
    for issue in result.issues:
        groups[issue.code].append(issue)
    for code, issues in sorted(groups.items(), key=lambda kv: (CODES[kv[0]][0] != "error", CODES[kv[0]][0] != "warning", -len(kv[1]))):
        sev, title, _ = CODES[code]
        print(f"\n[{sev.upper():7}] {title}: {len(issues)}")
        for issue in issues[:3]:
            where = issue.source or issue.image
            print(f"    {where}  {issue.message}")
        if len(issues) > 3:
            print(f"    ... and {len(issues) - 3} more")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.format == "yolo":
            ds = load_yolo(
                args.root,
                images=Path(args.images) if args.images else None,
                labels=Path(args.labels) if args.labels else None,
                names_path=Path(args.names) if args.names else None,
            )
        else:
            ds = load_coco(args.annotations, images_dir=args.images)
    except ValueError as exc:
        print(f"labelcheck: error: {exc}", file=sys.stderr)
        return 2

    result = check_dataset(ds, Config(min_side=args.min_side, dup_iou=args.dup_iou))
    _print_summary(result, args.quiet)

    if args.json:
        write_json(args.json, ds, result)
        print(f"\nJSON report: {args.json}")
    if args.html:
        write_html(args.html, ds, result, max_thumbs=args.max_thumbs)
        print(f"HTML report: {args.html}")

    s = result.stats
    if args.fail_on == "error" and s["errors"]:
        return 1
    if args.fail_on == "warning" and (s["errors"] or s["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
