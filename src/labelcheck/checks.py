"""The actual label audit. Pure functions over :class:`labelcheck.models.Dataset`."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import (
    ERROR,
    INFO,
    SEVERITY_ORDER,
    WARNING,
    Box,
    Dataset,
    Issue,
    Item,
)

# code -> (severity, short title, what it means / why it matters)
CODES: Dict[str, tuple] = {
    "unreadable_image": (ERROR, "Unreadable image", "The image file is corrupt or not a supported format, so training will crash or skip it."),
    "missing_image": (ERROR, "Missing image file", "The annotation file lists an image that is not on disk."),
    "size_mismatch": (ERROR, "Image size mismatch", "Width/height in the JSON differ from the real image, so every box on it is misplaced."),
    "parse_error": (ERROR, "Malformed annotation", "The line or record cannot be parsed; most trainers silently drop or crash on it."),
    "dangling_annotation": (ERROR, "Annotation without image", "An annotation points to an image id that does not exist."),
    "invalid_class": (ERROR, "Invalid class id", "The class id is negative or not defined in the class list."),
    "out_of_bounds": (ERROR, "Box outside image", "Part of the box or polygon lies outside the image."),
    "zero_size": (ERROR, "Zero or negative size", "Width or height is not positive, so the box has no area."),
    "degenerate_polygon": (ERROR, "Degenerate polygon", "Fewer than 3 distinct points or no area."),
    "missing_label_file": (WARNING, "Image without label file", "No label file exists. Fine for background images, a bug if the file was lost."),
    "orphan_label_file": (WARNING, "Label file without image", "The label file has no matching image, usually after renaming or deleting images."),
    "tiny_box": (WARNING, "Very small box", "The shorter side is only a couple of pixels; often a mis-click."),
    "duplicate_box": (WARNING, "Duplicate box", "Two boxes of the same class overlap almost exactly."),
    "conflicting_class": (WARNING, "Same box, different class", "Two boxes overlap almost exactly but have different classes."),
    "empty_label": (INFO, "Empty annotation", "The image has no objects. Intended for background images, otherwise unlabelled data."),
}


@dataclass
class Config:
    min_side: float = 2.0  # pixels; shorter side below this is a tiny_box
    dup_iou: float = 0.95  # IoU at or above this counts as the same box
    tol_norm: float = 1e-3  # out-of-bounds tolerance for normalised coordinates
    tol_px: float = 1.0  # out-of-bounds tolerance for pixel coordinates
    max_boxes_pairwise: int = 1500  # skip duplicate search on images with more boxes


@dataclass
class Result:
    issues: List[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def count(self, severity: str) -> int:
        return sum(1 for i in self.issues if i.severity == severity)


def iou(a: Box, b: Box) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _polygon_area(points) -> float:
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _name(ds: Dataset, cls: int) -> str:
    return f"{cls} ({ds.names[cls]})" if cls in ds.names else str(cls)


def _bounds(item: Item, cfg: Config):
    """(max_x, max_y, tolerance) in the item's native units, or None if unknown."""
    if item.normalized:
        return 1.0, 1.0, cfg.tol_norm
    if item.width and item.height:
        return float(item.width), float(item.height), cfg.tol_px
    return None


def _check_item(ds: Dataset, item: Item, cfg: Config) -> List[Issue]:
    out: List[Issue] = []
    shown = item.image

    def add(code: str, msg: str, idx: Optional[int] = None, source: str = ""):
        out.append(Issue(code, CODES[code][0], msg, shown, source=source, ann_index=idx))

    if item.has_label_file is False:
        add("missing_label_file", "No label file for this image", source=item.label_file or "")
    elif not item.annotations and item.has_label_file is not None:
        add("empty_label", "Label file exists but contains no objects", source=item.label_file or "")
    elif not item.annotations and item.has_label_file is None:
        add("empty_label", "Image has no annotations")

    bounds = _bounds(item, cfg)
    valid_boxes = []  # (index, box) with positive area, for the duplicate search

    for idx, ann in enumerate(item.annotations):
        src = ann.source

        if ann.cls < 0 or (ds.valid_ids is not None and ann.cls not in ds.valid_ids):
            hint = ""
            if ds.valid_ids:
                hint = f" (defined: {min(ds.valid_ids)}..{max(ds.valid_ids)})"
            add("invalid_class", f"Class id {ann.cls} is not defined{hint}", idx, src)

        if ann.box is not None:
            x1, y1, x2, y2 = ann.box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                add("zero_size", f"Box has non-positive size (w={w:.4g}, h={h:.4g})", idx, src)
            else:
                valid_boxes.append((idx, ann.box))
                px = item.to_pixels(ann.box)
                if px is not None:
                    pw, ph = px[2] - px[0], px[3] - px[1]
                    if min(pw, ph) < cfg.min_side:
                        add("tiny_box", f"Box is only {pw:.1f}x{ph:.1f} px", idx, src)

            if bounds:
                mx, my, tol = bounds
                over = max(-x1, -y1, x2 - mx, y2 - my)
                if over > tol:
                    unit = "of image size" if item.normalized else "px"
                    amount = f"{over:.3f}" if item.normalized else f"{over:.1f}"
                    add("out_of_bounds", f"Box extends {amount} {unit} outside the image", idx, src)

        for poly in ann.polygons:
            if len({(round(x, 9), round(y, 9)) for x, y in poly}) < 3:
                add("degenerate_polygon", f"Polygon has fewer than 3 distinct points ({len(poly)} given)", idx, src)
                continue
            area = _polygon_area(poly)
            if item.normalized and item.width and item.height:
                area *= item.width * item.height
                too_small = area < 1.0
            elif item.normalized:
                too_small = area < 1e-9
            else:
                too_small = area < 1.0
            if too_small:
                add("degenerate_polygon", "Polygon has (almost) zero area", idx, src)
            elif bounds:
                mx, my, tol = bounds
                if any(x < -tol or y < -tol or x > mx + tol or y > my + tol for x, y in poly):
                    add("out_of_bounds", "Polygon has points outside the image", idx, src)

    if len(valid_boxes) <= cfg.max_boxes_pairwise:
        for a in range(len(valid_boxes)):
            for b in range(a + 1, len(valid_boxes)):
                (i, bi), (j, bj) = valid_boxes[a], valid_boxes[b]
                if iou(bi, bj) < cfg.dup_iou:
                    continue
                ci, cj = item.annotations[i].cls, item.annotations[j].cls
                score = iou(bi, bj)
                if ci == cj:
                    add("duplicate_box", f"Boxes #{i + 1} and #{j + 1} overlap (IoU {score:.2f}), class {_name(ds, ci)}", j, item.annotations[j].source)
                else:
                    add(
                        "conflicting_class",
                        f"Boxes #{i + 1} and #{j + 1} overlap (IoU {score:.2f}) but are class {_name(ds, ci)} vs {_name(ds, cj)}",
                        j,
                        item.annotations[j].source,
                    )
    return out


def check_dataset(ds: Dataset, cfg: Optional[Config] = None) -> Result:
    cfg = cfg or Config()
    issues: List[Issue] = []
    for li in ds.load_issues:
        # loaders create issues with their own severity; keep it consistent with CODES
        li.severity = CODES.get(li.code, (li.severity,))[0]
        issues.append(li)
    for item in ds.items:
        issues.extend(_check_item(ds, item, cfg))

    issues.sort(key=lambda i: (SEVERITY_ORDER[i.severity], i.code, i.image, i.source))

    class_counts: Counter = Counter()
    n_ann = 0
    for item in ds.items:
        for a in item.annotations:
            class_counts[a.cls] += 1
            n_ann += 1
    by_code = Counter(i.code for i in issues)
    flagged = {i.image for i in issues if i.severity in (ERROR, WARNING)}
    stats = {
        "kind": ds.kind,
        "root": ds.root,
        "images": len(ds.items),
        "annotations": n_ann,
        "images_with_problems": len(flagged & {it.image for it in ds.items}),
        "errors": sum(1 for i in issues if i.severity == ERROR),
        "warnings": sum(1 for i in issues if i.severity == WARNING),
        "infos": sum(1 for i in issues if i.severity == INFO),
        "by_code": dict(by_code),
        "classes": {int(k): {"name": ds.names.get(k), "count": v} for k, v in sorted(class_counts.items())},
        "orphan_files": by_code.get("orphan_label_file", 0),
    }
    return Result(issues=issues, stats=stats)
