"""Read YOLO and COCO datasets into :class:`labelcheck.models.Dataset`.

Loaders never raise on bad data inside the files: problems become ``Issue``
objects so one broken label file cannot hide the rest of the report. They only
raise ``ValueError`` when the dataset layout itself cannot be understood.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    ERROR,
    WARNING,
    Annotation,
    Dataset,
    Issue,
    Item,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NON_LABEL_TXT = {"classes.txt", "labels.txt", "names.txt", "readme.txt", "notes.txt"}


def _open_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- names


def _parse_names_yaml(text: str) -> Dict[int, str]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        names = data.get("names")
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if names is None and isinstance(data.get("nc"), int):
            return {i: str(i) for i in range(data["nc"])}
        return {}
    except ImportError:
        pass
    # Tiny fallback parser for the three layouts Ultralytics / Roboflow export.
    m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.S | re.M)
    if m:
        items = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        return {i: n for i, n in enumerate(items)}
    names: Dict[int, str] = {}
    in_block = False
    seq = 0
    for line in text.splitlines():
        if re.match(r"^names\s*:\s*$", line):
            in_block = True
            continue
        if in_block:
            if line and not line.startswith((" ", "\t", "-")):
                break
            kv = re.match(r"^\s+(\d+)\s*:\s*(.+?)\s*$", line)
            li = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if kv:
                names[int(kv.group(1))] = kv.group(2).strip("'\"")
            elif li:
                names[seq] = li.group(1).strip("'\"")
                seq += 1
    return names


def load_names(path: Optional[Path], root: Optional[Path] = None) -> Dict[int, str]:
    """Class names from a data.yaml / classes.txt. Auto-detects inside ``root``."""
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    elif root:
        for name in ("data.yaml", "dataset.yaml", "data.yml", "classes.txt", "obj.names"):
            candidates.append(Path(root) / name)
    for c in candidates:
        if not c.is_file():
            if path:
                raise ValueError(f"Names file not found: {c}")
            continue
        text = c.read_text(encoding="utf-8-sig")
        if c.suffix.lower() in (".yaml", ".yml"):
            names = _parse_names_yaml(text)
        else:
            names = {i: ln.strip() for i, ln in enumerate(text.splitlines()) if ln.strip()}
        if names:
            return names
    return {}


# ---------------------------------------------------------------------------- YOLO


def parse_yolo_label(path: Path, shown: str) -> Tuple[List[Annotation], List[Tuple[str, str]]]:
    """Return (annotations, [(source, error message), ...])."""
    anns: List[Annotation] = []
    errs: List[Tuple[str, str]] = []
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return anns, [(shown, f"Cannot read label file: {exc}")]
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        src = f"{shown}:{n}"
        try:
            values = [float(x) for x in parts]
        except ValueError:
            errs.append((src, "Non-numeric value"))
            continue
        if not all(math.isfinite(v) for v in values):
            errs.append((src, "NaN or infinite value"))
            continue
        if values[0] != int(values[0]):
            errs.append((src, f"Class id is not an integer: {parts[0]}"))
            continue
        cls, nums = int(values[0]), values[1:]
        if len(nums) == 4:
            cx, cy, w, h = nums
            anns.append(Annotation(cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), [], src))
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            pts = list(zip(nums[0::2], nums[1::2]))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            anns.append(Annotation(cls, (min(xs), min(ys), max(xs), max(ys)), [pts], src))
        else:
            errs.append(
                (src, f"Expected 5 values (box) or class + at least 3 x,y pairs (polygon), got {len(parts)}")
            )
    return anns, errs


def _find_yolo_pairs(root: Path, images: Optional[Path], labels: Optional[Path]) -> List[Tuple[Path, Path]]:
    if images or labels:
        if not (images and labels):
            raise ValueError("Give both --images and --labels, or neither.")
        return [(Path(images), Path(labels))]
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return [(root / "images", root / "labels")]
    pairs = [
        (c / "images", c / "labels")
        for c in sorted(root.iterdir())
        if c.is_dir() and (c / "images").is_dir() and (c / "labels").is_dir()
    ]
    if not pairs:
        raise ValueError(
            f"Could not find images/ and labels/ folders under {root}. "
            "Pass --images and --labels explicitly."
        )
    return pairs


def load_yolo(
    root,
    images: Optional[Path] = None,
    labels: Optional[Path] = None,
    names_path: Optional[Path] = None,
) -> Dataset:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset folder not found: {root}")
    names = load_names(names_path, root)
    items: List[Item] = []
    load_issues: List[Issue] = []

    for img_dir, lbl_dir in _find_yolo_pairs(root, images, labels):
        if not img_dir.is_dir():
            raise ValueError(f"Images folder not found: {img_dir}")
        if not lbl_dir.is_dir():
            raise ValueError(f"Labels folder not found: {lbl_dir}")
        matched = set()
        for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            lbl = (lbl_dir / img.relative_to(img_dir)).with_suffix(".txt")
            shown = _display(img, root)
            item = Item(
                image=shown,
                image_path=str(img),
                normalized=True,
                label_file=_display(lbl, root),
                has_label_file=lbl.is_file(),
            )
            try:
                item.width, item.height = _open_size(img)
            except Exception as exc:  # corrupt / unsupported file
                load_issues.append(Issue("unreadable_image", ERROR, f"Cannot open image ({type(exc).__name__}): corrupt or unsupported file", shown))
            if lbl.is_file():
                matched.add(lbl.resolve())
                item.annotations, errs = parse_yolo_label(lbl, item.label_file)
                for src, msg in errs:
                    load_issues.append(Issue("parse_error", ERROR, msg, shown, source=src))
            items.append(item)
        for lbl in sorted(lbl_dir.rglob("*.txt")):
            if lbl.name.lower() in NON_LABEL_TXT or lbl.resolve() in matched:
                continue
            load_issues.append(
                Issue("orphan_label_file", WARNING, "Label file has no matching image", _display(lbl, root))
            )

    return Dataset(
        kind="yolo",
        root=str(root),
        items=items,
        names=names,
        valid_ids=set(names) if names else None,
        load_issues=load_issues,
    )


# ---------------------------------------------------------------------------- COCO


def _polygons_from_segmentation(seg) -> List[List[Tuple[float, float]]]:
    polys: List[List[Tuple[float, float]]] = []
    if isinstance(seg, list):
        for flat in seg:
            if isinstance(flat, list) and len(flat) >= 2 and len(flat) % 2 == 0:
                try:
                    polys.append([(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)])
                except (TypeError, ValueError):
                    pass
            elif isinstance(flat, list):
                # odd / too short list: keep as an (invalid) polygon so checks can flag it
                try:
                    vals = [float(v) for v in flat]
                    polys.append([(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)])
                except (TypeError, ValueError):
                    pass
    return polys


def load_coco(ann_path, images_dir=None) -> Dataset:
    ann_path = Path(ann_path)
    if not ann_path.is_file():
        raise ValueError(f"COCO annotation file not found: {ann_path}")
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON: {ann_path} ({exc})") from exc
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("This does not look like a COCO file (no 'images' list).")

    images_dir = Path(images_dir) if images_dir else None
    names = {c["id"]: str(c.get("name", c["id"])) for c in data.get("categories", []) if "id" in c}
    load_issues: List[Issue] = []
    by_id: Dict[object, Item] = {}
    items: List[Item] = []

    for im in data["images"]:
        file_name = str(im.get("file_name", f"image_id_{im.get('id')}"))
        item = Item(
            image=file_name,
            width=im.get("width"),
            height=im.get("height"),
            normalized=False,
            has_label_file=None,
        )
        if images_dir is not None:
            path = images_dir / file_name
            if not path.is_file():
                load_issues.append(Issue("missing_image", ERROR, "Image file not found on disk", file_name))
            else:
                item.image_path = str(path)
                try:
                    aw, ah = _open_size(path)
                    if item.width and item.height and (aw, ah) != (item.width, item.height):
                        load_issues.append(
                            Issue(
                                "size_mismatch",
                                ERROR,
                                f"JSON says {item.width}x{item.height}, file is {aw}x{ah}",
                                file_name,
                            )
                        )
                    item.width, item.height = item.width or aw, item.height or ah
                except Exception as exc:
                    load_issues.append(Issue("unreadable_image", ERROR, f"Cannot open image ({type(exc).__name__}): corrupt or unsupported file", file_name))
        by_id[im.get("id")] = item
        items.append(item)

    for ann in data.get("annotations", []):
        aid = ann.get("id")
        src = f"annotation id {aid}"
        item = by_id.get(ann.get("image_id"))
        if item is None:
            load_issues.append(
                Issue(
                    "dangling_annotation",
                    ERROR,
                    f"Annotation references unknown image_id {ann.get('image_id')}",
                    f"image_id {ann.get('image_id')}",
                    source=src,
                )
            )
            continue
        cls = ann.get("category_id")
        if not isinstance(cls, int):
            load_issues.append(Issue("parse_error", ERROR, "Missing or non-integer category_id", item.image, source=src))
            continue
        polys = _polygons_from_segmentation(ann.get("segmentation"))
        bbox = ann.get("bbox")
        box = None
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            x, y, w, h = bbox
            box = (float(x), float(y), float(x) + float(w), float(y) + float(h))
        elif polys:
            xs = [p[0] for poly in polys for p in poly]
            ys = [p[1] for poly in polys for p in poly]
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            load_issues.append(
                Issue("parse_error", ERROR, "Annotation has neither a valid bbox nor a polygon", item.image, source=src)
            )
            continue
        item.annotations.append(Annotation(cls, box, polys, src))

    return Dataset(
        kind="coco",
        root=str(ann_path.parent),
        items=items,
        names=names,
        valid_ids=set(names) if names else None,
        load_issues=load_issues,
    )
