"""Plain data containers shared by loaders, checks and reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 (native units of the item)
Point = Tuple[float, float]

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass
class Annotation:
    cls: int
    box: Optional[Box]
    polygons: List[List[Point]] = field(default_factory=list)
    source: str = ""  # "labels/a.txt:3" or "coco annotation id 17"


@dataclass
class Item:
    """One image together with its annotations.

    Coordinates are stored in the format's native units: YOLO is normalised
    (0..1), COCO is pixels. ``normalized`` says which one applies.
    """

    image: str  # path shown to the user (relative when possible)
    image_path: Optional[str] = None  # absolute path used to open the file
    width: Optional[int] = None
    height: Optional[int] = None
    normalized: bool = False
    annotations: List[Annotation] = field(default_factory=list)
    label_file: Optional[str] = None
    has_label_file: Optional[bool] = None  # None = not applicable (COCO)

    def to_pixels(self, box: Box) -> Optional[Box]:
        if not self.normalized:
            return box
        if not self.width or not self.height:
            return None
        x1, y1, x2, y2 = box
        return (x1 * self.width, y1 * self.height, x2 * self.width, y2 * self.height)


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    image: str
    source: str = ""
    ann_index: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "image": self.image,
            "source": self.source,
            "ann_index": self.ann_index,
        }


@dataclass
class Dataset:
    kind: str  # "yolo" | "coco"
    root: str
    items: List[Item]
    names: dict  # class id -> name (may be empty)
    valid_ids: Optional[set] = None  # None = unknown, skip class-id validation
    load_issues: List[Issue] = field(default_factory=list)
