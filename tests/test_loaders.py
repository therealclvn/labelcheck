import json
import sys

import pytest

from labelcheck import check_dataset, load_coco, load_yolo
from labelcheck.loaders import _parse_names_yaml, load_names, parse_yolo_label

from helpers import make_image, make_yolo_dataset


def issue_codes(ds):
    return sorted(i.code for i in check_dataset(ds).issues)


def test_yolo_box_and_polygon_parsing(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.5 0.1 0.3 0.5\n\n")
    anns, errs = parse_yolo_label(f, "a.txt")
    assert errs == []
    assert anns[0].box == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert anns[1].polygons == [[(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)]]
    assert anns[1].box == pytest.approx((0.1, 0.1, 0.5, 0.5))


@pytest.mark.parametrize(
    "line,fragment",
    [
        ("0 0.5 0.5 0.2", "got 4"),
        ("x 0.5 0.5 0.2 0.2", "Non-numeric"),
        ("0 nan 0.5 0.2 0.2", "NaN"),
        ("1.5 0.5 0.5 0.2 0.2", "not an integer"),
        ("0 0.1 0.1 0.2 0.2 0.3", "got 6"),
    ],
)
def test_yolo_bad_lines_are_reported_not_raised(tmp_path, line, fragment):
    f = tmp_path / "a.txt"
    f.write_text(line + "\n")
    anns, errs = parse_yolo_label(f, "a.txt")
    assert anns == []
    assert len(errs) == 1 and fragment in errs[0][1] and errs[0][0] == "a.txt:1"


def test_yolo_dataset_end_to_end(tmp_path):
    root = make_yolo_dataset(
        tmp_path,
        {
            "ok": "0 0.5 0.5 0.2 0.2\n",
            "nolabel": None,
            "bg": "",
            "bad": "0 0.5 0.5\n",
        },
    )
    (root / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    ds = load_yolo(root)
    assert ds.names == {0: "a", 1: "b"}
    assert [i.image for i in ds.items] == sorted(i.image for i in ds.items)
    assert issue_codes(ds) == ["empty_label", "empty_label", "missing_label_file", "orphan_label_file", "parse_error"]


def test_yolo_nested_and_split_layouts(tmp_path):
    make_image(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert len(load_yolo(tmp_path).items) == 1

    split = tmp_path / "export"
    for part in ("train", "valid"):
        make_image(split / part / "images" / "x.jpg")
        (split / part / "labels").mkdir(parents=True)
        (split / part / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert len(load_yolo(split).items) == 2


def test_yolo_unreadable_image(tmp_path):
    root = make_yolo_dataset(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n"})
    (root / "images" / "a.jpg").write_bytes(b"garbage")
    assert "unreadable_image" in issue_codes(load_yolo(root))


def test_yolo_layout_errors(tmp_path):
    with pytest.raises(ValueError):
        load_yolo(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_yolo(tmp_path)  # empty folder: no images/labels
    with pytest.raises(ValueError):
        load_yolo(tmp_path, images=tmp_path)  # only one of the two given


def test_names_formats(tmp_path):
    assert _parse_names_yaml("names: [cat, dog]\n") == {0: "cat", 1: "dog"}
    assert _parse_names_yaml("names:\n  - cat\n  - dog\n") == {0: "cat", 1: "dog"}
    assert _parse_names_yaml("names:\n  0: cat\n  2: dog\n") == {0: "cat", 2: "dog"}
    classes = tmp_path / "classes.txt"
    classes.write_text("cat\ndog\n\n")
    assert load_names(classes) == {0: "cat", 1: "dog"}
    assert load_names(None, tmp_path) == {0: "cat", 1: "dog"}  # auto-detected
    with pytest.raises(ValueError):
        load_names(tmp_path / "nope.yaml")


def test_names_fallback_parser_without_pyyaml(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)  # makes "import yaml" raise ImportError
    assert _parse_names_yaml("path: .\nnames: ['a b', c]\n") == {0: "a b", 1: "c"}
    assert _parse_names_yaml("path: .\nnames:\n  0: x\n  1: y\ntrain: t\n") == {0: "x", 1: "y"}
    assert _parse_names_yaml("names:\n  - x\n  - y\n") == {0: "x", 1: "y"}


def coco_file(tmp_path, images, annotations, categories=None):
    cats = categories if categories is not None else [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"images": images, "annotations": annotations, "categories": cats}))
    return p


def test_coco_end_to_end(tmp_path):
    make_image(tmp_path / "imgs" / "a.jpg", (200, 100))
    make_image(tmp_path / "imgs" / "b.jpg", (200, 100))
    images = [
        {"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
        {"id": 2, "file_name": "b.jpg", "width": 300, "height": 300},  # wrong size
        {"id": 3, "file_name": "gone.jpg", "width": 200, "height": 100},  # not on disk
    ]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
        {"id": 2, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20]},  # unknown category
        {"id": 3, "image_id": 9, "category_id": 1, "bbox": [1, 1, 5, 5]},  # unknown image
        {"id": 4, "image_id": 1, "category_id": 1},  # nothing to check
        {"id": 5, "image_id": 1, "category_id": 2, "bbox": [150, 10, 80, 50]},  # past the right edge
    ]
    ds = load_coco(coco_file(tmp_path, images, anns), images_dir=tmp_path / "imgs")
    assert issue_codes(ds) == [
        "dangling_annotation", "empty_label", "empty_label", "invalid_class",
        "missing_image", "out_of_bounds", "parse_error", "size_mismatch",
    ]


def test_coco_polygon_only_annotation_and_rle(tmp_path):
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[10, 10, 50, 10, 30, 40]]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10], "segmentation": {"counts": "x", "size": [100, 100]}},
    ]
    ds = load_coco(coco_file(tmp_path, images, anns))
    a, b = ds.items[0].annotations
    assert a.box == (10, 10, 50, 40) and len(a.polygons) == 1
    assert b.polygons == []
    assert issue_codes(ds) == []


def test_coco_bad_files(tmp_path):
    with pytest.raises(ValueError):
        load_coco(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        load_coco(bad)
    notcoco = tmp_path / "x.json"
    notcoco.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_coco(notcoco)
