from labelcheck import Config, check_dataset, iou
from labelcheck.models import Annotation, Dataset, Item


def ds(items, names=None, valid=None):
    names = names or {}
    return Dataset("yolo", "root", items, names, valid_ids=valid)


def item(anns, normalized=True, w=200, h=100, has_label=True):
    return Item("img.jpg", None, w, h, normalized, anns, "img.txt", has_label)


def box(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def codes(result):
    return sorted(i.code for i in result.issues)


def test_iou_basic():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert abs(iou((0, 0, 2, 2), (1, 0, 3, 2)) - 1 / 3) < 1e-9


def test_clean_item_has_no_issues():
    r = check_dataset(ds([item([Annotation(0, box(0.5, 0.5, 0.2, 0.2))])], valid={0}))
    assert r.issues == []
    assert r.stats["annotations"] == 1


def test_out_of_bounds_normalized_and_tolerance():
    far = Annotation(0, box(0.95, 0.5, 0.2, 0.2))  # right edge at 1.05
    edge = Annotation(0, box(0.9, 0.5, 0.2, 0.2))  # right edge at 1.0: fine
    assert codes(check_dataset(ds([item([far])]))) == ["out_of_bounds"]
    assert codes(check_dataset(ds([item([edge])]))) == []
    nearly = Annotation(0, (0.5, 0.5, 1.0005, 0.6))  # within 1e-3 rounding tolerance
    assert codes(check_dataset(ds([item([nearly])]))) == []


def test_out_of_bounds_pixels_coco_style():
    it = item([Annotation(0, (150, 10, 230, 60))], normalized=False)
    assert codes(check_dataset(ds([it]))) == ["out_of_bounds"]
    unknown_size = item([Annotation(0, (150, 10, 230, 60))], normalized=False, w=None, h=None)
    assert codes(check_dataset(ds([unknown_size]))) == []


def test_zero_and_negative_size():
    zero = Annotation(0, box(0.5, 0.5, 0.0, 0.2))
    neg = Annotation(0, (0.6, 0.2, 0.5, 0.4))
    assert codes(check_dataset(ds([item([zero])]))) == ["zero_size"]
    assert codes(check_dataset(ds([item([neg])]))) == ["zero_size"]


def test_tiny_box_uses_pixels_and_config():
    a = Annotation(0, box(0.5, 0.5, 0.005, 0.5))  # 1 px wide on a 200 px image
    assert codes(check_dataset(ds([item([a])]))) == ["tiny_box"]
    assert codes(check_dataset(ds([item([a])]), Config(min_side=0.5))) == []


def test_duplicate_and_conflicting():
    a = Annotation(0, box(0.5, 0.5, 0.2, 0.2), source="l:1")
    b = Annotation(0, box(0.5, 0.5, 0.2, 0.2), source="l:2")
    c = Annotation(1, box(0.5, 0.5, 0.2, 0.2), source="l:3")
    assert codes(check_dataset(ds([item([a, b])]))) == ["duplicate_box"]
    assert codes(check_dataset(ds([item([a, c])]))) == ["conflicting_class"]
    shifted = Annotation(0, box(0.7, 0.5, 0.2, 0.2))
    assert codes(check_dataset(ds([item([a, shifted])]))) == []


def test_invalid_class():
    a = Annotation(5, box(0.5, 0.5, 0.2, 0.2))
    neg = Annotation(-1, box(0.5, 0.5, 0.2, 0.2))
    assert codes(check_dataset(ds([item([a])], valid={0, 1}))) == ["invalid_class"]
    assert codes(check_dataset(ds([item([neg])]))) == ["invalid_class"]  # negative is wrong even without a name list
    assert codes(check_dataset(ds([item([a])]))) == []  # unknown class list: cannot judge


def test_degenerate_polygons():
    line = Annotation(0, (0.1, 0.1, 0.5, 0.5), [[(0.1, 0.1), (0.3, 0.3), (0.5, 0.5)]])
    two = Annotation(0, (0.1, 0.1, 0.5, 0.5), [[(0.1, 0.1), (0.5, 0.5)]])
    ok = Annotation(0, (0.1, 0.1, 0.5, 0.5), [[(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)]])
    assert codes(check_dataset(ds([item([line])]))) == ["degenerate_polygon"]
    assert codes(check_dataset(ds([item([two])]))) == ["degenerate_polygon"]
    assert codes(check_dataset(ds([item([ok])]))) == []


def test_polygon_point_outside_image():
    p = Annotation(0, (0.1, 0.1, 0.9, 0.9), [[(0.1, 0.1), (1.2, 0.1), (0.5, 0.9)]])
    assert "out_of_bounds" in codes(check_dataset(ds([item([p])])))


def test_label_file_states():
    missing = item([], has_label=False)
    empty = item([], has_label=True)
    coco_empty = item([], has_label=None)
    assert codes(check_dataset(ds([missing]))) == ["missing_label_file"]
    assert codes(check_dataset(ds([empty]))) == ["empty_label"]
    assert codes(check_dataset(ds([coco_empty]))) == ["empty_label"]


def test_stats_and_ordering():
    items = [
        item([Annotation(0, box(0.5, 0.5, 0.2, 0.2)), Annotation(1, box(0.2, 0.2, 0.1, 0.1))]),
        item([], has_label=True),
    ]
    items[1].image = "empty.jpg"
    r = check_dataset(ds(items, names={0: "a", 1: "b"}, valid={0, 1}))
    assert r.stats["images"] == 2
    assert r.stats["classes"][1] == {"name": "b", "count": 1}
    assert r.stats["infos"] == 1 and r.stats["errors"] == 0
    assert r.stats["images_with_problems"] == 0  # info does not count as a problem
