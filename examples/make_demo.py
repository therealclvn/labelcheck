"""Generate a small synthetic dataset (YOLO + COCO) with deliberate labelling mistakes.

    python examples/make_demo.py demo_data
    labelcheck yolo demo_data/yolo --html demo_yolo.html
    labelcheck coco demo_data/coco/annotations.json --images demo_data/coco/images --html demo_coco.html
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 640, 480
CLASSES = ["car", "person", "dog"]
COLORS = [(200, 70, 70), (70, 110, 200), (200, 170, 60)]


def draw_scene(rng, n_objects):
    img = Image.new("RGB", (W, H), (rng.randint(150, 200), rng.randint(170, 210), rng.randint(150, 200)))
    d = ImageDraw.Draw(img)
    boxes = []
    for _ in range(n_objects):
        cls = rng.randrange(len(CLASSES))
        bw, bh = rng.randint(70, 170), rng.randint(60, 150)
        x, y = rng.randint(10, W - bw - 10), rng.randint(10, H - bh - 10)
        d.rectangle([x, y, x + bw, y + bh], fill=COLORS[cls], outline=(20, 20, 20), width=2)
        boxes.append((cls, x, y, bw, bh))
    return img, boxes


def yolo_line(cls, x, y, bw, bh):
    return f"{cls} {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}"


def make_yolo(dst: Path):
    rng = random.Random(7)
    img_dir, lbl_dir = dst / "images" / "train", dst / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (dst / "data.yaml").write_text("path: .\ntrain: images/train\nnames:\n  0: car\n  1: person\n  2: dog\n")

    for i in range(18):
        img, boxes = draw_scene(rng, rng.randint(1, 4))
        img.save(img_dir / f"img_{i:03d}.jpg", quality=88)
        lines = [yolo_line(*b) for b in boxes]
        (lbl_dir / f"img_{i:03d}.txt").write_text("\n".join(lines) + "\n")

    def label(i):
        return lbl_dir / f"img_{i:03d}.txt"

    # --- injected mistakes -------------------------------------------------
    label(1).write_text(label(1).read_text() + "1 0.97 0.50 0.20 0.30\n")  # sticks out of the image
    label(2).write_text(label(2).read_text() + "0 0.40 0.40 0.000000 0.20\n")  # zero width
    label(3).write_text(label(3).read_text() + "2 0.60 0.60 0.002 0.003\n")  # tiny box
    first = label(4).read_text().splitlines()[0]
    label(4).write_text(label(4).read_text() + first + "\n")  # exact duplicate
    first = label(5).read_text().splitlines()[0].split()
    label(5).write_text(label(5).read_text() + " ".join(["1"] + first[1:]) + "\n" if first[0] != "1" else label(5).read_text() + " ".join(["2"] + first[1:]) + "\n")
    label(6).write_text(label(6).read_text() + "9 0.50 0.50 0.20 0.20\n")  # class id not in names
    label(7).unlink()  # image without a label file
    (lbl_dir / "img_099.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # label file without an image
    label(8).write_text("")  # empty label (background)
    label(9).write_text(label(9).read_text() + "0 0.5 0.5 0.2\nperson 0.5 0.5 0.2 0.2\n")  # malformed lines
    label(10).write_text(label(10).read_text() + "1 0.10 0.10 0.30 0.30 0.50 0.50\n")  # polygon on a straight line
    (img_dir / "img_011.jpg").write_bytes(b"this is not a jpeg")  # corrupt image


def make_coco(dst: Path):
    rng = random.Random(11)
    img_dir = dst / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
    }
    ann_id = 1
    for i in range(10):
        img, boxes = draw_scene(rng, rng.randint(1, 3))
        name = f"c_{i:03d}.jpg"
        img.save(img_dir / name, quality=88)
        data["images"].append({"id": i + 1, "file_name": name, "width": W, "height": H})
        for cls, x, y, bw, bh in boxes:
            poly = [x, y, x + bw, y, x + bw, y + bh, x, y + bh]
            data["annotations"].append(
                {"id": ann_id, "image_id": i + 1, "category_id": cls + 1, "bbox": [x, y, bw, bh],
                 "segmentation": [poly], "area": bw * bh, "iscrowd": 0}
            )
            ann_id += 1

    def add(**kw):
        nonlocal ann_id
        data["annotations"].append({"id": ann_id, "iscrowd": 0, **kw})
        ann_id += 1

    add(image_id=1, category_id=1, bbox=[600, 400, 120, 120])  # out of bounds
    add(image_id=2, category_id=2, bbox=[100, 100, 0, 50])  # zero width
    add(image_id=3, category_id=7, bbox=[100, 100, 60, 60])  # unknown category
    add(image_id=999, category_id=1, bbox=[10, 10, 50, 50])  # image does not exist
    add(image_id=4, category_id=1, bbox=[50, 50, 80, 80], segmentation=[[50, 50, 130, 130]])  # 2-point polygon
    data["images"][5]["width"], data["images"][5]["height"] = 800, 600  # declared size is wrong
    data["images"].append({"id": 11, "file_name": "not_on_disk.jpg", "width": W, "height": H})
    (dst / "annotations.json").write_text(json.dumps(data))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
    make_yolo(root / "yolo")
    make_coco(root / "coco")
    print(f"Demo datasets written to {root}/yolo and {root}/coco")


if __name__ == "__main__":
    main()
