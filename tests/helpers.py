from pathlib import Path

from PIL import Image


def make_image(path: Path, size=(200, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 160, 120)).save(path)


def make_yolo_dataset(root: Path, labels: dict, names=("a", "b"), size=(200, 100), yaml=True):
    """labels: stem -> text (None = no label file)."""
    for stem, text in labels.items():
        make_image(root / "images" / f"{stem}.jpg", size)
        if text is not None:
            (root / "labels").mkdir(parents=True, exist_ok=True)
            (root / "labels" / f"{stem}.txt").write_text(text)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    if yaml and names:
        body = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
        (root / "data.yaml").write_text(body)
    return root
