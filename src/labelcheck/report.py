"""JSON and self-contained HTML reports (thumbnails are embedded as base64)."""
from __future__ import annotations

import base64
import html
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .checks import CODES, Result
from .models import ERROR, INFO, SEVERITY_ORDER, WARNING, Dataset, Issue, Item

HIT = (230, 50, 50)  # the offending annotation
OTHER = (70, 190, 120)  # all other annotations on the image


def to_json(ds: Dataset, result: Result) -> dict:
    return {
        "tool": "labelcheck",
        "version": __version__,
        "summary": result.stats,
        "issues": [i.as_dict() for i in result.issues],
    }


def write_json(path, ds: Dataset, result: Result) -> None:
    Path(path).write_text(json.dumps(to_json(ds, result), ensure_ascii=False, indent=2), encoding="utf-8")


def _thumb(item: Item, ann_index: Optional[int], max_side: int = 360) -> Optional[str]:
    """Base64 JPEG of the image with its annotations drawn, or None if unavailable."""
    if not item.image_path:
        return None
    try:
        from PIL import Image, ImageDraw

        img = Image.open(item.image_path).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    draw = ImageDraw.Draw(img)

    def px(x: float, y: float):
        return (x * w * scale, y * h * scale) if item.normalized else (x * scale, y * scale)

    order = [i for i in range(len(item.annotations)) if i != ann_index]
    if ann_index is not None and 0 <= ann_index < len(item.annotations):
        order.append(ann_index)  # draw the offender last so it stays on top
    for i in order:
        ann = item.annotations[i]
        hot = i == ann_index
        color, width = (HIT, 3) if hot else (OTHER, 1)
        if ann.box:
            x1, y1 = px(ann.box[0], ann.box[1])
            x2, y2 = px(ann.box[2], ann.box[3])
            draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=color, width=width)
        for poly in ann.polygons:
            pts = [px(x, y) for x, y in poly]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=color, width=width)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode("ascii")


CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--err:#c62828;--warn:#b26a00;--info:#2a6db0;--bar:#7aa0d4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161614;--fg:#ecebe6;--mut:#9a9992;--card:#20201d;--line:#34342f;--err:#ef7a7a;--warn:#e0a040;--info:#6aaae8;--bar:#4a6f9f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.15rem;margin:32px 0 4px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sub{color:var(--mut);margin:0 0 20px;word-break:break-all}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:1.5rem}.stat span{color:var(--mut);font-size:.85rem}
.err b{color:var(--err)}.warn b{color:var(--warn)}.info b{color:var(--info)}
.badge{font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:99px;border:1px solid currentColor}
.badge.error{color:var(--err)}.badge.warning{color:var(--warn)}.badge.info{color:var(--info)}
.desc{color:var(--mut);margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.shot{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.shot img{display:block;width:100%;height:auto}.shot div{padding:8px 10px;font-size:.85rem}
.shot code,td code{font-size:.78rem;color:var(--mut);word-break:break-all}
table{width:100%;border-collapse:collapse;font-size:.85rem;background:var(--card);border:1px solid var(--line);border-radius:10px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mut);font-weight:600}
details{margin-top:12px}summary{cursor:pointer;color:var(--mut)}
.bar{height:8px;background:var(--bar);border-radius:4px;min-width:2px}
.ok{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-top:24px}
footer{margin-top:40px;color:var(--mut);font-size:.8rem}
@media (max-width:520px){main{padding:16px 16px 40px}}
"""


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def render_html(ds: Dataset, result: Result, max_thumbs: int = 12, max_rows: int = 200) -> str:
    s = result.stats
    items: Dict[str, Item] = {it.image: it for it in ds.items}
    by_code: Dict[str, List[Issue]] = defaultdict(list)
    for issue in result.issues:
        by_code[issue.code].append(issue)

    out: List[str] = []
    out.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    out.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    out.append(f"<title>labelcheck report</title><style>{CSS}</style></head><body><main>")
    out.append("<h1>Label check report</h1>")
    out.append(f"<p class='sub'>{_e(s['kind'].upper())} dataset &middot; {_e(s['root'])}</p>")
    out.append("<div class='cards'>")
    for label, value, cls in (
        ("Images", s["images"], ""),
        ("Annotations", s["annotations"], ""),
        ("Errors", s["errors"], "err"),
        ("Warnings", s["warnings"], "warn"),
        ("Info", s["infos"], "info"),
        ("Images with problems", f"{s['images_with_problems']} / {s['images']}", ""),
    ):
        out.append(f"<div class='stat {cls}'><b>{_e(value)}</b><span>{_e(label)}</span></div>")
    out.append("</div>")

    if not result.issues:
        out.append("<div class='ok'><b>No problems found.</b> Every box, polygon and file passed all checks.</div>")

    codes = sorted(
        by_code,
        key=lambda c: (SEVERITY_ORDER[CODES.get(c, (WARNING,))[0]], -len(by_code[c])),
    )
    for code in codes:
        sev, title, desc = CODES.get(code, (WARNING, code, ""))
        issues = by_code[code]
        out.append(f"<h2>{_e(title)} <span class='badge {sev}'>{sev} &middot; {len(issues)}</span></h2>")
        out.append(f"<p class='desc'>{_e(desc)}</p>")

        # one thumbnail per distinct image, first N images
        shots, seen = [], set()
        for issue in issues:
            if len(shots) >= max_thumbs:
                break
            item = items.get(issue.image)
            if item is None or issue.image in seen:
                continue
            thumb = _thumb(item, issue.ann_index)
            if thumb:
                seen.add(issue.image)
                shots.append((issue, thumb))
        if shots:
            out.append("<div class='grid'>")
            for issue, thumb in shots:
                out.append(
                    "<div class='shot'>"
                    f"<img loading='lazy' alt='' src='data:image/jpeg;base64,{thumb}'>"
                    f"<div>{_e(issue.message)}<br><code>{_e(issue.source or issue.image)}</code></div></div>"
                )
            out.append("</div>")

        rows = issues[:max_rows]
        if len(issues) == 1:
            label = "Details"
        elif len(issues) <= max_rows:
            label = f"All {len(issues)} issues"
        else:
            label = f"First {max_rows} of {len(issues)} issues (full list in the JSON report)"
        out.append(f"<details{' open' if not shots else ''}><summary>{_e(label)}</summary><table>")
        out.append("<tr><th>Image / file</th><th>Where</th><th>Detail</th></tr>")
        for issue in rows:
            out.append(
                f"<tr><td><code>{_e(issue.image)}</code></td><td><code>{_e(issue.source)}</code></td>"
                f"<td>{_e(issue.message)}</td></tr>"
            )
        out.append("</table></details>")

    classes = s.get("classes") or {}
    if classes:
        top = max(c["count"] for c in classes.values()) or 1
        out.append("<h2>Class distribution</h2><table><tr><th>Class</th><th>Boxes</th><th></th></tr>")
        for cid, info in classes.items():
            name = f"{cid} ({_e(info['name'])})" if info.get("name") else str(cid)
            out.append(
                f"<tr><td>{name}</td><td>{info['count']}</td>"
                f"<td style='width:50%'><div class='bar' style='width:{100 * info['count'] / top:.1f}%'></div></td></tr>"
            )
        out.append("</table>")

    out.append(f"<footer>Generated by labelcheck {__version__}. Green boxes are other annotations on the image, red marks the flagged one.</footer>")
    out.append("</main></body></html>")
    return "".join(out)


def write_html(path, ds: Dataset, result: Result, max_thumbs: int = 12) -> None:
    Path(path).write_text(render_html(ds, result, max_thumbs=max_thumbs), encoding="utf-8")
