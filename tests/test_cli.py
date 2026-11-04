import json

from labelcheck.cli import main

from helpers import make_yolo_dataset


def bad_dataset(tmp_path):
    return make_yolo_dataset(
        tmp_path / "ds",
        {"a": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", "b": "0 0.95 0.5 0.3 0.3\n"},
    )


def good_dataset(tmp_path):
    return make_yolo_dataset(tmp_path / "ok", {"a": "0 0.5 0.5 0.2 0.2\n"})


def test_exit_codes(tmp_path, capsys):
    assert main(["yolo", str(good_dataset(tmp_path))]) == 0
    assert main(["yolo", str(bad_dataset(tmp_path))]) == 1  # has an out-of-bounds error
    assert main(["yolo", str(bad_dataset(tmp_path)), "--fail-on", "never"]) == 0
    assert main(["yolo", str(tmp_path / "nowhere")]) == 2
    assert "labelcheck: error" in capsys.readouterr().err


def test_fail_on_warning(tmp_path):
    only_warning = make_yolo_dataset(tmp_path / "w", {"a": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n"})
    assert main(["yolo", str(only_warning)]) == 0
    assert main(["yolo", str(only_warning), "--fail-on", "warning"]) == 1


def test_reports_are_written(tmp_path, capsys):
    ds = bad_dataset(tmp_path)
    html_path, json_path = tmp_path / "r.html", tmp_path / "r.json"
    main(["yolo", str(ds), "--html", str(html_path), "--json", str(json_path)])
    page = html_path.read_text(encoding="utf-8")
    assert "Box outside image" in page and "Duplicate box" in page
    assert "data:image/jpeg;base64," in page  # thumbnails embedded
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"]["errors"] == 1
    assert {i["code"] for i in report["issues"]} == {"out_of_bounds", "duplicate_box"}


def test_clean_dataset_report_says_so(tmp_path):
    html_path = tmp_path / "r.html"
    main(["yolo", str(good_dataset(tmp_path)), "--html", str(html_path)])
    assert "No problems found" in html_path.read_text(encoding="utf-8")


def test_html_escapes_file_names(tmp_path):
    ds = make_yolo_dataset(tmp_path / "x", {"<script>alert(1)": "0 0.95 0.5 0.3 0.3\n"})
    html_path = tmp_path / "r.html"
    main(["yolo", str(ds), "--html", str(html_path)])
    page = html_path.read_text(encoding="utf-8")
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_threshold_options(tmp_path):
    ds = make_yolo_dataset(tmp_path / "t", {"a": "0 0.5 0.5 0.2 0.2\n0 0.52 0.5 0.2 0.2\n"})
    assert main(["yolo", str(ds), "--fail-on", "warning"]) == 0
    assert main(["yolo", str(ds), "--fail-on", "warning", "--dup-iou", "0.7"]) == 1
