"""Model registry: maps short names to ultralytics cfg + pretrained asset + SAC/TR indices."""

from __future__ import annotations

# (name, cfg_yaml, pretrained_asset, sac_indices, tr_indices, decode_format, use_rtdetr_class)
# decode_format: "anchor_free" (v8/v9/rtdetr) or "anchor_based" (v5/v6/v10)
# use_rtdetr_class: True for RT-DETR (uses RTDETR() not YOLO())
ULTRALYTICS_MODELS: list[tuple[str, str, str, tuple[int, ...], tuple[int, ...], str, bool]] = [
    # --- v3 ---
    ("yolov3", "yolov3.yaml", "yolov3u", (), (), "anchor_free", False),
    # v3 uses BottleneckCSP not C3/C2f — SAC insertion points TBD
    # --- v5 ---
    ("yolov5n", "yolov5n.yaml", "yolov5nu", (), (), "anchor_based", False),
    ("yolov5s", "yolov5s.yaml", "yolov5su", (), (), "anchor_based", False),
    ("yolov5s_sac", "yolov5s.yaml", "yolov5su", (2, 4, 6), (), "anchor_based", False),
    ("yolov5s_tr", "yolov5s.yaml", "yolov5su", (), (8,), "anchor_based", False),
    ("yolov5s_sactr", "yolov5s.yaml", "yolov5su", (2, 4, 6), (8,), "anchor_based", False),
    ("yolov5m", "yolov5m.yaml", "yolov5mu", (), (), "anchor_based", False),
    ("yolov5l", "yolov5l.yaml", "yolov5lu", (), (), "anchor_based", False),
    ("yolov5x", "yolov5x.yaml", "yolov5xu", (), (), "anchor_based", False),
    # --- v6 ---
    ("yolov6n", "yolov6.yaml", "yolov6n", (), (), "anchor_based", False),
    ("yolov6n_sac", "yolov6.yaml", "yolov6n", (2, 4, 6, 8), (), "anchor_based", False),
    # --- v8 ---
    ("yolov8n", "yolov8n.yaml", "yolov8n", (), (), "anchor_free", False),
    ("yolov8n_sac", "yolov8n.yaml", "yolov8n", (2, 4, 6, 8), (), "anchor_free", False),
    ("yolov8s", "yolov8s.yaml", "yolov8s", (), (), "anchor_free", False),
    ("yolov8s_sac", "yolov8s.yaml", "yolov8s", (2, 4, 6, 8), (), "anchor_free", False),
    ("yolov8m", "yolov8m.yaml", "yolov8m", (), (), "anchor_free", False),
    ("yolov8l", "yolov8l.yaml", "yolov8l", (), (), "anchor_free", False),
    ("yolov8x", "yolov8x.yaml", "yolov8x", (), (), "anchor_free", False),
    # --- v9 ---
    ("yolov9t", "yolov9t.yaml", "yolov9t", (), (), "anchor_free", False),
    ("yolov9s", "yolov9s.yaml", "yolov9s", (), (), "anchor_free", False),
    ("yolov9m", "yolov9m.yaml", "yolov9m", (), (), "anchor_free", False),
    ("yolov9c", "yolov9c.yaml", "yolov9c", (), (), "anchor_free", False),
    ("yolov9c_sac", "yolov9c.yaml", "yolov9c", (2, 4, 6, 8), (), "anchor_free", False),
    ("yolov9e", "yolov9e.yaml", "yolov9e", (), (), "anchor_free", False),
    # --- v10 ---
    ("yolov10n", "yolov10n.yaml", "yolov10n", (), (), "anchor_based", False),
    ("yolov10s", "yolov10s.yaml", "yolov10s", (), (), "anchor_based", False),
    ("yolov10s_sac", "yolov10s.yaml", "yolov10s", (2, 4, 6, 8), (), "anchor_based", False),
    ("yolov10m", "yolov10m.yaml", "yolov10m", (), (), "anchor_based", False),
    ("yolov10b", "yolov10b.yaml", "yolov10b", (), (), "anchor_based", False),
    ("yolov10l", "yolov10l.yaml", "yolov10l", (), (), "anchor_based", False),
    ("yolov10x", "yolov10x.yaml", "yolov10x", (), (), "anchor_based", False),
    # --- RT-DETR ---
    ("rtdetr_r50", "rtdetr-resnet50.yaml", "rtdetr-resnet50", (), (), "anchor_free", True),
    (
        "rtdetr_r50_sac",
        "rtdetr-resnet50.yaml",
        "rtdetr-resnet50",
        (2, 4, 6, 8),
        (),
        "anchor_free",
        True,
    ),
    # --- YOLO11/12/26 (baseline only) ---
    ("yolo11n", "yolo11n.yaml", "yolo11n", (), (), "anchor_free", False),
    ("yolo11s", "yolo11s.yaml", "yolo11s", (), (), "anchor_free", False),
    ("yolo12n", "yolo12n.yaml", "yolo12n", (), (), "anchor_free", False),
    ("yolo12s", "yolo12s.yaml", "yolo12s", (), (), "anchor_free", False),
    ("yolo26n", "yolo26n.yaml", "yolo26n", (), (), "anchor_free", False),
    ("yolo26s", "yolo26s.yaml", "yolo26s", (), (), "anchor_free", False),
]

# Build a lookup dict
MODEL_LOOKUP: dict[str, tuple[str, str, str, tuple[int, ...], tuple[int, ...], str, bool]] = {
    row[0]: row for row in ULTRALYTICS_MODELS
}
