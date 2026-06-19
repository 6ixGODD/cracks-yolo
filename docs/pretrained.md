# Pretrained weights (`cracks_yolo.weights`)

## Design

`load_pretrained(model, spec, weights_dir, strict=False) -> LoadReport`
fetches a `.pt` checkpoint, remaps state_dict keys, and loads with
`strict=False`. Returns a `LoadReport` describing what matched, what was
missing, and what was unexpected.

The `DetectorModel.from_pretrained` classmethod on each zoo class
delegates to `load_pretrained`. Falls back to random init (with a warning)
if the download fails.

## `PretrainedSpec`

```python
@dataclass(frozen=True)
class PretrainedSpec:
    key: str                       # cache filename: weights/{key}.pt
    url: str                       # HTTPS URL to official .pt release
    state_dict_key_map: dict[str, str]  # prefix-remap table
```

Each zoo class declares `pretrained_spec: PretrainedSpec | None` as a
class attribute. Baseline variants (no SAC/TR) declare a spec; SAC/TR
variants declare `None` (no COCO weights exist for the SAC/TR layers).

## `LoadReport`

```python
@dataclass
class LoadReport:
    model: nn.Module
    matched: list[str]      # keys successfully loaded
    missing: list[str]      # keys in model, absent from checkpoint (e.g. SAC/TR layers)
    unexpected: list[str]   # keys in checkpoint, absent from model
    key: str
    url: str
    cached: bool            # True if file was already on disk
```

## Caching

- Checkpoint path: `{weights_dir}/{key}.pt` (default `weights/{key}.pt`).
- If the file exists, it's loaded directly — no network access.
- If missing, `_download(url, dest)` streams via `requests.get(stream=True)`
  with 60s timeout, writing in 64KiB chunks. Progress logged via loguru.

## Checkpoint format handling

`_load_state_dict(path)` handles three checkpoint layouts:

1. **Raw state_dict** (`{key: Tensor}`) — returned as-is.
2. **`{"state_dict": {...}, ...}`** — the `"state_dict"` value is extracted
   (common in PyTorch Lightning / mmdet checkpoints).
3. **`{"model": nn.Module, ...}`** — Ultralytics-style. `model.state_dict()`
   is called to extract the state_dict.

If the checkpoint is not a dict (or is a dict with none of these keys), a
`TypeError` is raised.

## Key remapping

`_remap_keys(state_dict, key_map)` rewrites state_dict key prefixes per
the spec. Used to align official release keys with `cracks_yolo`'s class
layout. Empty `key_map` means no remap needed.

Example: YOLOv5s official release uses `model.0.weight` (the Ultralytics
container's index); `cracks_yolo` stores the backbone as `backbone.0.weight`.
`state_dict_key_map={"model.": "backbone."}` rewrites the prefixes.

## SAC/TR layers: partial-load semantics

SAC (`SAConv2d`, `BottleneckSAC`, `C3SAC`, `C2fSAC`) and TR
(`TransformerBlock`, `C3TR`) layers are **not** in the COCO pretrained
weights — they're enhancements we add. With `strict=False` (the default):

- SAC/TR keys appear in `LoadReport.missing` (model expects them,
  checkpoint doesn't have them → random init).
- The rest of the backbone/neck/head loads cleanly from COCO.

With `strict=True`, missing keys raise `RuntimeError`. Useful for
catching accidental drift from upstream naming in tests.

## `from_pretrained` on each zoo class

```python
@classmethod
def from_pretrained(cls, num_classes, weights_dir=None, strict=False) -> _YOLOv8sBase:
    spec = cls.pretrained_spec
    if spec is None:
        return cls(num_classes=num_classes)  # SAC/TR variant — no COCO weights
    report = load_pretrained(
        model=cls(num_classes=num_classes),
        spec=spec,
        weights_dir=weights_dir,
        strict=strict,
    )
    return report.model
```

## URL registry

`cracks_yolo/weights/registry.py`:

```python
PRETRAINED_URLS: dict[str, str] = {
    "yolov5s":  "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov7w":  "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
    "yolov8s":  "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov10s": "https://huggingface.co/jameslahm/yolov10s/resolve/main/yolov10s.pt",
}
```

(Exact URLs may be adjusted to match the most recent stable release.)

## Adding a new pretrained source

1. Add the URL to `PRETRAINED_URLS` in `cracks_yolo/weights/registry.py`.
2. On the model class, set `pretrained_spec = PretrainedSpec(key="<short-name>", url=PRETRAINED_URLS["<short-name>"], state_dict_key_map={...})`.
3. If the checkpoint key layout differs from `cracks_yolo`'s, populate
   `state_dict_key_map` to align them.
4. Add a test in `tests/weights/test_loader.py` (mocked `requests.get`)
   that loads the new spec into a `_DummyModel` and verifies the
   `LoadReport`.

## Tests

`tests/weights/test_loader.py` uses a `_FakeResponse` context manager +
`patch("cracks_yolo.weights.loader.requests.get", return_value=fake_resp)`
to test the download path without network access. A `_DummyModel` with
two linear layers exercises the partial-load reporting (cache hit,
download, strict=False missing keys, strict=True raises, unexpected keys).
