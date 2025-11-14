from __future__ import annotations

from cracks_yolo.exceptions import CracksYoloError


def main() -> int:
    try:
        _main()
    except KeyboardInterrupt:
        return 130
    except CracksYoloError as e:
        return e.code
    except Exception:
        return 1
    return 0


def _main() -> None: ...
