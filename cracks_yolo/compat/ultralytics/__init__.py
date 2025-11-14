from __future__ import annotations

import importlib.util

__ultralytics_available__ = importlib.util.find_spec("ultralytics") is not None

if __ultralytics_available__:
    from cracks_yolo.compat.ultralytics.conv import *  # noqa: F403
    from cracks_yolo.compat.ultralytics.patches import *  # noqa: F403
