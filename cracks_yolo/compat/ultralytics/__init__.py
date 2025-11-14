from __future__ import annotations

try:
    import ultralytics
except ImportError:
    raise

from cracks_yolo.compat.ultralytics.conv import *  # noqa: F403
from cracks_yolo.compat.ultralytics.patches import *  # noqa: F403
