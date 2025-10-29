from __future__ import annotations

import typing as t

import torch.nn
import ultralytics.nn.modules
import ultralytics.nn.tasks

C = t.TypeVar("C", bound=type[torch.nn.Module])


def register_to_ultralytics(cls: C) -> C:
    module_name = cls.__name__
    setattr(ultralytics.nn.modules, module_name, cls)  # register to ultralytics.nn.modules
    if module_name not in ultralytics.nn.modules.__all__:
        ultralytics.nn.modules.__all__ += (module_name,)

    setattr(ultralytics.nn.tasks, module_name, cls)  # register to ultralytics.nn.tasks

    return cls
