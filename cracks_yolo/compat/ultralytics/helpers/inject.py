from __future__ import annotations

import typing as t

import torch.nn
import ultralytics.nn.modules
import ultralytics.nn.tasks

C = t.TypeVar("C", bound=type[torch.nn.Module])


_REGISTERED = set()


def register_to_ultralytics(cls: C) -> C:
    module_name = cls.__name__
    if module_name in _REGISTERED:
        return cls  # already registered, skip

    _REGISTERED.add(module_name)
    setattr(ultralytics.nn.modules, module_name, cls)
    if module_name not in getattr(ultralytics.nn.modules, "__all__", ()):
        ultralytics.nn.modules.__all__ += (module_name,)
    setattr(ultralytics.nn.tasks, module_name, cls)
    return cls
