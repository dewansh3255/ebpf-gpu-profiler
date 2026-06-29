"""Shared model factory for the federated-learning phase.

Keeping a single definition guarantees the server and every client build a
byte-compatible architecture (state_dict keys must match for FedAvg).

Architecture choice mirrors the Phase-1 rationale:
  - resnet18: light kernels -> exposes kernel-launch-bound CPU starvation.
  - resnet50: ~97 MB of parameters per update -> far larger gradient payloads
    over the wire, exposing communication-bound network synchronisation cost.
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet50

_FACTORIES = {"resnet18": resnet18, "resnet50": resnet50}


def build_model(arch="resnet18", num_classes=10):
    arch = (arch or "resnet18").lower()
    if arch not in _FACTORIES:
        raise ValueError(
            f"Unsupported arch '{arch}'. Choose one of {list(_FACTORIES)}.")
    model = _FACTORIES[arch](num_classes=num_classes)
    # Adapt the ImageNet stem (224x224) for 32x32 CIFAR images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
