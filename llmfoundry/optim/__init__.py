# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.optim import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    DecoupledAdamW,
    LinearWithWarmupScheduler,
)

from llmfoundry.optim.adaptive_lion import DecoupledAdaLRLion, DecoupledClipLion
from llmfoundry.optim.lion import DecoupledLionW
from llmfoundry.optim.no_op import NoOp
from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler
from llmfoundry.registry import optimizers, schedulers

import torch


def register_pytorch_optimizers():
    """Register all PyTorch optimizers with the LLM Foundry registry."""

    # List of optimizers to skip
    skip_optimizers = {
        "Optimizer",  # Base class
        "LBFGS",  # Requires closure, might need special handling
    }

    for name, obj in torch.optim.__dict__.items():
        if (
            isinstance(obj, type)
            and issubclass(obj, torch.optim.Optimizer)
            and name not in skip_optimizers
        ):
            registry_name = f"pytorch_{name.lower()}"

            obj.__doc__ = (
                f"PyTorch {name} optimizer.\n\n"
                + (obj.__doc__ or "")
                + f"\n\nRegistered as '{registry_name}' in LLM Foundry."
            )

            optimizers.register(registry_name, func=obj)


optimizers.register("adalr_lion", func=DecoupledAdaLRLion)
optimizers.register("clip_lion", func=DecoupledClipLion)
optimizers.register("decoupled_lionw", func=DecoupledLionW)
optimizers.register("decoupled_adamw", func=DecoupledAdamW)
optimizers.register("no_op", func=NoOp)

# Register PyTorch optimizers
register_pytorch_optimizers()

schedulers.register("constant_with_warmup", func=ConstantWithWarmupScheduler)
schedulers.register(
    "cosine_with_warmup",
    func=CosineAnnealingWithWarmupScheduler,
)
schedulers.register("linear_decay_with_warmup", func=LinearWithWarmupScheduler)
schedulers.register(
    "inv_sqrt_with_warmup",
    func=InverseSquareRootWithWarmupScheduler,
)

__all__ = [
    "DecoupledLionW",
    "DecoupledClipLion",
    "DecoupledAdaLRLion",
    "NoOp",
    "InverseSquareRootWithWarmupScheduler",
]
