import os
import time

from unittest.mock import patch
import pytest
import numpy as np
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import ray
import ray.data
from ray.exceptions import RayTaskError
from ray.air import session
from ray import tune

import ray.train as train
from ray.air.config import ScalingConfig
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.train.examples.pytorch.torch_linear_example import LinearDataset
from ray.train.torch.config import TorchConfig, _TorchBackend
from ray.train.torch.torch_trainer import TorchTrainer
from ray.train.trainer import TrainingFailedError
from ray.train._internal.worker_group import WorkerGroup


try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass


def test_torch_prepare_model_uses_device(ray_start_4_cpus_2_gpus):
    """Tests if `prepare_model` uses the train.torch.get_device even if it does not
    match with the local rank."""
    # The below test should pass without errors.

    @patch.object(
            ray.train.torch.train_loop_utils,
            "get_device",
            lambda: torch.device(f"cuda:{1 - session.get_local_rank()}"),
            )
    def train_func():
        # These assert statements must hold for prepare_model to wrap with DDP.
        ######## code changes #######
        model = torch.nn.Linear(1, 1)
        model = model.to("xpu")
        data = torch.ones(1)
        data = data.to("xpu")
        model = ipex.optimize(model)
        ######## code changes #######
        model = train.torch.prepare_model(model)
        model(data)

    trainer = TorchTrainer(
            train_func, scaling_config=ScalingConfig(num_workers=2, use_gpu=True)
            )
    trainer.fit()
