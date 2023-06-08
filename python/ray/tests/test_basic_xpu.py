# coding: utf-8
import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import dpctl
import dpctl.program as dpctl_prog

import torch
import torchvision
import intel_extension_for_pytorch

from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.test_utils import client_test_enabled
from ray.cluster_utils import Cluster, cluster_not_supported
from ray.exceptions import GetTimeoutError, RayTaskError
from ray.tests.client_test_utils import create_remote_signal_actor

if client_test_enabled():
    from ray.util.client import ray
else:
    import ray

logger = logging.getLogger(__name__)


def get_spirv_abspath(fn):                                                                                                                                                                                                                                     
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    spirv_file = os.path.join(curr_dir, "xpu_input_files", fn)
    return spirv_file


@ray.remote(num_cpus=0, num_gpus=1)
def gpu_func():
    try:
        q = dpctl.SyclQueue("level_zero", property="enable_profiling")
    except dpctl.SyclQueueCreationError:
        pytest.skip("No Level-zero queue is available")
    spirv_file = get_spirv_abspath("multi_kernel.spv")
    with open(spirv_file, "rb") as fin:
        spirv = fin.read()

    prog = dpctl_prog.create_program_from_spirv(q, spirv)

    assert type(prog) is dpctl_prog.SyclProgram
    assert type(prog.addressof_ref()) is int
    assert prog.has_sycl_kernel("add")
    assert prog.has_sycl_kernel("axpy")

    addKernel = prog.get_sycl_kernel("add")
    axpyKernel = prog.get_sycl_kernel("axpy")

    assert "add" == addKernel.get_function_name()
    assert "axpy" == axpyKernel.get_function_name()
    assert 3 == addKernel.get_num_args()
    assert 4 == axpyKernel.get_num_args()
    assert type(addKernel.addressof_ref()) is int
    assert type(axpyKernel.addressof_ref()) is int


def test_basic_xpu(shutdown_only):
    ray.init(num_cpus=0, num_gpus=1)

    future = gpu_func.remote()

    start = time.time()
    ready, not_ready = ray.wait([future], timeout=1)
    # assert 0.2 < time.time() - start < 0.3
    assert len(ready) == 0
    assert len(not_ready) == 1


def to_str(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents.rstrip()


@ray.remote(num_cpus=1, num_gpus=1)
def prod_func():
    input = torch.randn(4, dtype=torch.float32, device=torch.device("cpu"))                                                
    c1 = to_str(input)                                                                                             
    c2 = to_str(torch.prod(input))                                                                                
    
    input_dpcpp = input.to("xpu")
    g1 = to_str(input_dpcpp.cpu())                                                                                 
    g2 = to_str(torch.prod(input_dpcpp).cpu())

    return (c1, c2, g1, g2)


def test_basic_prod(shutdown_only):
    ray.init(num_cpus=1, num_gpus=1)
    job = prod_func.remote()

    res = ray.get(job)

    print(f"res = {res}")

    assert res[0] == res[2]
    assert res[1] == res[3]


def test_nms(shutdown_only):
    ray.init(num_cpus=1, num_gpus=1)

    @ray.remote(num_cpus=1, num_gpus=1)
    def nms_func():
        box = torch.FloatTensor([[2, 3.1, 1, 7], [3, 4, 8, 4.8], [4, 4, 5.6, 7],
            [0.1, 0, 8, 1], [4, 4, 5.7, 7.2]]).xpu()
        score = torch.FloatTensor([0.5, 0.3, 0.2, 0.4, 0.3]).xpu()
        out_ref = torch.LongTensor([0, 3, 1, 4])
        out = torchvision.ops.nms(box, score, 0.3)
        return (to_str(out.cpu()), to_str(out_ref))

    job = nms_func.remote()
    res = ray.get(job)
    print(f"in test_nms, res = {res}")
    assert res[0] == res[1]


def test_batched_nms(shutdown_only):
    ray.init(num_cpus=1, num_gpus=1)
    @ray.remote(num_cpus=1, num_gpus=1)
    def batched_nms_func():
        box1 = torch.FloatTensor([[2, 3.1, 1, 7], [3, 4, 8, 4.8], [4, 4, 5.6, 7],
                                 [0.1, 0, 8, 1], [4, 4, 5.7, 7.2]])
        score1 = torch.FloatTensor([0.5, 0.3, 0.2, 0.4, 0.3])
        idx1 = torch.LongTensor([2,1,3,4,0])
        box2 = torch.FloatTensor([[2, 3.1, 1, 5], [3, 4, 8, 4.8], [4, 4, 5.6, 7],
                                    [0.1, 0, 6, 1], [4, 4, 5.7, 7.2]])
        score2 = torch.FloatTensor([0.5, 0.1, 0.2, 0.4, 0.8])
        idx2 = torch.LongTensor([0,1,2,4,3])
        boxes = torch.cat([box1, box2], dim=0).xpu()
        scores = torch.cat([score1, score2], dim=0).xpu()
        idxs = torch.cat([idx1, idx2], dim=0).xpu()
        out = torchvision.ops.batched_nms(boxes, scores, idxs, 0.3)
        out_ref = torch.LongTensor([9, 0, 5, 3, 1, 4, 7])
        return (to_str(out.cpu()), to_str(out_ref))
    job = batched_nms_func.remote()
    res = ray.get(job)

    print(f"in test_batched_nms, res = {res}")
    assert res[0] == res[1]

"""
def test_linear(shutdown_only):
    ray.init(num_cpus=1, num_gpus=1)
    @ray.remote(num_cpus=1, num_gpus=0)
    def cpu_task_func():
        device = torch.device("cpu:0")
        x = torch.tensor([[1, 2, 3, 4, 5],
                          [2, 3, 4, 5, 6],
                          [3, 4, 5, 6, 7],
                          [4, 5, 6, 7, 8],
                          [5, 6, 7, 8, 9]],
                         dtype=torch.float,
                         device=device)
        l = torch.nn.Linear(5, 5).to(device, torch.float)
        r = l(x)
        return to_str(r)

    @ray.remote(num_cpus=0, num_xpus=1)
    def xpu_task_func():
        device = torch.device("xpu:0")
        x = torch.tensor([[1, 2, 3, 4, 5],
                          [2, 3, 4, 5, 6],
                          [3, 4, 5, 6, 7],
                          [4, 5, 6, 7, 8],
                          [5, 6, 7, 8, 9]],
                         dtype=torch.float,
                         device=device)
        l = torch.nn.Linear(5, 5).to(device, torch.float)
        r = l(x)
        return to_str(r)

    jobs = [cpu_task_func.remote(), xpu_task_func.remote()]
    res = ray.get(jobs)

    print(to_str(res))
    assert res[0] == res[1]
"""
