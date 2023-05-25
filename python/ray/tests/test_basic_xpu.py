# coding: utf-8
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

def test_init_gpu(shutdown_only):
    ray.init(num_cpus=0, num_gpus=1)

    future = gpu_func.remote()

    start = time.time()
    ready, not_ready = ray.wait([future], timeout=1)
    assert 0.2 < time.time() - start < 0.3
    assert len(ready) == 0
    assert len(not_ready) == 1
