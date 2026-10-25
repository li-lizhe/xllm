# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for the hardware typing helpers in ``xllm/python/platform.py``.

The Ascend chip name comes from ``npu-smi info`` or from ``acl.get_soc_name()``,
so both sources are faked here; availability of the real hardware is not
required.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from xllm.auto_config import utils

Platform = utils.Platform

# Verbatim per-device rows from ``npu-smi info`` on an Ascend 910B2 host; the
# chip name is the token in the Name column.
SMI_910B2_OUTPUT = """
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B2               | OK            | 111.7                50                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                3434 / 65536            |
+===========================+===============+======================================================================+
| 1     910B2               | OK            | 94.6                 49                      0    / 0                |
| 0                         | 0000:C2:00.0  | 0                    0    / 0                3431 / 65536            |
+===========================+===============+======================================================================+
"""


def _clear_platform_caches() -> None:
    """Drop the cached detections so a case re-reads the faked sources."""
    for name in ("get_npu_chip", "get_ascend_soc_generation"):
        Platform.__dict__[name].__func__.cache_clear()


@pytest.fixture(autouse=True)
def reset_platform_caches() -> Iterator[None]:
    _clear_platform_caches()
    yield
    _clear_platform_caches()


def _fake_no_npu_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behave like a host where the ``npu-smi`` probe cannot be used."""
    monkeypatch.setattr(subprocess, "run", _missing_npu_smi)


def _missing_npu_smi(*args: Any, **kwargs: Any) -> Any:
    raise OSError("npu-smi is not installed")


def _fake_acl(monkeypatch: pytest.MonkeyPatch, soc_name: str) -> None:
    """Expose an ``acl`` module whose ``get_soc_name()`` returns ``soc_name``."""
    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(get_soc_name=lambda: soc_name))


def test_chip_comes_from_the_smi_name_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=SMI_910B2_OUTPUT))

    assert Platform.get_npu_chip() == "910b2"
    assert Platform.get_ascend_soc_generation() == "a2"


@pytest.mark.parametrize(
    ("soc_name", "expected_chip", "expected_generation"),
    [
        ("Ascend910B2", "910b2", "a2"),
        ("Ascend910B4", "910b4", "a2"),
        ("Ascend910_93", "910_93", "a3"),
        ("Ascend910C", "910c", "a3"),
        ("Ascend950", "950", "a5"),
        ("Ascend310P3", "unknown", None),
    ],
)
def test_acl_soc_name_resolves_the_chip_and_its_generation(
    monkeypatch: pytest.MonkeyPatch,
    soc_name: str,
    expected_chip: str,
    expected_generation: str | None,
) -> None:
    _fake_no_npu_smi(monkeypatch)
    _fake_acl(monkeypatch, soc_name)

    assert Platform.get_npu_chip() == expected_chip
    assert Platform.get_ascend_soc_generation() == expected_generation


def test_smi_chip_wins_over_acl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=SMI_910B2_OUTPUT))
    _fake_acl(monkeypatch, "Ascend910_93")

    assert Platform.get_npu_chip() == "910b2"
