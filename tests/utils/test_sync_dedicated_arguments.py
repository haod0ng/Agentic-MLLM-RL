# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.sync_dedicated import validate_sync_dedicated_args


def _args(**overrides):
    values = {
        "is_sync_dedicated": True,
        "weight_version_validation_timeout_s": 30.0,
        "fully_async": False,
        "colocate": False,
        "hybrid": False,
        "max_staleness": 0,
        "offload_train": False,
        "offload_rollout": False,
        "num_gpus_per_node": 4,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 4,
        "rollout_num_gpus": 12,
        "rollout_num_gpus_per_engine": 1,
        "resource": {
            "actor": [1, 4],
            "rollout": [1, 12],
            "judge_accuracy": [1, 4],
            "judge_multiturn_vlm": [1, 4],
        },
    }
    values.update(overrides)
    return Namespace(**values)


def test_sync_dedicated_arguments_accept_exact_disaggregated_mode():
    validate_sync_dedicated_args(_args())


@pytest.mark.parametrize("mode", ["fully_async", "colocate", "hybrid"])
def test_sync_dedicated_arguments_reject_other_execution_modes(mode: str):
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_sync_dedicated_args(_args(**{mode: True}))


def test_sync_dedicated_arguments_require_zero_staleness_and_resident_models():
    with pytest.raises(ValueError, match="max-staleness 0"):
        validate_sync_dedicated_args(_args(max_staleness=1))
    with pytest.raises(ValueError, match="no-offload"):
        validate_sync_dedicated_args(_args(offload_rollout=True))


def test_sync_dedicated_arguments_require_all_four_gpu_roles():
    resource = _args().resource
    resource.pop("judge_multiturn_vlm")
    with pytest.raises(ValueError, match="judge_multiturn_vlm"):
        validate_sync_dedicated_args(_args(resource=resource))


def test_sync_dedicated_arguments_reject_resource_or_engine_shape_drift():
    resource = _args().resource
    resource["rollout"] = [1, 8]
    with pytest.raises(ValueError, match="resource rollout"):
        validate_sync_dedicated_args(_args(resource=resource))
    with pytest.raises(ValueError, match="fixed topology"):
        validate_sync_dedicated_args(_args(rollout_num_gpus_per_engine=2))
