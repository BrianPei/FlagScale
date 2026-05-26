import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_cuda = types.ModuleType("torch.cuda")

    class _FakeDtype:
        pass

    class _FakeOutOfMemoryError(RuntimeError):
        pass

    fake_dist.is_initialized = MagicMock(return_value=False)
    fake_dist.get_rank = MagicMock(return_value=0)
    fake_dist.get_world_size = MagicMock(return_value=1)
    fake_dist.get_backend = MagicMock(return_value="gloo")
    fake_dist.new_group = MagicMock()
    fake_dist.monitored_barrier = MagicMock()
    fake_dist.barrier = MagicMock()
    fake_dist.all_reduce = MagicMock()
    fake_dist.init_process_group = MagicMock()
    fake_dist.destroy_process_group = MagicMock()
    fake_dist.batch_isend_irecv = MagicMock(return_value=[])
    fake_dist.irecv = MagicMock()
    fake_dist.isend = MagicMock()
    fake_dist.P2POp = MagicMock()
    fake_dist.ReduceOp = types.SimpleNamespace(SUM="sum")

    fake_cuda.OutOfMemoryError = _FakeOutOfMemoryError
    fake_cuda.is_available = MagicMock(return_value=False)
    fake_cuda.device_count = MagicMock(return_value=0)
    fake_cuda.set_device = MagicMock()
    fake_cuda.empty_cache = MagicMock()
    fake_cuda.synchronize = MagicMock()

    fake_torch.distributed = fake_dist
    fake_torch.cuda = fake_cuda
    fake_torch.dtype = _FakeDtype
    fake_torch.float32 = _FakeDtype()
    fake_torch.double = _FakeDtype()
    fake_torch.half = _FakeDtype()
    fake_torch.device = MagicMock(side_effect=lambda value: value)
    fake_torch.tensor = MagicMock()
    fake_torch.zeros = MagicMock()
    fake_torch.ones_like = MagicMock()
    fake_torch.randn = MagicMock()
    fake_torch.matmul = MagicMock()
    fake_torch.inverse = MagicMock()
    fake_torch.isnan = MagicMock(return_value=False)
    fake_torch.isinf = MagicMock(return_value=False)
    fake_torch.any = MagicMock(return_value=False)
    fake_torch.allclose = MagicMock(return_value=True)

    sys.modules["torch"] = fake_torch
    sys.modules["torch.distributed"] = fake_dist
    sys.modules["torch.cuda"] = fake_cuda


class TestGPUHealthCheck:
    """Test cases for GPU health check module"""

    def setup_method(self):
        """Reset global variables before each test"""
        import flagscale.runner.elastic.gpu_health_check as health_check

        health_check._PARALLEL_STATE = {
            "data": {"nccl": None, "gloo": None, "global_ranks": None},
            "tensor": {"nccl": None, "gloo": None, "global_ranks": None},
            "pipeline": {"nccl": None, "gloo": None, "global_ranks": None},
            "embedding": {"nccl": None, "gloo": None},
            "model": {"nccl": None},
            "gloo_world": None,
        }
        health_check._GLOBAL_ARGS = None
        health_check._CHECK_RESULTS = {
            "tensor_parallel": {"status": "pending", "error": None},
            "data_parallel": {"status": "pending", "error": None},
            "pipeline_parallel": {"status": "pending", "error": None},
            "gpu_hardware": {"status": "pending", "error": None},
            "gpu_computation": {"status": "pending", "error": None},
        }

    def test_parse_args_default_values(self):
        """Test argument parsing with default values"""
        from flagscale.runner.elastic.gpu_health_check import parse_args

        test_args = [
            "--tensor-model-parallel-size",
            "2",
            "--pipeline-model-parallel-size",
            "2",
        ]
        with (
            patch("sys.argv", ["gpu_health_check.py", *test_args]),
            patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "8", "LOCAL_RANK": "0"}),
        ):
            # with patch("sys.argv", ["gpu_health_check.py"] + test_args):
            #    with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "8", "LOCAL_RANK": "0"}):
            args = parse_args()

            assert args.tensor_model_parallel_size == 2
            assert args.pipeline_model_parallel_size == 2
            assert args.distributed_backend == "nccl"
            assert args.distributed_timeout_minutes == 10
            assert args.rank == 0
            assert args.world_size == 8
            assert args.local_rank == 0

    def test_parse_args_custom_values(self):
        """Test argument parsing with custom values"""
        from flagscale.runner.elastic.gpu_health_check import parse_args

        test_args = [
            "--tensor-model-parallel-size",
            "4",
            "--pipeline-model-parallel-size",
            "2",
            "--distributed-backend",
            "gloo",
            "--distributed-timeout-minutes",
            "30",
        ]

        with (
            patch("sys.argv", ["gpu_health_check.py", *test_args]),
            patch.dict(
                os.environ, {"RANK": "0", "WORLD_SIZE": "16", "LOCAL_RANK": "0"}
            ),
        ):
            args = parse_args()

            assert args.tensor_model_parallel_size == 4
            assert args.pipeline_model_parallel_size == 2
            assert args.distributed_backend == "gloo"
            assert args.distributed_timeout_minutes == 30

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_initialize_model_parallel_valid_config(
        self, mock_rank, mock_world_size, mock_init
    ):
        """Test initialize_model_parallel with valid configuration"""
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        with (
            patch("torch.distributed.new_group") as mock_new_group,
            patch("torch.distributed.get_backend", return_value="nccl"),
        ):
            mock_group = MagicMock()
            mock_new_group.return_value = mock_group

            # Test TP=2, PP=2, world_size=8 (2*2*2=8, valid)
            initialize_model_parallel(
                tensor_model_parallel_size=2, pipeline_model_parallel_size=2
            )

            assert mock_new_group.called

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_backend", return_value="nccl")
    def test_initialize_model_parallel_single_process_groups(
        self, mock_rank, mock_world_size, mock_init, mock_get_backend
    ):
        """Test initialize_model_parallel with single-process groups (TP=1, PP=1)"""
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        with patch("torch.distributed.new_group"):
            # TP=1, PP=1 means data parallel only
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    @patch("torch.distributed.get_backend", return_value="nccl")
    @patch("torch.distributed.monitored_barrier")
    def test_check_communication_basic(
        self,
        mock_barrier,
        mock_get_backend,
        mock_set_device,
        mock_cuda,
        mock_rank,
        mock_world_size,
        mock_init,
    ):
        """Test basic communication test functionality"""
        from flagscale.runner.elastic.gpu_health_check import check_communication

        # Mock args using set_args
        mock_args = MagicMock()
        mock_args.tensor_model_parallel_size = 1
        mock_args.pipeline_model_parallel_size = 1
        mock_args.local_rank = 0

        with (
            patch("torch.distributed.barrier"),
            patch(
                "flagscale.runner.elastic.gpu_health_check.safe_check_execution"
            ) as mock_safe_exec,
        ):
            mock_safe_exec.return_value = True

            check_communication()

            # Verify that safe_check_execution was called
            assert mock_safe_exec.called

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.cuda.is_available", return_value=True)
    def test_check_hardware(self, mock_cuda, mock_rank, mock_init):
        """Test GPU hardware check functionality"""
        from flagscale.runner.elastic import gpu_health_check

        mock_args = MagicMock()
        mock_args.local_rank = 0
        gpu_health_check._GLOBAL_ARGS = mock_args
        with patch.object(
            gpu_health_check, "check_hardware_single", return_value=True
        ) as mock_single:
            gpu_health_check.check_hardware()
            assert mock_single.called

    def test_check_computation(self):
        """Test GPU computation check functionality"""
        from flagscale.runner.elastic import gpu_health_check as health_check

        health_check._GLOBAL_ARGS = MagicMock()
        health_check._GLOBAL_ARGS.local_rank = 0
        health_check._GLOBAL_ARGS.rank = 0
        health_check._GLOBAL_ARGS.world_size = 1

        with (
            patch(
                "flagscale.runner.elastic.gpu_health_check.check_computation_for_different_dtype",
                return_value=True,
            ),
            patch(
                "flagscale.runner.elastic.gpu_health_check.check_computation_endurance",
                return_value=True,
            ),
            patch(
                "flagscale.runner.elastic.gpu_health_check.check_ecc_error",
                return_value=True,
            ),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.barrier"),
            patch("torch.distributed.all_reduce"),
        ):
            health_check.check_computation()

            with patch(
                "flagscale.runner.elastic.gpu_health_check.log_check_result"
            ) as mock_log:
                health_check.check_computation()
                mock_log.assert_called_with("gpu_computation", "passed")

    def test_process_group_size_calculation(self):
        """Test process group size calculations"""
        # world_size = TP * PP * DP
        # Test valid configurations
        test_cases = [
            (8, 1, 1, 8),  # DP=8
            (8, 2, 1, 4),  # TP=2, DP=4
            (8, 2, 2, 2),  # TP=2, PP=2, DP=2
            (8, 4, 2, 1),  # TP=4, PP=2, DP=1
            (16, 2, 2, 4),  # TP=2, PP=2, DP=4
        ]

        for world_size, tp, pp, expected_dp in test_cases:
            calculated_dp = world_size // (tp * pp)
            assert (
                calculated_dp == expected_dp
            ), f"Failed for world_size={world_size}, TP={tp}, PP={pp}"

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_backend", return_value="nccl")
    def test_initialize_model_parallel_debug_output(
        self, mock_get_backend, mock_rank, mock_world_size, mock_init
    ):
        """Test that initialize_model_parallel produces debug output"""
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        with (
            patch("torch.distributed.new_group"),
            patch("builtins.print") as mock_print,
        ):
            initialize_model_parallel(
                tensor_model_parallel_size=2, pipeline_model_parallel_size=2
            )

            assert mock_print.called

            # Check that initialization messages were printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            debug_output = " ".join(print_calls)
            assert (
                "initialize_model_parallel" in debug_output.lower()
                or "START" in debug_output
                or mock_print.call_count > 0
            )

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_backend", return_value="nccl")
    def test_multiple_ranks(self, mock_backend, mock_rank, mock_world_size, mock_init):
        """Test behavior with different rank values"""
        import flagscale.runner.elastic.gpu_health_check as health_check
        from flagscale.runner.elastic.gpu_health_check import initialize_model_parallel

        for rank in range(8):
            mock_rank.return_value = rank

            # Reset globals for each iteration
            health_check._PARALLEL_STATE = {
                "data": {"nccl": None, "gloo": None, "global_ranks": None},
                "tensor": {"nccl": None, "gloo": None, "global_ranks": None},
                "pipeline": {"nccl": None, "gloo": None, "global_ranks": None},
                "embedding": {"nccl": None, "gloo": None},
                "model": {"nccl": None},
                "gloo_world": None,
            }

            with patch("torch.distributed.new_group"):
                initialize_model_parallel(
                    tensor_model_parallel_size=2, pipeline_model_parallel_size=2
                )

    def test_invalid_parallel_config_detection(self):
        """Test detection of invalid parallel configurations"""
        # Test that world_size % (TP * PP) == 0
        invalid_configs = [
            (8, 3, 1),  # 8 % 3 != 0
            (8, 2, 3),  # 8 % 6 != 0
            (7, 2, 2),  # 7 % 4 != 0
        ]

        for world_size, tp, pp in invalid_configs:
            # These should not divide evenly
            assert (
                world_size % (tp * pp) != 0
            ), f"Expected invalid config: world_size={world_size}, TP={tp}, PP={pp}"

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=8)
    def test_main_function_single_process(
        self, mock_device_count, mock_cuda, mock_rank, mock_init
    ):
        """Test main function in single-process mode"""
        from flagscale.runner.elastic.gpu_health_check import main

        with patch(
            "flagscale.runner.elastic.gpu_health_check.parse_args"
        ) as mock_parse_args:
            mock_args = MagicMock()
            mock_args.tensor_model_parallel_size = 1
            mock_args.pipeline_model_parallel_size = 1
            mock_args.rank = 0
            mock_args.world_size = 1
            mock_args.local_rank = 0
            mock_parse_args.return_value = mock_args

            with (
                patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}),
                patch(
                    "flagscale.runner.elastic.gpu_health_check.safe_check_execution"
                ) as mock_safe_exec,
            ):
                mock_safe_exec.return_value = True
                with patch(
                    "flagscale.runner.elastic.gpu_health_check.print_check_summary"
                ):
                    main()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=8)
    def test_main_function_multi_process(
        self, mock_device_count, mock_cuda, mock_is_init
    ):
        """Test main function in multi-process mode"""
        from flagscale.runner.elastic.gpu_health_check import main

        with patch(
            "flagscale.runner.elastic.gpu_health_check.parse_args"
        ) as mock_parse_args:
            mock_args = MagicMock()
            mock_args.tensor_model_parallel_size = 2
            mock_args.pipeline_model_parallel_size = 2
            mock_args.rank = 0
            mock_args.world_size = 8
            mock_args.local_rank = 0
            mock_args.distributed_backend = "nccl"
            mock_args.distributed_timeout_minutes = 10
            mock_parse_args.return_value = mock_args

            with (
                patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "0"}),
                patch(
                    "flagscale.runner.elastic.gpu_health_check.initialize_distributed"
                ),
                patch("flagscale.runner.elastic.gpu_health_check.check_communication"),
                patch("flagscale.runner.elastic.gpu_health_check.check_hardware"),
                patch("flagscale.runner.elastic.gpu_health_check.check_computation"),
                patch("flagscale.runner.elastic.gpu_health_check.print_check_summary"),
            ):
                main()

    def test_args_validation(self):
        """Test that argument values are validated"""
        from flagscale.runner.elastic.gpu_health_check import parse_args

        # Test with minimum valid values
        test_args = [
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
        ]

        with (
            patch("sys.argv", ["gpu_health_check.py", *test_args]),
            patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}),
        ):
            args = parse_args()
            assert args.tensor_model_parallel_size >= 1
            assert args.pipeline_model_parallel_size >= 1
            assert args.distributed_timeout_minutes > 0

    def test_safe_check_execution_handles_timeout_and_exception(self):
        """Test timeout and generic exception branches."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        def timeout_check():
            raise TimeoutError("check timed out")

        def error_check():
            raise RuntimeError("boom")

        with patch.object(health_check, "log_check_result") as mock_log:
            assert (
                health_check.safe_check_execution(timeout_check, "tensor_parallel")
                is False
            )
            mock_log.assert_called_with("tensor_parallel", "failed", "check timed out")

        with patch.object(health_check, "log_check_result") as mock_log:
            assert (
                health_check.safe_check_execution(error_check, "data_parallel") is False
            )
            mock_log.assert_called_with("data_parallel", "failed", "Exception: boom")

    def test_log_check_result_updates_state_and_prints_rank0(self):
        """Test result state mutation and rank-0 output."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("builtins.print") as mock_print,
        ):
            health_check.log_check_result("gpu_hardware", "failed", "overheat")

        assert health_check._CHECK_RESULTS["gpu_hardware"] == {
            "status": "failed",
            "error": "overheat",
        }
        mock_print.assert_called_once()

    def test_check_hardware_single_all_gpus_pass(self):
        """Test normal multi-GPU hardware status parsing."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        mem_info = types.SimpleNamespace(total=100, used=20)
        fake_pynvml = types.SimpleNamespace(
            NVML_TEMPERATURE_GPU=0,
            nvmlInit=MagicMock(),
            nvmlShutdown=MagicMock(),
            nvmlDeviceGetCount=MagicMock(return_value=2),
            nvmlDeviceGetHandleByIndex=MagicMock(side_effect=lambda idx: f"gpu-{idx}"),
            nvmlDeviceGetName=MagicMock(side_effect=["A100-0", "A100-1"]),
            nvmlDeviceGetTemperature=MagicMock(side_effect=[70, 80]),
            nvmlDeviceGetPowerUsage=MagicMock(return_value=200000),
            nvmlDeviceGetEnforcedPowerLimit=MagicMock(return_value=400000),
            nvmlDeviceGetMemoryInfo=MagicMock(return_value=mem_info),
        )

        with (
            patch.dict(sys.modules, {"pynvml": fake_pynvml}),
            patch.object(health_check, "log_check_result") as mock_log,
        ):
            assert health_check.check_hardware_single() is True

        fake_pynvml.nvmlShutdown.assert_called_once()
        mock_log.assert_called_once_with("gpu_hardware", status="passed")

    def test_check_hardware_single_detects_abnormal_gpu_state(self):
        """Test overheat and memory-full failure branches."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        mem_ok = types.SimpleNamespace(total=100, used=30)
        mem_full = types.SimpleNamespace(total=100, used=99)
        fake_pynvml = types.SimpleNamespace(
            NVML_TEMPERATURE_GPU=0,
            nvmlInit=MagicMock(),
            nvmlShutdown=MagicMock(),
            nvmlDeviceGetCount=MagicMock(return_value=2),
            nvmlDeviceGetHandleByIndex=MagicMock(side_effect=lambda idx: f"gpu-{idx}"),
            nvmlDeviceGetName=MagicMock(side_effect=["A100-0", "A100-1"]),
            nvmlDeviceGetTemperature=MagicMock(side_effect=[86, 91]),
            nvmlDeviceGetPowerUsage=MagicMock(return_value=390000),
            nvmlDeviceGetEnforcedPowerLimit=MagicMock(return_value=400000),
            nvmlDeviceGetMemoryInfo=MagicMock(side_effect=[mem_ok, mem_full]),
        )

        with (
            patch.dict(sys.modules, {"pynvml": fake_pynvml}),
            patch.object(health_check, "log_check_result") as mock_log,
        ):
            assert health_check.check_hardware_single() is False

        args, kwargs = mock_log.call_args
        assert args == ("gpu_hardware",)
        assert kwargs["status"] == "failed"
        assert "overheat" in kwargs["error"]
        assert "memory almost full" in kwargs["error"]

    def test_check_hardware_single_handles_missing_pynvml_and_runtime_error(self):
        """Test dependency-missing and generic exception fallback branches."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        with (
            patch.dict(sys.modules, {"pynvml": None}),
            patch.object(health_check, "log_check_result") as mock_log,
        ):
            assert health_check.check_hardware_single() is False
            mock_log.assert_called_with(
                "gpu_hardware", status="failed", error="pynvml is not installed"
            )

        fake_pynvml = types.SimpleNamespace(
            nvmlInit=MagicMock(side_effect=RuntimeError("driver unavailable"))
        )
        with (
            patch.dict(sys.modules, {"pynvml": fake_pynvml}),
            patch.object(health_check, "log_check_result") as mock_log,
        ):
            assert health_check.check_hardware_single() is False
            mock_log.assert_called_with(
                "gpu_hardware", status="failed", error="driver unavailable"
            )

    @pytest.mark.parametrize(
        ("nan_result", "inf_result", "expected"),
        [(False, False, True), (True, False, False), (False, True, False)],
    )
    def test_check_computation_for_different_dtype_handles_nan_and_inf(
        self, nan_result, inf_result, expected
    ):
        """Test computation result parsing for normal, NaN and Inf outputs."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        health_check._GLOBAL_ARGS = MagicMock(local_rank=0)
        tensor = MagicMock()
        tensor.to.return_value = tensor

        with (
            patch("torch.randn", return_value=tensor),
            patch("torch.matmul", return_value="matmul-result"),
            patch("torch.isnan", return_value="nan-check"),
            patch("torch.isinf", return_value="inf-check"),
            patch("torch.any", side_effect=[nan_result, inf_result]),
        ):
            assert (
                health_check.check_computation_for_different_dtype("float32", "float")
                is expected
            )

    def test_check_computation_single_reports_failure_when_any_subcheck_fails(self):
        """Test single-GPU computation aggregation failure branch."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        with (
            patch.object(
                health_check,
                "check_computation_for_different_dtype",
                side_effect=[True, False, True],
            ),
            patch.object(
                health_check, "check_computation_endurance", return_value=True
            ),
            patch.object(health_check, "check_ecc_error", return_value=True),
            patch.object(health_check, "log_check_result") as mock_log,
        ):
            assert health_check.check_computation_single() is False

        mock_log.assert_called_once_with("gpu_computation", "failed")

    def test_check_ecc_error_handles_success_oom_and_runtime_error(self):
        """Test ECC stress check success and command/runtime fallback branches."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        health_check._GLOBAL_ARGS = MagicMock(local_rank=0)
        tensor = MagicMock()
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.randn", return_value=tensor),
            patch("torch.matmul", return_value="result"),
            patch("torch.isnan", return_value="nan"),
            patch("torch.isinf", return_value="inf"),
            patch("torch.any", return_value=False),
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            assert health_check.check_ecc_error() is True
            assert empty_cache.call_count == 5

        with patch(
            "torch.randn", side_effect=health_check.torch.cuda.OutOfMemoryError("oom")
        ):
            assert health_check.check_ecc_error() is False

        with patch("torch.randn", side_effect=RuntimeError("cuda launch failed")):
            assert health_check.check_ecc_error() is False

    def test_print_check_summary_single_and_multi_gpu_outputs(self):
        """Test summary branches for single-GPU and multi-GPU scopes."""
        from flagscale.runner.elastic import gpu_health_check as health_check

        health_check._GLOBAL_ARGS = MagicMock(world_size=1)
        health_check._CHECK_RESULTS["gpu_hardware"] = {
            "status": "passed",
            "error": None,
        }
        health_check._CHECK_RESULTS["gpu_computation"] = {
            "status": "failed",
            "error": "bad",
        }

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("builtins.print") as mock_print,
        ):
            health_check.print_check_summary()

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "HARDWARE ONLY" in output
        assert "bad" in output

        health_check._GLOBAL_ARGS = MagicMock(world_size=2)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("builtins.print") as mock_print,
        ):
            health_check.print_check_summary()

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "ALL CHECKS" in output
