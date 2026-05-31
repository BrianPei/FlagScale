from dataclasses import dataclass

import pytest

from flagscale.serve.metric.serve_metric import calculate_metrics


@dataclass
class _Output:
    success: bool
    output_tokens: int | None = None
    generated_text: str = ""
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] | None = None


class _TokenizerResult:
    def __init__(self, token_count):
        self.input_ids = list(range(token_count))


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return _TokenizerResult(len(text.split()))


def test_calculate_metrics_counts_successes_and_tokenizer_fallback():
    input_requests = [("a", 4), ("b", 6), ("c", 5)]
    outputs = [
        _Output(True, output_tokens=3, latency=0.7, ttft=0.1, itl=[0.1, 0.2]),
        _Output(
            True,
            output_tokens=None,
            generated_text="hello world",
            latency=0.4,
            ttft=0.2,
            itl=[0.05],
        ),
        _Output(False),
    ]

    metrics, actual_lens = calculate_metrics(
        input_requests,
        outputs,
        dur_s=2.0,
        tokenizer=_Tokenizer(),
        selected_percentile_metrics=["ttft"],
        selected_percentiles=[50, 90],
    )

    assert actual_lens == [3, 2, 0]
    assert metrics.completed == 2
    assert metrics.total_input == 10
    assert metrics.total_output == 5
    assert metrics.request_throughput == 1.0
    assert metrics.output_throughput == 2.5
    assert metrics.total_token_throughput == 7.5
    assert metrics.mean_ttft_ms == pytest.approx(150.0)
    assert metrics.median_ttft_ms == pytest.approx(150.0)
    assert metrics.mean_tpot_ms == pytest.approx(250.0)
    assert metrics.mean_itl_ms == pytest.approx(116.6666667)
    assert metrics.mean_e2el_ms == pytest.approx(550.0)
    assert metrics.percentiles_ttft_ms[0] == (50, pytest.approx(150.0))
    assert metrics.percentiles_e2el_ms[1] == (90, pytest.approx(670.0))


def test_calculate_metrics_warns_and_returns_zero_metrics_when_all_requests_fail():
    outputs = [_Output(False), _Output(False)]

    with pytest.warns(UserWarning, match="All requests failed"):
        metrics, actual_lens = calculate_metrics(
            [("a", 3), ("b", 4)],
            outputs,
            dur_s=4.0,
            tokenizer=_Tokenizer(),
            selected_percentile_metrics=[],
            selected_percentiles=[50],
        )

    assert actual_lens == [0, 0]
    assert metrics.completed == 0
    assert metrics.total_input == 0
    assert metrics.total_output == 0
    assert metrics.request_throughput == 0.0
    assert metrics.mean_ttft_ms == 0.0
    assert metrics.percentiles_tpot_ms == [(50, 0.0)]
