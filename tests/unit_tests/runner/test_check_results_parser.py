from tests.test_utils.runners.check_results import (
    extract_marked_output,
    extract_metrics_from_log,
)


def test_extract_marked_output_skips_license_header():
    lines = [
        "# Copyright 2026 FlagOS Contributors\n",
        "\n",
        "**************************************************\n",
        "output.prompt='hello'\n",
        "output.outputs[0].text='world'\n",
    ]

    assert extract_marked_output(lines) == lines[2:]


def test_extract_marked_output_preserves_unmarked_results():
    lines = ["legacy output\n"]

    assert extract_marked_output(lines) == lines


def test_extract_metrics_from_log_supports_pipe_separated_format():
    lines = [
        " [2026-01-15 09:13:30] iteration 4/10 | consumed samples: 8 | lm loss: 1.161108E+01 |\n"
    ]

    result = extract_metrics_from_log(lines, ["lm loss:"])

    assert result == {"lm loss:": {"values": [11.61108]}}


def test_extract_metrics_from_log_supports_ansi_and_non_pipe_format():
    lines = [
        "\x1b[32m[2026-05-19 17:32:11.504168]\x1b[0m iteration 1/10 lm loss: 1.193016E+01 loss scale: 1.0\n"
    ]

    result = extract_metrics_from_log(lines, ["lm loss:"])

    assert result == {"lm loss:": {"values": [11.93016]}}
