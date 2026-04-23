import pytest
import torch
from transformers import AutoTokenizer

import nl_probes.utils.dataset_utils as dataset_utils
from nl_probes.utils.common import load_tokenizer


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3.6-27B",
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
)
def test_positions(model_name):
    tokenizer = load_tokenizer(model_name)
    num_positions = 10
    layer = 9
    prompt = "This is a test prompt"

    datapoint = dataset_utils.create_training_datapoint(
        "test",
        prompt,
        "test output",
        layer,
        num_positions,
        tokenizer,
        torch.randn(num_positions, 64),
        -1,
    )

    assert len(datapoint.positions) == num_positions
    assert datapoint.positions[-1] - datapoint.positions[0] == num_positions - 1
    print(f"{model_name} -> {datapoint}")
