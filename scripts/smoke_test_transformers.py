from __future__ import annotations

import torch

from src.models.transformers import RTTransformerForecaster, TransformerForecaster


def run_smoke_check() -> None:
    batch_size = 2
    sequence_length = 64
    input_dim = 4
    horizon = 16
    output_dim = 4
    sample = torch.randn(batch_size, sequence_length, input_dim)

    transformer = TransformerForecaster(input_dim=input_dim, horizon=horizon, output_dim=output_dim)
    rt_transformer = RTTransformerForecaster(input_dim=input_dim, horizon=horizon, output_dim=output_dim)

    transformer_output = transformer(sample)
    rt_output = rt_transformer(sample)

    assert transformer_output.shape == (batch_size, horizon, output_dim)
    assert rt_output.shape == (batch_size, horizon, output_dim)
    assert not torch.isnan(transformer_output).any()
    assert not torch.isnan(rt_output).any()
    print("transformer smoke test passed")


if __name__ == "__main__":
    run_smoke_check()
