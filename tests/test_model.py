import torch
from src.model import LitAutoEncoder


def test_forward_shape() -> None:
    model = LitAutoEncoder()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)

    assert y.shape == (4, 28 * 28)
