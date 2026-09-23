import pytest
import torch

from alphazero.network import MuZeroNet, normalize_hidden_state
from alphazero.utils import auto_device


def test_normalization_uses_all_channels_and_spatial_elements():
    values = torch.tensor([
        [[[0.0, 2.0]], [[4.0, 8.0]]],
        [[[100.0, 104.0]], [[108.0, 116.0]]],
    ])
    expected = torch.tensor([[[[0.0, 0.25]], [[0.5, 1.0]]]]).expand_as(values)
    torch.testing.assert_close(normalize_hidden_state(values), expected)


def test_hidden_states_are_normalized_per_sample():
    torch.manual_seed(7)
    model = MuZeroNet(3, 3, num_channels=8).to(auto_device()).eval()
    device = next(model.parameters()).device
    observations = torch.randn(2, 3, 3, 3, device=device)
    observations[1] *= 20
    hidden = model.representation(observations)
    actions = torch.tensor([0, 8], device=device)
    recurrent = model.dynamics(hidden, actions)

    for states in (hidden, recurrent):
        torch.testing.assert_close(states.flatten(1).amin(1), torch.zeros(2, device=device))
        torch.testing.assert_close(states.flatten(1).amax(1), torch.ones(2, device=device))

    # 同一观测的结果不能随同批的其他样本改变。
    alone = model.representation(observations[:1])
    torch.testing.assert_close(hidden[:1], alone)
    torch.testing.assert_close(
        recurrent[:1], model.dynamics(alone, actions[:1])
    )


def test_constant_hidden_state_is_finite_and_differentiable():
    model = MuZeroNet(3, 3, num_channels=8).to(auto_device())
    device = next(model.parameters()).device
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    observation = torch.ones(1, 3, 3, 3, device=device, requires_grad=True)
    hidden = model.representation(observation)
    recurrent = model.dynamics(
        hidden, torch.tensor([0], device=device)
    )
    assert torch.count_nonzero(hidden) == 0
    assert torch.count_nonzero(recurrent) == 0
    (hidden.sum() + recurrent.sum()).backward()
    assert observation.grad is not None
    assert torch.isfinite(observation.grad).all()
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("scale", [0.5, 0.2, 1.0])
def test_gradient_scaling_preserves_forward_values(scale):
    from alphazero.network import scale_gradient

    value = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    scaled = scale_gradient(value, scale)
    torch.testing.assert_close(scaled, value)
    scaled.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value.detach() * scale)
