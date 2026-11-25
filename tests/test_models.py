from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from base_model_research.config import DEFAULT_TRAINING_TARGETS, GATE_VARIANTS, ModelConfig
from base_model_research.diagnostics import check_parameter_counts
from base_model_research.models import parameter_count
from base_model_research.parameterization import build_model, make_optimizer


def test_all_four_exact_parameter_counts():
    counts = check_parameter_counts()
    standard = ModelConfig.load("configs/models/standard_proxy.yaml")
    assert standard.width_mult == 1 and standard.d_model == 144 and standard.vocab_size == 32_000
    assert standard.at_width(4) == ModelConfig.load("configs/models/standard_main.yaml")
    assert ModelConfig.load("configs/models/modified_proxy.yaml") == replace(
        standard, family="previous_attention_relu_with_alpha"
    )
    assert counts["modified_proxy"]["parameters"] == counts["standard_proxy"]["parameters"] + 27
    assert counts["standard_main"]["parameters"] == 157_814_784
    assert counts["modified_main"]["parameters"] == 158_395_392
    assert max(row["training_tokens"] for row in counts.values()) == DEFAULT_TRAINING_TARGETS


@pytest.mark.parametrize("family", ["standard", "modified", "previous_attention_relu"])
def test_causal_and_tied(tiny_config, hp, family):
    model = build_model(replace(tiny_config, family=family, n_layers=2), hp).eval()
    assert model.embedding.weight is model.readout.weight
    x = torch.tensor([[1, 2, 3, 4, 5]])
    changed = x.clone()
    changed[:, 3:] = 0
    torch.testing.assert_close(model(x)[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    assert parameter_count(model) == sum(p.numel() for p in model.parameters())


def test_previous_attention_relu_matches_explicit_forward_and_backward(tiny_config, hp):
    cfg = replace(tiny_config, n_layers=3)
    reference = build_model(cfg, hp)
    model = build_model(replace(cfg, family="previous_attention_relu"), hp)
    model.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    x = reference.embedding(tokens) * hp.embedding_multiplier
    previous = torch.zeros_like(x)
    for layer in reference.layers:
        current = layer.attention(layer.input_norm(x))
        h = F.relu(x + current + previous)
        x = h + layer.ffn(layer.post_attention_norm(h))
        previous = current
    expected = reference.readout(reference.final_norm(x))
    actual = model(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_modified_equations_and_first_layer(tiny_config, hp):
    model = build_model(replace(tiny_config, family="modified", n_layers=2), hp)
    x = torch.randn(1, 3, 4)
    a_previous, f_previous = None, None
    for layer in model.layers:
        a = layer.attention(layer.input_norm(x))
        attention_previous = x if a_previous is None else a_previous
        h = a + F.gelu(F.linear(x, layer.attention_gate.current.weight)) * F.linear(
            attention_previous, layer.attention_gate.previous.weight
        )
        f = layer.ffn(layer.post_attention_norm(h))
        ffn_previous = h if f_previous is None else f_previous
        expected = f + F.gelu(F.linear(h, layer.ffn_gate.current.weight)) * F.linear(
            ffn_previous, layer.ffn_gate.previous.weight
        )
        actual, actual_a, actual_f = layer(x, a_previous, f_previous)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_a, a)
        torch.testing.assert_close(actual_f, f)
        x, a_previous, f_previous = expected, a, f


def test_previous_attention_alpha_is_a_trainable_scalar_on_only_the_previous_branch(tiny_config, hp):
    model = build_model(replace(tiny_config, family="previous_attention_relu_with_alpha", n_layers=3), hp)
    first, second, third = model.layers
    assert first.previous_attention_alpha is None
    for layer in (second, third):
        assert layer.previous_attention_alpha.shape == ()
        assert layer.previous_attention_alpha.item() == 0.0
    assert second.previous_attention_alpha is not third.previous_attention_alpha

    x = torch.randn(2, 3, tiny_config.d_model)
    previous = torch.randn_like(x, requires_grad=True)
    with torch.no_grad():
        second.previous_attention_alpha.fill_(-0.5)
    output, current, _ = second(x, previous)
    h = F.relu(x + current - 0.5 * previous)
    expected = h + second.ffn(second.post_attention_norm(h))
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert second.previous_attention_alpha.grad.abs() > 0
    assert previous.grad.abs().sum() > 0


def test_previous_block_branch_is_differentiable(tiny_config, hp):
    model = build_model(replace(tiny_config, family="modified"), hp)
    layer = model.layers[0]
    x = torch.randn(1, 2, 4, requires_grad=True)
    previous_a = torch.randn_like(x, requires_grad=True)
    previous_f = torch.randn_like(x, requires_grad=True)
    result, _, _ = layer(x, previous_a, previous_f)
    result.square().sum().backward()
    assert previous_a.grad.abs().sum() > 0
    assert previous_f.grad.abs().sum() > 0


@pytest.mark.parametrize("family", GATE_VARIANTS)
def test_gate_variant_equations_for_attention_and_ffn(tiny_config, hp, family):
    model = build_model(replace(tiny_config, family=family, n_layers=2), hp)
    first, second = model.layers
    x = torch.randn(2, 3, tiny_config.d_model)
    h, a, f = first(x)
    torch.testing.assert_close(h, x + a + f)
    a.retain_grad()
    f.retain_grad()

    def expected(gate, x, y, previous):
        if family == "multiply_previous":
            return F.gelu(F.relu(x + y)) * F.linear(previous, gate.previous_proj.weight)
        skip = previous
        if family == "scale_previous":
            assert gate.scale.item() == 1.0
            with torch.no_grad():
                gate.scale.fill_(0.5)
            skip = 0.5 * previous
        return F.gelu(F.linear(x, gate.input_proj.weight)) * F.linear(y, gate.output_proj.weight) + skip

    current_a = second.attention(second.input_norm(h))
    after_attention = expected(second.attention_gate, h, current_a, a)
    current_f = second.ffn(second.post_attention_norm(after_attention))
    after_ffn = expected(second.ffn_gate, after_attention, current_f, f)
    actual, actual_a, actual_f = second(h, a, f)
    torch.testing.assert_close(actual_a, current_a)
    torch.testing.assert_close(actual_f, current_f)
    torch.testing.assert_close(actual, after_ffn)
    actual.sum().backward()
    assert a.grad is not None and f.grad is not None
    if family == "scale_previous":
        assert second.attention_gate.scale.grad is not None and second.ffn_gate.scale.grad is not None


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_chunked_loss_and_gradients_match_full_logits(tiny_config, hp, family):
    cfg = replace(tiny_config, family=family, n_layers=2)
    model = build_model(cfg, hp)
    reference = build_model(cfg, hp)
    reference.load_state_dict(model.state_dict())
    x = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    y = torch.tensor([[2, 3, 4, 5], [3, 4, -100, -100]])
    full = F.cross_entropy(reference(x).reshape(-1, 8), y.reshape(-1), reduction="sum")
    chunked = model.loss_sum(x, y, chunk_size=3)
    torch.testing.assert_close(chunked, full)
    full.backward()
    chunked.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("family", ["standard", "modified"])
def test_activation_checkpointing_gradients(tiny_config, hp, family):
    cfg = replace(tiny_config, family=family, n_layers=2)
    a, b = build_model(cfg, hp), build_model(cfg, hp)
    b.load_state_dict(a.state_dict())
    b.activation_checkpointing = True
    x = torch.tensor([[1, 2, 3, 4]])
    a.loss_sum(x, x, 3).backward()
    b.loss_sum(x, x, 3).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_mup_lr_groups_and_shared_output_scaling(tiny_config, hp, training_config):
    proxy = build_model(tiny_config, hp)
    main = build_model(tiny_config.at_width(4), hp)
    optimizer = make_optimizer(main, training_config)
    rates = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    assert rates[id(main.layers[0].ffn.up_proj.weight)] == hp.learning_rate / 4
    assert rates[id(main.embedding.weight)] == hp.learning_rate
    assert main.layers[0].attention.q_norm.weight.infshape.ninf() == 1
    hidden = torch.randn(1, 2, main.config.d_model)
    expected = F.linear(hidden, main.embedding.weight) * hp.output_multiplier / 4
    torch.testing.assert_close(main.readout(hidden), expected)
    assert proxy.readout.width_mult() == 1


def test_modified_gates_are_hidden_mup_weights(tiny_config, hp, training_config):
    model = build_model(replace(tiny_config, family="modified").at_width(4), hp)
    optimizer = make_optimizer(model, training_config)
    rates = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    for name, parameter in model.named_parameters():
        if "_gate." in name:
            assert parameter.infshape.ninf() == 2
            assert rates[id(parameter)] == hp.learning_rate / 4
