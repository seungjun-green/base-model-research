import torch
from mup import MuAdam, set_base_shapes
from mup import init as mup_init

from .models import ResearchLM


def register_shapes(model):
    """Register width axes without modifying parameter values (including on resume)."""
    config = model.config
    with torch.device("meta"):
        base = ResearchLM(config.at_width(1), model.hp)
        delta = ResearchLM(config.at_width(2), model.hp)
    set_base_shapes(model, base, delta=delta, rescale_params=False)
    return model


def build_model(config, hp, device="cpu", initialize=True):
    # Avoid a full pass of default initializations and temporary readout allocation.
    with torch.device("meta"):
        model = ResearchLM(config, hp)
    model.to_empty(device=device)
    # to_empty may separate tied aliases; explicitly restore sharing before shapes.
    model.readout.weight = model.embedding.weight
    register_shapes(model)
    if initialize:
        for name, parameter in model.named_parameters():
            if name.endswith(".previous_attention_alpha"):
                torch.nn.init.zeros_(parameter)
            elif parameter.ndim == 1:
                torch.nn.init.ones_(parameter)
            else:
                mup_init.normal_(parameter, std=config.initializer_std * hp.init_multiplier)
    return model


def make_optimizer(model, training):
    return MuAdam(
        model.parameters(),
        lr=model.hp.learning_rate,
        betas=tuple(training.betas),
        eps=training.eps,
        weight_decay=training.weight_decay,
    )


def parameter_report(model, optimizer=None):
    learning_rates = (
        {}
        if optimizer is None
        else {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    )
    return [
        {
            "name": name,
            "shape": list(p.shape),
            "ninf": p.infshape.ninf(),
            "width_multiplier": p.infshape.width_mult(),
            "lr": learning_rates.get(id(p)),
        }
        for name, p in model.named_parameters()
    ]
