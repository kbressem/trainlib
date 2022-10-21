import monai.optimizers as monai_optimizers
import torch
import torch.optim as torch_optimizers 
import munch


def get_optimizer(model: torch.nn.Module, config: munch.Munch) -> torch.optim.Optimizer:
    """Create an optimizer of `type` with specific keyword arguments from config.
    Example:

        config.optimizer
        >>> {'Novograd': {'lr': 0.001, 'weight_decay': 0.01}}

        get_optimizer(model, config)
        >>> Novograd (
        >>> Parameter Group 0
        >>>     amsgrad: False
        >>>     betas: (0.9, 0.999)
        >>>     eps: 1e-08
        >>>     grad_averaging: False
        >>>     lr: 0.0001
        >>>     weight_decay: 0.001
        >>> )

    """
    optimizer_type = list(config.optimizer.keys())[0]
    opt_config = config.optimizer[optimizer_type]
    if hasattr(torch_optimizers, optimizer_type):
        optimizer_fun = getattr(torch_optimizers, optimizer_type)
    elif hasattr(monai_optimizers, optimizer_type):
        optimizer_fun = getattr(monai_optimizers, optimizer_type)
    else:
        raise AttributeError(f"Optimizer {optimizer_type} is not found in `monai.optimizers` or `torch.optim`")
    optimizer = optimizer_fun(model.parameters(), **opt_config)
    return optimizer
