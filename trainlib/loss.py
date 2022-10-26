from typing import Callable

import monai.losses as monai_losses
import munch
import torch.nn


def get_loss(config: munch.Munch) -> Callable:
    """Create a loss function of `type` with specific keyword arguments from config.
    Example:

        config.loss
        >>> {'DiceCELoss': {'include_background': False, 'softmax': True, 'to_onehot_y': True}}

        get_loss(config)
        >>> DiceCELoss(
        >>>   (dice): DiceLoss()
        >>>   (cross_entropy): CrossEntropyLoss()
        >>> )

    """
    loss_type = list(config.loss.keys())[0]
    loss_config = config.loss[loss_type]
    if hasattr(monai_losses, loss_type):
        loss_class = getattr(monai_losses, loss_type)
    elif hasattr(torch.nn, loss_type):
        loss_class = getattr(torch.nn, loss_type)
    else:
        raise AttributeError(f"Loss function '{loss_type}' is neither in `monai.losses` nor in `torch.nn`")

    loss = loss_class(**loss_config)
    return loss
