import monai.networks.nets as monai_models
import munch
from torch import nn

from trainlib.utils import import_patched


def get_model(config: munch.Munch) -> nn.Module:
    """Create a model function of `type` with specific keyword arguments from config.
    Example:

        config.model
        >>> {'UNet': {
                'act': PRELU,
                'channels': [16, 32],
                'droput: 0.1,
                'num_res_units': 1,
                'out_channels': 2,
                'strides': [2]
                }
            }

        get_model(config)
        >>> UNet(
        >>>     (model): Sequential(
        >>>         (0): ResidualUnit(
        >>>         (conv): Sequential(
        >>>         ...

    """
    model_type = list(config.model.keys())[0]
    model_config = config.model[model_type]
    model_config["spatial_dims"] = config.ndim
    if model_type == "ResNet" and "n_input_channels" not in model_config.keys():
        # name of `in_channels` argument is not identical across all models in monai
        model_config["n_input_channels"] = len(config.data.image_cols)
    elif "in_channels" not in model_config.keys():
        model_config["in_channels"] = len(config.data.image_cols)

    try:
        model_class = import_patched(config.patch.model, model_type)
    except AttributeError:
        if hasattr(monai_models, model_type):
            model_class = getattr(monai_models, model_type)
        else:
            raise AttributeError(f"model function '{model_type}' is not in `monai.networks.nets` nor patched")
    model = model_class(**model_config)
    return model
