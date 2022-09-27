import inspect
from typing import Callable, List

import monai
from monai.transforms import Compose
from monai.utils.enums import CommonKeys

from .utils import import_patched


def get_transform(tfm_name: str, config: dict, **kwargs):
    """Get transform form monai.transforms with arguments from config"""
    try:
        transform = import_patched(config.patch.transforms, tfm_name)
    except AttributeError:
        if hasattr(monai.transforms, tfm_name):
            transform = getattr(monai.transforms, tfm_name)
            assert "dictionary" in transform.__module__, f"{tfm_name} is not a dictionary transform"
        else:
            raise AttributeError(f"{tfm_name} not in `monai.transforms` nor in {config.patch.transforms}")
    # merge config for transform type (train, valid, test) with kwargs
    for k in config.transforms.keys():
        if isinstance(config.transforms[k], dict) and tfm_name in config.transforms[k].keys():
            transform_config = config.transforms[k]
            for k in kwargs.keys():
                if k in transform_config[tfm_name].keys():
                    transform_config[tfm_name].pop(k)
            kwargs = {**transform_config[tfm_name], **kwargs}

    allowed_kwargs = inspect.signature(transform.__init__).parameters.keys()
    if "keys" not in kwargs.keys():
        kwargs["keys"] = config.data.image_cols + config.data.label_cols
    if "mode" in allowed_kwargs and "mode" not in kwargs.keys():
        kwargs["mode"] = config.transforms.mode
    if "prob" in allowed_kwargs and "prob" not in kwargs.keys():
        kwargs["prob"] = config.transforms.prob
    # Finally remove all kwargs, which are not accepted by the function
    for k in list(kwargs.keys()):
        if k not in allowed_kwargs:
            kwargs.pop(k)
    return transform(**kwargs)


def get_base_transforms(config: dict) -> List[Callable]:
    """Transforms applied everytime at the start of the transform pipeline"""
    tfms = [
        get_transform("LoadImaged", config=config, allow_missing_keys=True),
        get_transform("EnsureChannelFirstd", config=config, allow_missing_keys=True),
    ]
    if "base" in config.transforms.keys():
        tfm_names = list(config.transforms.base)
        tfms += [get_transform(tn, config) for tn in tfm_names]
    return tfms


def get_train_transforms(config: dict) -> Compose:
    """Build transforms dynamically from config for data augmentation during training.
    Args:
        config: parsed YAML file with global configurations
    Returns:
        Composed transforms
    """
    tfms = get_base_transforms(config=config)

    if "train" in config.transforms.keys():
        tfm_names = list(config.transforms.train)
        train_tfms = [get_transform(tn, config) for tn in tfm_names]
        tfms += [tfm for tfm in train_tfms if tfm not in tfms]  # add rest

    # Concat mutlisequence data to single Tensors on the ChannelDim
    # Rename images to `CommonKeys.IMAGE` and labels to `CommonKeys.LABELS`
    # for more compatibility with monai.engines

    tfms += [
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.image_cols,
            name=CommonKeys.IMAGE,
            dim=0,
        ),
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.label_cols,
            name=CommonKeys.LABEL,
            dim=0,
        ),
    ]

    return Compose(tfms)


def get_val_transforms(config: dict) -> Compose:
    """Transforms applied only to the valid dataset"""
    tfms = get_base_transforms(config=config)
    tfms += [
        get_transform("EnsureTyped", config=config),
        get_transform(
            "ScaleIntensityd",
            config=config,
            keys=config.data.image_cols,
            minv=0,
            maxv=1,
            allow_missing_keys=True,
        ),
        get_transform(
            "NormalizeIntensityd",
            config=config,
            keys=config.data.image_cols,
            allow_missing_keys=True,
        ),
    ]
    if "valid" in config.transforms.keys():
        tfm_names = list(config.transforms.valid)
        tfms += [get_transform(tn, config) for tn in tfm_names]

    tfms += [
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.image_cols,
            name=CommonKeys.IMAGE,
            dim=0,
        ),
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.label_cols,
            name=CommonKeys.LABEL,
            dim=0,
        ),
    ]
    return Compose(tfms)


def get_test_transforms(config: dict) -> Compose:
    """Transforms applied only to the test dataset"""
    tfms = get_base_transforms(config=config)
    tfms += [
        get_transform("EnsureTyped", config=config, allow_missing_keys=True),
        get_transform(
            "ScaleIntensityd",
            config=config,
            keys=config.data.image_cols,
            minv=0,
            maxv=1,
            allow_missing_keys=True,
        ),
        get_transform(
            "NormalizeIntensityd",
            config=config,
            keys=config.data.image_cols,
            allow_missing_keys=True,
        ),
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.image_cols,
            name=CommonKeys.IMAGE,
            dim=0,
        ),
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.label_cols,
            name=CommonKeys.LABEL,
            dim=0,
            allow_missing_keys=True,
        ),
    ]

    return Compose(tfms)


def get_post_transforms(config: dict):
    """Transforms applied to the model output, before metrics are calculated"""
    tfms = [
        get_transform("EnsureTyped", config=config, keys=[CommonKeys.PRED, CommonKeys.LABEL]),
        get_transform(
            "AsDiscreted",
            config=config,
            keys=CommonKeys.PRED,
            argmax=True,
            to_onehot=config.model.out_channels,
            num_classes=config.model.out_channels,
        ),
        get_transform(
            "AsDiscreted",
            config=config,
            keys=CommonKeys.LABEL,
            to_onehot=config.model.out_channels,
            num_classes=config.model.out_channels,
        ),
    ]

    if "postprocessing" in config.transforms.keys():
        tfm_names = list(config.transforms.postprocessing)
        tfms += [get_transform(tn, config) for tn in tfm_names]

    return Compose(tfms)
