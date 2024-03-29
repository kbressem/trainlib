import inspect
from typing import Callable, List

import monai
import munch
from monai.transforms import Compose
from monai.utils.enums import CommonKeys

from trainlib.utils import get_n_classes_of_model_from_config, import_patched


def _concat_image_and_maybe_label(config: munch.Munch) -> List[Callable]:
    """Final concatenation of images and label, so that they can be accessed via a standardized key"""

    concat_transforms = [
        get_transform(
            "ConcatItemsd",
            config=config,
            keys=config.data.image_cols,
            name=CommonKeys.IMAGE,
            dim=0,
        ),
        get_transform("SelectItemsd", config=config, keys=config.data.label_cols + config.data.image_cols),
    ]

    if config.task == "segmentation":
        concat_transforms += [
            get_transform(
                "ConcatItemsd",
                config=config,
                keys=config.data.label_cols,
                name=CommonKeys.LABEL,
                dim=0,
            )
        ]
    # TODO: This only works for one label. It would be better if a way is found to also concat these labels
    # CommonKeys.LABEL should still be used, as the monai.engines also use this internally.
    if config.task == "classification" and config.data.label_cols[0] != CommonKeys.LABEL:
        concat_transforms += [
            get_transform("CopyItemsd", config=config, keys=config.data.label_cols, times=1, names=[CommonKeys.LABEL])
        ]

    return concat_transforms


def get_transform(tfm_name: str, config: munch.Munch, **kwargs):
    """Get transform from monai.transforms with arguments from config"""
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
            if transform_config[tfm_name] is not None:
                kwargs = {**transform_config[tfm_name], **kwargs}

    allowed_kwargs = inspect.signature(transform.__init__).parameters.keys()  # type: ignore
    if "keys" not in kwargs.keys():
        kwargs["keys"] = config.data.image_cols + config.data.label_cols
    if "prob" in allowed_kwargs and "prob" not in kwargs.keys():
        kwargs["prob"] = config.transforms.prob
    # Finally remove all kwargs, which are not accepted by the function
    for k in list(kwargs.keys()):
        if k not in allowed_kwargs:
            kwargs.pop(k)
    return transform(**kwargs)


def get_base_transforms(config: munch.Munch) -> List[Callable]:
    """Transforms applied everytime at the start of the transform pipeline"""
    tfms = []
    if "base" in config.transforms.keys():
        tfm_names = list(config.transforms.base)
        tfms += [get_transform(tn, config) for tn in tfm_names]
    return tfms


def get_train_transforms(config: munch.Munch) -> Compose:
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

    # Concat multisequence data to single Tensors on the ChannelDim
    # Rename images to `CommonKeys.IMAGE` and labels to `CommonKeys.LABELS`
    # for more compatibility with monai.engines

    tfms += _concat_image_and_maybe_label(config)

    return Compose(tfms)


def get_valid_transforms(config: munch.Munch) -> Compose:
    """Transforms applied only to the valid dataset"""
    tfms = get_base_transforms(config=config)
    tfms += [
        get_transform("EnsureTyped", config=config, data_type="tensor"),
    ]
    if "valid" in config.transforms.keys():
        tfm_names = list(config.transforms.valid)
        tfms += [get_transform(tn, config) for tn in tfm_names]

    tfms += _concat_image_and_maybe_label(config)
    return Compose(tfms)


def get_test_transforms(config: munch.Munch) -> Compose:
    """Transforms applied only to the test dataset"""
    tfms = get_base_transforms(config=config)
    tfms += [
        get_transform("EnsureTyped", config=config, allow_missing_keys=True),
    ]
    if "test" in config.transforms.keys():
        tfm_names = list(config.transforms.test)
        tfms += [get_transform(tn, config) for tn in tfm_names]

    tfms += _concat_image_and_maybe_label(config)

    return Compose(tfms)


def get_post_transforms(config: munch.Munch):
    """Transforms applied to the model output, before metrics are calculated"""

    tfms = [get_transform("EnsureTyped", config=config, keys=[CommonKeys.PRED, CommonKeys.LABEL])]

    n_classes = get_n_classes_of_model_from_config(config)

    tfms += [
        get_transform(
            "AsDiscreted",
            config=config,
            keys=[CommonKeys.PRED, CommonKeys.LABEL],
            argmax=[True, False],
            to_onehot=n_classes,
            num_classes=n_classes,
        ),
    ]

    if "postprocessing" in config.transforms.keys():
        tfm_names = list(config.transforms.postprocessing)
        tfms += [get_transform(tn, config) for tn in tfm_names]

    if config.task == "classification":
        tfms += [
            get_transform(
                "ToDeviced",  # Avoids ROCAUC error when tensor is on device: cuda
                device="cpu",
                config=config,
                keys=[CommonKeys.PRED, CommonKeys.LABEL],
            )
        ]

    return Compose(tfms)
