import monai
from monai.transforms import (
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityd,
    Spacingd,
)
from monai.utils.enums import CommonKeys

from . import patch
from .patch import EnsureChannelFirstd

# images should be interploated with `bilinear` but masks with `nearest`
# ---------- base transforms ----------
# applied everytime


def get_base_transforms(config: dict, minv: int = 0, maxv: int = 1) -> list:
    tfms = [
        LoadImaged(keys=config.data.image_cols + config.data.label_cols, allow_missing_keys=True),
        EnsureChannelFirstd(
            keys=config.data.image_cols + config.data.label_cols, allow_missing_keys=True
        ),
        Spacingd(
            keys=config.data.image_cols + config.data.label_cols,
            mode=config.transforms.mode,
            pixdim=config.transforms.spacing,
            allow_missing_keys=True,
        ),
    ]
    return tfms


# ---------- train transforms ----------


def get_transform(tfm_name: str, config: dict):
    "Get transform form monai.transforms with arguments form config"
    if hasattr(patch, tfm_name):
        transform = getattr(patch, tfm_name)
    else:
        transform = getattr(monai.transforms, tfm_name)
        assert "dictionary" in transform.__module__, f"{tfm_name} is not a dictionary transform"
    kwargs = config.transforms[tfm_name]
    allowed_kwargs = transform.__init__.__code__.co_varnames
    if "keys" not in kwargs.keys():
        kwargs["keys"] = config.data.image_cols + config.data.label_cols
    if "mode" in allowed_kwargs and "mode" not in kwargs.keys():
        kwargs["mode"] = config.transforms.mode
    if "prob" in allowed_kwargs and "prob" not in kwargs.keys():
        kwargs["prob"] = config.transforms.prob
    return transform(**kwargs)


def get_train_transforms(config: dict):
    """Build transforms dynamically from config.

    The order of transforms in the config will be irgnored and transforms are
    ordered as: spatial transforms, croppad, spatial transforms, rest

    Args:
        config: parsed YAML file with global configurations
    Returns:
        Composed transforms
    """
    tfms = get_base_transforms(config=config)

    # some arguments in config.transforms are not a transform but a global argument
    # such as the probability a transfor is applied

    not_a_transform = ["prob", "spacing", "orientation", "mode"]
    tfm_names = [tn for tn in config.transforms if tn not in not_a_transform]
    train_tfms = [get_transform(tn, config) for tn in tfm_names]
    tfms += [tfm for tfm in train_tfms if tfm not in tfms]  # add rest

    # Concat mutlisequence data to single Tensors on the ChannelDim
    # Rename images to `CommonKeys.IMAGE` and labels to `CommonKeys.LABELS`
    # for more compatibility with monai.engines

    tfms += [
        ScaleIntensityd(keys=config.data.image_cols, minv=0, maxv=1, allow_missing_keys=True),
        NormalizeIntensityd(keys=config.data.image_cols, allow_missing_keys=True),
        ConcatItemsd(keys=config.data.image_cols, name=CommonKeys.IMAGE, dim=0),
        ConcatItemsd(keys=config.data.label_cols, name=CommonKeys.LABEL, dim=0),
    ]

    return Compose(tfms)


# ---------- valid transforms ----------


def get_val_transforms(config: dict):
    tfms = get_base_transforms(config=config)
    tfms += [EnsureTyped(keys=config.data.image_cols + config.data.label_cols)]
    tfms += [
        ScaleIntensityd(keys=config.data.image_cols, minv=0, maxv=1, allow_missing_keys=True),
        NormalizeIntensityd(keys=config.data.image_cols, allow_missing_keys=True),
        ConcatItemsd(keys=config.data.image_cols, name=CommonKeys.IMAGE, dim=0),
        ConcatItemsd(keys=config.data.label_cols, name=CommonKeys.LABEL, dim=0),
    ]
    return Compose(tfms)


# ---------- test transforms ----------
# same as valid transforms


def get_test_transforms(config: dict):
    tfms = get_base_transforms(config=config)
    tfms += [
        EnsureTyped(keys=config.data.image_cols + config.data.label_cols, allow_missing_keys=True)
    ]
    tfms += [
        ScaleIntensityd(keys=config.data.image_cols, minv=0, maxv=1, allow_missing_keys=True),
        NormalizeIntensityd(keys=config.data.image_cols, allow_missing_keys=True),
        ConcatItemsd(
            keys=config.data.image_cols, name=CommonKeys.IMAGE, dim=0, allow_missing_keys=True
        ),
        ConcatItemsd(
            keys=config.data.label_cols, name=CommonKeys.LABEL, dim=0, allow_missing_keys=True
        ),
    ]

    return Compose(tfms)


def get_val_post_transforms(config: dict):
    tfms = [
        EnsureTyped(keys=[CommonKeys.PRED, CommonKeys.LABEL]),
        AsDiscreted(
            keys=CommonKeys.PRED,
            argmax=True,
            to_onehot=config.model.out_channels,
            num_classes=config.model.out_channels,
        ),
        Lambdad(keys=CommonKeys.LABEL, func=lambda x: x[0:1]),
        AsDiscreted(
            keys=CommonKeys.LABEL,
            to_onehot=config.model.out_channels,
            num_classes=config.model.out_channels,
        ),
    ]
    return Compose(tfms)
