import inspect
from typing import Callable, Dict, List, Mapping, Hashable


import monai
import munch
from monai.transforms import Compose, MapTransform
from monai.utils.enums import CommonKeys
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.utils import TransformBackends

from trainlib.utils import import_patched


class UnsqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dim: int, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.dim = dim

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key].unsqueeze(self.dim)
        return d


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


def get_val_transforms(config: munch.Munch) -> Compose:
    """Transforms applied only to the valid dataset"""
    tfms = get_base_transforms(config=config)
    tfms += [
        get_transform("EnsureTyped", config=config),
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


def get_test_transforms(config: munch.Munch) -> Compose:
    """Transforms applied only to the test dataset"""
    tfms = get_base_transforms(config=config)
    tfms += [
        get_transform("EnsureTyped", config=config, allow_missing_keys=True),
    ]
    if "test" in config.transforms.keys():
        tfm_names = list(config.transforms.test)
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
            allow_missing_keys=True,
        ),
    ]

    return Compose(tfms)


def get_post_transforms(config: munch.Munch):
    """Transforms applied to the model output, before metrics are calculated"""

    model_name = list(config.model.keys())[0]
    out_channels = config.model[model_name].out_channels

    tfms = [
        get_transform("EnsureTyped", config=config, keys=[CommonKeys.PRED, CommonKeys.LABEL]),
        get_transform(
            "AsDiscreted",
            config=config,
            keys=[CommonKeys.PRED, CommonKeys.LABEL],
            argmax=[True, False],
            to_onehot=out_channels
        ),
    ]

    if "postprocessing" in config.transforms.keys():
        tfm_names = list(config.transforms.postprocessing)
        tfms += [get_transform(tn, config) for tn in tfm_names]

    # Items pass through the transforms pipeline as C x WH[D]
    # But metrics expect them to be B x C x WH[D]
    # We need to add the missing batch-dim, otherwise calculated metrics are wrong
    tfms += [
        UnsqueezeDimd(
            keys=[CommonKeys.PRED, CommonKeys.LABEL], 
            dim=0 
        )
    ]
    return Compose(tfms)
