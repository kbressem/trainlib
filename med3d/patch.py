from copy import deepcopy
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai import transforms
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.utils import ImageMetaKey as Key
from monai.utils import (
    Method,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()


class SpatialCropPad(transforms.SpatialCrop):
    def __init__(
        self,
        pad: bool = False,
        pad_method: Union[Method, str] = Method.SYMMETRIC,
        pad_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.REFLECT,
        *args,
        **kwargs,
    ) -> None:
        """
        Patches `SpatialCrop` to also do padding if the image is smaller as the roi
        Args:
            pad: wether to pad or not to pad
            pad_method: {``"symmetric"``, ``"end"``}
                Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            pad_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        """
        self.pad, self.pad_method, self.pad_mode = pad, pad_method, pad_mode
        super().__init__(*args, **kwargs)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]

        if self.pad:
            roi_size = [s.stop - s.start for s in self.slices]
            if any([i_sz < r_sz for i_sz, r_sz in zip(img.shape[1:], roi_size)]):
                img = transforms.SpatialPad(roi_size, self.pad_method, self.pad_mode)(
                    img
                )
        return img[tuple(slices)]


class RandCropByPosNegLabel(transforms.RandCropByPosNegLabel):
    def __init__(
        self,
        pad: bool = True,
        pad_method: Union[Method, str] = Method.SYMMETRIC,
        pad_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.REFLECT,
        *args,
        **kwargs,
    ):
        """
        Patches `RandCropByPosNegLabel` to also do padding if the image is smaller as the roi
        Args:
            pad: wether to pad or not to pad
            pad_method: {``"symmetric"``, ``"end"``}
                Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            pad_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        """

        self.pad, self.pad_method, self.pad_mode = pad, pad_method, pad_mode
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        img: NdarrayOrTensor,
        label: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
    ) -> List[NdarrayOrTensor]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image
        self.allow_smaller = self.allow_smaller or self.pad
        self.randomize(label, fg_indices, bg_indices, image)
        results: List[NdarrayOrTensor] = []
        if self.centers is not None:
            for center in self.centers:
                roi_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
                cropper = SpatialCropPad(
                    roi_center=center,
                    roi_size=roi_size,
                    pad=self.pad,
                    pad_method=self.pad_method,
                    pad_mode=self.pad_mode,
                )
                results.append(cropper(img))

        return results


class RandCropByPosNegLabeld(transforms.RandCropByPosNegLabeld):
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        pad: bool = True,
        pad_method: Union[Method, str] = Method.SYMMETRIC,
        pad_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.REFLECT,
    ) -> None:
        self.pad, self.pad_method, self.pad_mode = pad, pad_method, pad_mode
        transforms.MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}."
            )
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = (
            d.pop(self.fg_indices_key, None)
            if self.fg_indices_key is not None
            else None
        )
        bg_indices = (
            d.pop(self.bg_indices_key, None)
            if self.bg_indices_key is not None
            else None
        )

        self.allow_smaller = self.allow_smaller or self.pad
        self.randomize(label, fg_indices, bg_indices, image)
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [
            dict(d) for _ in range(self.num_samples)
        ]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                orig_size = img.shape[1:]
                roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
                cropper = SpatialCropPad(
                    roi_center=tuple(center),
                    roi_size=roi_size,
                    pad=self.pad,
                    pad_method=self.pad_method,
                    pad_mode=self.pad_mode,
                )
                results[i][key] = cropper(img)
                self.push_transform(
                    results[i], key, extra_info={"center": center}, orig_size=orig_size
                )
            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(
                d, self.meta_keys, self.meta_key_postfix
            ):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results
