import itertools
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk  # noqa N813


def resample_to_ras_and_spacing(
    image: sitk.Image, spacing: Optional[List[int]] = None, method: int = sitk.sitkLinear
) -> sitk.Image:
    """Resample an image to direction (1,0,0,0,1,0,0,0,1) specified spacing"""

    # define the Euler Transformation for Image Rotation
    euler3d = sitk.Euler3DTransform()  # define transform for rotation of the image
    image_center = np.array(image.GetSize()) / 2.0  # set rotation center to image center
    image_center_as_sitk_point = image.TransformContinuousIndexToPhysicalPoint(image_center)
    euler3d.SetCenter(image_center_as_sitk_point)

    # get index of volume edges
    w, h, d = image.GetSize()
    extreme_points = [
        (0, 0, 0),
        (w, 0, 0),
        (0, h, 0),
        (0, 0, d),
        (w, h, 0),
        (w, 0, d),
        (0, h, d),
        (w, h, d),
    ]
    # transform edges to physical points in the global coordinate system
    extreme_points = [image.TransformIndexToPhysicalPoint(pnt) for pnt in extreme_points]
    inv_euler3d = euler3d.GetInverse()
    extreme_points_transformed = [inv_euler3d.TransformPoint(pnt) for pnt in extreme_points]

    # get new min and max coordinates of image edges
    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]

    max_x = max(extreme_points_transformed)[0]
    max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    # define new direction
    # take spacing and size from original image
    # calculate new origin from extree points
    output_spacing = spacing if spacing else image.GetSpacing()
    output_direction = tuple(np.identity(3).flatten())
    output_origin = [min_x, min_y, min_z]
    output_size = [
        int((max_x - min_x) / output_spacing[0]),
        int((max_y - min_y) / output_spacing[1]),
        int((max_z - min_z) / output_spacing[2]),
    ]

    resampled_image = sitk.Resample(
        image,
        output_size,
        euler3d,
        method,
        output_origin,
        output_spacing,
        output_direction,
    )
    for k in image.GetMetaDataKeys():
        resampled_image.SetMetaData(k, image.GetMetaData(k))
    return resampled_image


def indices_for_patches(patch_edge_length: int, image_edge_length: int) -> Tuple[slice, ...]:
    """Calculate indices to split image axis to n supregions of approximate `patch_edge_length`"""
    n = math.ceil(image_edge_length / patch_edge_length) + 1
    steps = np.round(np.linspace(0, image_edge_length, n))
    return tuple([slice(int(x), int(y), 1) for x, y in zip(steps[:-1], steps[1:])])


def store_old_affine(patch: sitk.Image, image: sitk.Image) -> sitk.Image:
    """Store information about size, spacing, direction and orientation in metadata"""
    patch.SetMetaData("Original Size", str(image.GetSize()))
    patch.SetMetaData("Original Spacing", str(image.GetSpacing()))
    patch.SetMetaData("Original Direction", str(image.GetDirection()))
    patch.SetMetaData("Original Origin", str(image.GetOrigin()))
    return patch


def image_to_patches(image: sitk.Image, patch_size: List[int]) -> Tuple[List[sitk.Image], Dict]:
    """Convert an sitk.Image to multiple smaller patches.

                    O--O  O--O
        O-----O     |  |  |  |
        |     |     O--O  O--O
        |     | ->
        |     |     O--O  O--O
        O-----O     |  |  |  |
                    O--O  O--O

    Indexing in SimpleITK is an overloaded function and will automatically
    correct the orientation and origin of the patch.

    Args:
        image: The sitk image to be converted to smaller patches
        patch_size: size of the image patch. Created patches are made to be als close
        as possible to `patch_size`

    Returns:
        List of patches
    """
    image_size = image.GetSize()
    indices = []
    for axis, edge_lengths in enumerate(zip(patch_size, image_size)):
        patch_edge_length, image_edge_length = edge_lengths
        assert patch_edge_length <= image_edge_length, f"Patch size exceeds image size at axis {axis}"
        indices.append(indices_for_patches(patch_edge_length, image_edge_length))
    indices = list(itertools.product(*indices))
    patches = [store_old_affine(image[idx], image) for idx in indices]
    meta_dict = {k: image.GetMetaData(k) for k in image.GetMetaDataKeys()}
    return patches, meta_dict


def string_tuple_to_numeric(string_tuple: str) -> Tuple[Union[int, float]]:
    """Convert a str of a tuple to tuple of numbers

    example:
    >>> string_tuple = str((1, 1, 1))
    >>> string_tuple_to_numeric(string_tuple)
    (1, 1, 1)

    Args:
    strin_tuple: A tuple with numbers as string
    """

    def _float_or_int(x):
        try:
            return int(x)
        except ValueError:
            return float(x)

    numeric_values = map(_float_or_int, re.sub("\(|\)| ", "", string_tuple).split(","))  # noqa W605
    return tuple(numeric_values)  # type: ignore


def patches_to_image(patches: List[sitk.Image], meta_dict: Optional[dict] = None) -> sitk.Image:
    """Convert a list of patches back to sitk.Image

       O--O  O--O
       |  |  |  |      O-----O
       O--O  O--O      |     |
                   ->  |     |
       O--O  O--O      |     |
       |  |  |  |      O-----O
       O--O  O--O

    Args:
        patches: List of sitk.Image, representing patches of a larger image

    Returns:
        Single sitk.Image created from patches
    """
    p1 = patches[0]
    ref_size = string_tuple_to_numeric(p1.GetMetaData("Original Size"))
    reference = sitk.Image(ref_size, p1.GetPixelID())
    reference.SetDirection(string_tuple_to_numeric(p1.GetMetaData("Original Direction")))
    reference.SetSpacing(string_tuple_to_numeric(p1.GetMetaData("Original Spacing")))
    reference.SetOrigin(string_tuple_to_numeric(p1.GetMetaData("Original Origin")))
    image = sum([sitk.Resample(p, reference) for p in patches])
    if meta_dict:
        for k in meta_dict.keys():
            image.SetMetaData(k, meta_dict[k])
    return image
