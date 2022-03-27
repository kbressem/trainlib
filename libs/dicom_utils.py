__all__ = ['show_image', 'Reader', 'equal_number_of_slices', 'crop', 'resample', 'resize']

import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed

def show_image(image, mask=None, mask_alpha=0.4, **kwargs):
    "Display a DICOM series, optional with mask overlay"
    def _inner(image, slice_id):
        img_slice = sitk.GetArrayViewFromImage(image)[slice_id, :, :]
        plt.imshow(img_slice, cmap='Greys_r', **kwargs)
        if mask:
            mask_slice = sitk.GetArrayViewFromImage(mask)[slice_id, :, :]
            plt.imshow(mask_slice, cmap='jet', alpha=mask_alpha, **kwargs)

    interact(_inner, image=fixed(image), slice_id = (0, image.GetDepth()-1));

class Reader():
    "Read a medical image volume"
    def __init__(self): pass

    def __call__(self, fn):
        fn = str(fn) # convert possible pathlib.Path to str
        if not os.path.exists(fn):
            raise FileNotFoundError(f'{fn} is neither a valid directory or file')
        if os.path.isdir(fn):
            return self.read_series(fn)
        if os.path.isfile(fn):
            return self.read_file(fn)

    def read_series(self, fn:str):
        "load a DICOM series as `sitk.Image`"
        SeriesReader = sitk.ImageSeriesReader()
        dicom_names = SeriesReader.GetGDCMSeriesFileNames(str(fn))
        SeriesReader.SetFileNames(dicom_names)
        im = SeriesReader.Execute()
        return sitk.Cast(im, sitk.sitkInt16)

    def read_file(self,fn:str):
        "load an image file as `sitk.Image`"
        return sitk.ReadImage(fn)

def equal_number_of_slices(fn_image, fn_mask):
    "compare number of slices in two images, returns bool"
    SeriesReader = sitk.ImageSeriesReader()
    dicom_names = SeriesReader.GetGDCMSeriesFileNames(str(fn_image))
    n_slices_image = len(dicom_names)
    n_slices_mask = sitk.ReadImage(fn_mask).GetDepth()
    return n_slices_image == n_slices_mask

def _flatten(t):
    return [item for sublist in t for item in sublist]

def crop(image, margin, interpolator=sitk.sitkLinear):
    """
    Crops a sitk.Image while retaining correct spacing. Negative margins will lead to zero padding

    Args:
        image: a sitk.Image
        margin: margins to crop. Single integer or float (percentage crop), lists of int/float or nestes lists are supported.
    """
    if isinstance(margin, (list, tuple)):
        assert len(margin) == 3, f'expected margin to be of length 3'
    else:
        assert isinstance(margin, (int, float)), f'expected margin to be a float value'
        margin = [margin, margin, margin]

    margin = [m if isinstance(m, (tuple, list)) else [m,m] for m in margin]
    old_size = image.GetSize()
    old_origin = image.GetOrigin()


    # calculate new origin and new image size
    # is that correct?
    if all([isinstance(m, float) for m in _flatten(margin)]):
        assert all([m >= 0 and m < 0.5 for m in _flatten(margin)]), f'margins must be between 0 and 0.5'
        to_crop = [[int(sz*_m) for _m in m] for sz, m in zip(old_size, margin)]
    elif  all([isinstance(m, int) for m in _flatten(margin)]):
        to_crop = margin
    else:
        raise ValueError('wrong format of margins')

    new_size = [sz - sum(c) for sz, c in zip(old_size, to_crop)]
    new_origin = [o+c[0] for o,c in zip(old_origin, to_crop)]

    # create reference plane to resample image
    ref_image = sitk.Image(new_size, image.GetPixelIDValue())
    ref_image.SetSpacing(image.GetSpacing())
    ref_image.SetOrigin(new_origin)
    ref_image.SetDirection(image.GetDirection())

    return sitk.Resample(image, ref_image, interpolator=interpolator)

def resample(image, spacing=None, interpolator = sitk.sitkLinear):
    """resample an image to direction (1,0,0,
                                       0,1,0,
                                       0,0,1).
       This should usualy yield an image with axial reconstruction
    """
    # define the Euler Transformation for Image Rotation
    euler3d = sitk.Euler3DTransform() # define transform for rotation of the image
    image_center = (np.array(image.GetSize())/2.0)  # set rotation center to image center
    image_center_as_sitk_point = image.TransformContinuousIndexToPhysicalPoint(image_center)
    euler3d.SetCenter(image_center_as_sitk_point)

    # get index of volume edges
    w,h,d = image.GetSize()
    extreme_points = [(0, 0, 0), (w, 0, 0), (0, h, 0), (0, 0, d),
                      (w, h, 0), (w, 0, d), (0, h, d), (w, h, d)]
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
    output_direction = tuple(np.eye(3).flatten())
    output_origin = [min_x, min_y, min_z]
    output_size = [int((max_x-min_x)/output_spacing[0]),
                   int((max_y-min_y)/output_spacing[1]),
                   int((max_z-min_z)/output_spacing[2])]

    resampled_image = sitk.Resample(image, output_size, euler3d, interpolator, output_origin, output_spacing, output_direction)
    return resampled_image

def resize(im, new_size=(128, 128, 64), interpolator = sitk.sitkLinear):
    "Resize a sitk.Image to `new_size` maintaining oriantation and direction"
    # get old image meta data
    old_size = im.GetSize()
    old_spacing = im.GetSpacing()
    old_orig = im.GetOrigin()
    old_dir = im.GetDirection()
    old_pixel_id = im.GetPixelIDValue()

    # calculate new spacing
    new_spacing = [spc * (old_sz/new_sz) for spc, old_sz, new_sz in zip(old_spacing, old_size, new_size)]

    # create reference plane to resample image
    ref_image = sitk.Image(new_size, old_pixel_id)
    ref_image.SetSpacing(new_spacing)
    ref_image.SetOrigin(old_orig)
    ref_image.SetDirection(old_dir)

    # resample image to `new_size`
    return sitk.Resample(im, ref_image, interpolator=interpolator)