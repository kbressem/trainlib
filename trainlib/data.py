"""Build DataLoaders, build datasets, adapt paths, handle CSV files"""

import logging
import shutil
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional

import munch
import pandas as pd
import torch
from monai.data import DataLoader as MonaiDataLoader
from monai.transforms import Compose
from monai.utils import first
from tqdm import tqdm

from trainlib import transforms
from trainlib.utils import num_workers

logger = logging.getLogger(__name__)


def import_dataset(config: munch.Munch):
    if config.data.dataset_type == "persistent":
        from monai.data import PersistentDataset

        cache_dir = Path(config.data.cache_dir)
        if cache_dir.exists():
            shutil.rmtree(str(cache_dir))  # rm previous cache DS
        cache_dir.mkdir(parents=True, exist_ok=True)
        Dataset = partial(PersistentDataset, cache_dir=config.data.cache_dir)  # noqa N806  # noqa N806
    elif config.data.dataset_type == "cache":
        from monai.data import CacheDataset  # noqa F401

        raise NotImplementedError("CacheDataset not yet implemented")
    else:
        from monai.data import Dataset  # type: ignore
    return Dataset


class DataLoader(MonaiDataLoader):
    """Overwrite monai DataLoader for enhanced viewing and debugging capabilities"""

    logger = logger

    def show_batch(
        self,
        image_key: str = "image",
        label_key: str = "label",
        image_transform=lambda x: x.squeeze().transpose(0, 2).flip(-2),
        label_transform=lambda x: x.squeeze().transpose(0, 2).flip(-2),
        **kwargs,
    ):
        """Args:
        image_key: dict key name for image to view
        label_key: dict kex name for corresponding label. Can be a tensor or str
        image_transform: transform input before it is passed to the viewer to ensure
            ndim of the image is equal to 3 and image is oriented correctly
        label_transform: transform labels before passed to the viewer, to ensure
            segmentations masks have same shape and orientations as images. Should be
            identity function of labels are str.
        """
        from trainlib.viewer import ListViewer

        batch = first(self)
        image = batch[image_key]
        label = batch[label_key]
        b, c, w, h, d = image.shape
        if label.shape[1] == 1:
            label = torch.stack([label] * c, 1)
        elif label.shape[1] == c:
            pass
        else:
            raise NotImplementedError(
                f"`show_batch` not implemented for label with {label.shape[0]}" f" channels if image has {c} channels"
            )
        image = torch.unbind(image.reshape(b * c, w, h, d), 0)
        label = torch.unbind(label.reshape(b * c, w, h, d), 0)

        ListViewer(
            [image_transform(im) for im in image],
            [label_transform(im) for im in label],
            **kwargs,
        ).show()

    def sanity_check(self, task: str = "segmentation", sample_size: Optional[int] = None) -> None:
        """Iterate through the dataset and check if transforms are applied without error
        and if the shape and format of the data is correct.

        Args:
            task: The deep learning tasks. Currently only `segmentation` is supported.
            sample_size: Check only the first n items in the data
        """

        data = self.dataset.data  # type: ignore
        transforms = self.dataset.transform  # type: ignore
        if sample_size:
            data = data[:sample_size]

        if task == "segmentation":
            self._sanity_check_segmentation(data, transforms)
        else:
            raise NotImplementedError(f"{task} is not yet implemented")

    def _sanity_check_segmentation(self, data: dict, transforms: Compose) -> None:

        unique_labels: list = []
        for data_dict in tqdm(data):
            try:
                out = transforms(data_dict)
            except Exception as e:
                self.logger.error(f"Exception: {e} raised")
                self.logger.error(data_dict)
            else:
                if not isinstance(out, list):
                    out = [out]
                for item in out:
                    image_fn = item["image"].meta["filename_or_obj"]
                    label_fn = item["label"].meta["filename_or_obj"]
                    image_shape = item["image"].shape
                    label_shape = item["label"].shape
                    if not image_shape == label_shape:
                        self.logger.error(
                            f"Shape missmatch found for {image_fn} ({image_shape})" f" and {label_fn} ({label_shape})"
                        )
                    if max(image_shape) > 1000 or max(label_shape) > 1000:
                        self.logger.warning(
                            "At least one dimension in your image or lables is very large: "
                            f"{image_shape} in file {image_fn} "
                            f"{label_shape} in file {label_fn}"
                        )
                    unique_labels += item["label"].unique().tolist()

        self.logger.info("Frequency of label values:")
        for value in set(unique_labels):
            self.logger.info(f"Value {value} appears in {unique_labels.count(value)} items in the dataset")


def segmentation_dataloaders(
    config: munch.Munch,
    train: Optional[bool] = None,
    valid: Optional[bool] = None,
    test: Optional[bool] = None,
):
    """Create segmentation dataloaders
    Args:
        config: config file
        train: whether to return a train DataLoader
        valid: whether to return a valid DataLoader
        test: whether to return a test DateLoader
    Args from config:
        data_dir: base directory for the data
        csv_name: path to csv file containing filenames and paths
        image_cols: columns in csv containing path to images
        label_cols: columns in csv containing path to label files
        dataset_type: PersistentDataset, CacheDataset and Dataset are supported
        cache_dir: cache directory to be used by PersistentDataset
        batch_size: batch size for training. Valid and test are always 1
        debug: run with reduced number of images
    Returns:
        list of:
            train_loader: DataLoader (optional, if train==True)
            valid_loader: DataLoader (optional, if valid==True)
            test_loader: DataLoader (optional, if test==True)
    """

    # parse needed rguments from config
    if train is None:
        train = config.data.train
    if valid is None:
        valid = config.data.valid
    if test is None:
        test = config.data.test

    data_dir = config.data.data_dir
    train_csv = config.data.train_csv
    valid_csv = config.data.valid_csv
    test_csv = config.data.test_csv
    image_cols = config.data.image_cols
    label_cols = config.data.label_cols
    batch_size = config.data.batch_size
    debug = config.debug

    # ---------- data dicts ----------
    # first a global data dict, containing only the filepath from image_cols
    # and label_cols is created. For this, the dataframe is reduced to only
    # the relevant columns. Then the rows are iterated, converting each row
    # into an individual dict, as expected by monai

    if not isinstance(image_cols, (tuple, list)):
        image_cols = [image_cols]
    if not isinstance(label_cols, (tuple, list)):
        label_cols = [label_cols]
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)

    if debug:
        train_df = train_df.sample(10)
        valid_df = valid_df.sample(5)

    # Assemble columns that contain a path
    path_cols = deepcopy(image_cols)
    for col in label_cols:
        try:
            path = train_df[col][0]
            path = Path(data_dir).resolve() / path
        except (KeyError, TypeError):
            continue
        if path.exists():
            path_cols.append(col)

    for col in path_cols:
        # create absolute file name from relative fn in df and data_dir
        train_df[col] = data_dir / train_df[col]
        valid_df[col] = data_dir / valid_df[col]
        test_df[col] = data_dir / test_df[col]

    # Dataframes should now be converted to a dict
    # The data_dict looks like this:
    # [
    #  {'image_col_1': 'data_dir/path/to/image1',
    #   'image_col_2': 'data_dir/path/to/image2'
    #   'label_col_1': 'data_dir/path/to/label1},
    #  {'image_col_1': 'data_dir/path/to/image1',
    #   'image_col_2': 'data_dir/path/to/image2'
    #   'label_col_1': 'data_dir/path/to/label1},
    #    ...]
    # Filename should now be absolute or relative to working directory

    # now we create separate data dicts for train, valid and test data respectively
    assert train or test or valid, "No dataset type is specified (train/valid or test)"
    if test:
        test_files = test_df.to_dict("records")
    if valid:
        val_files = valid_df.to_dict("records")
    if train:
        train_files = train_df.to_dict("records")

    # transforms are specified in transforms.py and are just loaded here
    if train:
        train_transforms = transforms.get_train_transforms(config)
    if valid:
        val_transforms = transforms.get_val_transforms(config)
    if test:
        test_transforms = transforms.get_test_transforms(config)

    # ---------- construct dataloaders ----------
    Dataset = import_dataset(config)  # noqa N806
    data_loaders = []
    if train:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers(),
            shuffle=True,
        )
        data_loaders.append(train_loader)

    if valid:
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers(), shuffle=False)
        data_loaders.append(val_loader)

    if test:
        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers(), shuffle=False)
        data_loaders.append(test_loader)

    # if only one dataloader is constructed, return only this dataloader else return a named tuple
    # with dataloaders, so it is clear which DataLoader is train/valid or test

    if len(data_loaders) == 1:
        return data_loaders[0]
    else:
        DataLoaders = namedtuple(  # type: ignore
            "DataLoaders",
            # create str with specification of loader type if train and test are true but
            # valid is false string will be 'train test'
            " ".join(["train" if train else "", "valid" if valid else "", "test" if test else ""]).strip(),
        )
        return DataLoaders(*data_loaders)
