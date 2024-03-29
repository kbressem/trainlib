"""Build DataLoaders, build datasets, adapt paths, handle CSV files"""

import logging
import shutil
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import munch
import pandas as pd
import torch
from monai.data import DataLoader as MonaiDataLoader
from monai.transforms import Compose
from monai.utils import ensure_tuple
from tqdm import tqdm

from trainlib import transforms
from trainlib.utils import num_workers
from trainlib.viewer import ListViewer

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


def _default_image_transform_3d(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze().transpose(0, 2).flip(-2)


def _default_image_transform_2d(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze().transpose(-2, -1)


def _resolve_if_exists(
    data_dir: Path, filename: Union[str, Path], warn_if_nonexistent: bool = False
) -> Union[str, Path]:
    full_fn = data_dir / str(filename)
    if full_fn.exists():
        return full_fn
    elif warn_if_nonexistent:
        logger.warn(f"{str(full_fn)} does not exist")
    return filename


def _resolve_column(df, col_name, data_dir, warn_if_nonexistent):
    df[col_name] = [_resolve_if_exists(data_dir, fn, warn_if_nonexistent) for fn in df[col_name]]
    return df


class DataLoader(MonaiDataLoader):
    """Overwrite monai DataLoader for enhanced viewing and debugging capabilities"""

    logger = logger
    start: int = 0  # first item for `show_batch`

    def __init__(self, dataset, num_workers, task, **kwargs):
        super().__init__(dataset, num_workers, **kwargs)
        self.task = task

    def show_batch(
        self,
        image_key: str = "image",
        label_key: str = "label",
        image_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        mode: Optional[str] = None,
        **kwargs,
    ):
        """Args:
        image_key: dict key name for image to view
        label_key: dict kex name for corresponding label. Can be a tensor or str
        image_transform: transform input before it is passed to the viewer.
            If None a default transform is applied to ensure dimensionality and orientation of the image are correct.
        label_transform: transform labels before passed to the viewer.
            If None a default transform is applied tp ensure dimensionality and orientation are correct.
        mode: If `mode = 'RGB'` channel-dim will be treated as colors. Otherwise each channels is
            displayed individually.
        """

        # mypy does not recognize that data/transforms are in dataset
        n_items: int = self.dataset.data.__len__()  # type: ignore
        batch_size: int = self.batch_size  # type: ignore
        transforms = self.dataset.transform  # type: ignore

        self.start = self.start + batch_size if (self.start + batch_size) < n_items else 0
        data = self.dataset.data[self.start : (self.start + batch_size)]  # type: ignore # noqa E203

        batch = [transforms(item) for item in data]
        image = torch.stack([item[image_key] for item in batch], 0)
        label = torch.stack([item[label_key] for item in batch], 0)

        b, c, *wh_d = image.shape
        ndim = len(wh_d)

        if image_transform is None:
            image_transform = globals()[f"_default_image_transform_{ndim}d"]

        if label_transform is None and isinstance(label, torch.Tensor):
            label_transform = globals()[f"_default_image_transform_{ndim}d"]

        elif label.shape[1] == 1 and mode != "RGB":
            label = torch.stack([label] * c, 1)
        elif label.shape[1] == c or (label.shape[1] == 1 and c == 3 and mode == "RGB"):
            pass
        else:
            raise NotImplementedError(
                f"`show_batch` not implemented for label with {label.shape[0]} channels if image has {c} channels"
            )

        if mode != "RGB":
            image = image.reshape(b * c, *wh_d)
            label = label.reshape(b * c, *wh_d)

        image_list = torch.unbind(image, 0)
        label_list = torch.unbind(label, 0)

        ListViewer(
            [image_transform(im) for im in image_list],
            [label_transform(im) for im in label_list],  # type: ignore
            mode=mode,
            **kwargs,
        ).show()

    def sanity_check(self, sample_size: Optional[int] = None) -> None:
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

        if self.task == "segmentation":
            self._sanity_check_segmentation(data, transforms)
        else:
            self._sanity_check_classification(data, transforms)

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

    def _sanity_check_classification(self, data: dict, transforms: Compose) -> None:

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
                    image_shape = item["image"].shape
                    if max(image_shape) > 1000:
                        self.logger.warning(
                            "At least one dimension in your image or lables is very large: "
                            f"{image_shape} in file {image_fn} "
                        )
                    unique_labels.append(item["label"])

        self.logger.info("Frequency of label values:")
        for value in set(unique_labels):
            self.logger.info(f"Value {value} appears in {unique_labels.count(value)} items in the dataset")


def dataloaders(
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

    # parse needed arguments from config
    splits = {"train": train or config.data.train, "valid": config.data.valid, "test": test or config.data.test}
    data_dir = config.data.data_dir
    csv_names = [config.data.get(f"{key}_csv") for key, value in splits.items() if value]
    if not csv_names:
        raise ValueError("No dataset type is specified (train, valid or test)")

    image_cols = ensure_tuple(config.data.image_cols)
    label_cols = ensure_tuple(config.data.label_cols)
    batch_size = config.data.batch_size
    debug = config.debug

    # ---------- data dicts ----------
    # first a global data dict, containing only the filepath from image_cols
    # and label_cols is created. For this, the dataframe is reduced to only
    # the relevant columns. Then the rows are iterated, converting each row
    # into an individual dict, as expected by monai

    data_frames = [pd.read_csv(fn) for fn in csv_names]

    if debug:
        data_frames = [df.sample(5) for df in data_frames]

    for col in image_cols + label_cols:
        # create absolute file name from relative fn in df and data_dir
        warn_if_nonexistent = col in image_cols or config.task == "segmentation"
        data_frames = [_resolve_column(df, col, data_dir, warn_if_nonexistent) for df in data_frames]

    # Dataframes should now be converted to a dict
    # For a segmentation problem, the data_dict would look like this:
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
    data_dicts = [df.to_dict("records") for df in data_frames]

    # transforms are specified in transforms.py and are just loaded here
    data_transforms = [getattr(transforms, f"get_{key}_transforms")(config) for key, value in splits.items() if value]

    # ---------- construct dataloaders ----------
    Dataset = import_dataset(config)  # noqa N806
    data_loaders = []

    for data_dict, transform in zip(data_dicts, data_transforms):
        data_set = Dataset(data=data_dict, transform=transform)
        data_loader = DataLoader(
            data_set, batch_size=batch_size, num_workers=num_workers(), shuffle=True, task=config.task
        )
        data_loaders.append(data_loader)

    # if only one dataloader is constructed, return only this dataloader else return a named tuple
    # with dataloaders, so it is clear which DataLoader is train/valid or test

    if len(data_loaders) == 1:
        return data_loaders[0]
    else:
        DataLoaders = namedtuple(  # type: ignore
            "DataLoaders",
            # create str with specification of loader type if train and test are true but
            # valid is false string will be 'train test'
            [key for key, value in splits.items() if value],
        )
        return DataLoaders(*data_loaders)
