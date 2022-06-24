import importlib
import multiprocessing
import os
import resource
import sys
from pathlib import Path
from typing import Callable, Union

import monai
import munch
import psutil
import yaml

USE_AMP = monai.utils.get_torch_version_tuple() >= (1, 6)


def load_config(fn: str = "config.yaml"):
    "Load config from YAML and return a serialized dictionary object"
    with open(fn, "r") as stream:
        config = yaml.safe_load(stream)
    config = munch.munchify(config)

    if not config.overwrite:
        i = 1
        while os.path.exists(config.run_id + f"_{i}"):
            i += 1
        config.run_id += f"_{i}"

    config.out_dir = os.path.join(config.run_id, config.out_dir)
    config.log_dir = os.path.join(config.run_id, config.log_dir)

    if not isinstance(config.data.image_cols, (tuple, list)):
        config.data.image_cols = [config.data.image_cols]
    if not isinstance(config.data.label_cols, (tuple, list)):
        config.data.label_cols = [config.data.label_cols]

    config.transforms.mode = ("bilinear",) * len(config.data.image_cols) + (
        "nearest",
    ) * len(config.data.label_cols)
    return config


def num_workers():
    "Get max supported workers -2 for multiprocessing"

    n_workers = (
        multiprocessing.cpu_count() - 2
    )  # leave two workers so machine can still respond

    # check if we will run into OOM errors because of too many workers
    # In most projects 2-4GB/Worker seems to be save
    available_ram_in_gb = psutil.virtual_memory()[0] / 1024**3
    max_workers = int(available_ram_in_gb // 4)
    if max_workers < n_workers:
        n_workers = max_workers

    # now check for max number of open files allowed on system
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # giving each worker at least 256 open processes should allow them to run smoothly
    max_workers = soft_limit // 256

    if max_workers < n_workers:
        print(
            "Will not use all available workers as number of allowed open files is to small"
            "to ensure smooth multiprocessing. Current limits are:\n"
            f"\t soft_limit: {soft_limit}\n"
            f"\t hard_limit: {hard_limit}\n"
            "try increasing the limits to at least {256*n_workers}."
            "See https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past"
            "-4096-ubuntu for more details"
        )
        n_workers = max_workers

    return n_workers


def import_patched(path: Union[str, Path], name: str) -> Callable:
    """Import function `name` from `path`, used to inject custom, project specific
    code into `trainlib` without changing the main codebase.
    E.g., if one wants to change the model, instead of editing trainlib/model.py,
    one can provide a path to another model in the config at: config.patch.model
    `trainlib` will then try to first import from the given patched model and, if this
    failes, fall to import from trainlib/model.py.

    Args:
        path: Path to python script, containing the patched functionality.
        name: Name of function to import
    """
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    module = path.name.replace(".py", "")
    patch = importlib.import_module(module)
    return getattr(patch, name)