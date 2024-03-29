import logging
import time
from typing import List, Optional

import ignite
import munch
import requests
import torch
import yaml
from monai.utils.type_conversion import convert_to_tensor

from trainlib.utils import get_n_classes_of_model_from_config

logger = logging.getLogger(__name__)


class PushnotificationHandler:
    """Send push notifications with pushover for remote monitoring
    Args:
        config: Global configuration file. Should contain `pushover_credentials` entry,
            where the path to a YAML file is supplied, containing the `app_token`,
            `user_key` and, if you are behind a proxy `proxies`.
        For more information on pushover visit: https://support.pushover.net/
    """

    def __init__(self, config: munch.Munch) -> None:
        self.config = config
        self.logger = logger
        if "pushover_credentials" not in self.config.keys():
            self.logger.warning(
                "No pushover credentials file submitted, "
                "will not try to push trainings progress to pushover device. "
                "If you want to receive status updated via pushover, provide the "
                "path to a yaml file, containing the `app_token`, `user_key` and `proxies` "
                "(optional) in the config at `pushover_credentials`"
            )
            self.enable_notifications = False
        elif config.debug:
            # No notifications in debug mode
            self.enable_notifications = False
        else:
            credentials = self.config.pushover_credentials
            with open(credentials, "r") as stream:
                credentials = yaml.safe_load(stream)
            self.app_token = credentials["app_token"]
            self.user_key = credentials["user_key"]
            self.proxies = credentials["proxies"] if "proxies" in credentials else None
            self.enable_notifications = True

        self.key_metric = -1
        self.improvement = False

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        if self.enable_notifications:
            engine.add_event_handler(ignite.engine.Events.STARTED, self.start_training)
            engine.add_event_handler(ignite.engine.Events.COMPLETED, self.push_metrics)
            engine.add_event_handler(ignite.engine.Events.TERMINATE, self.push_terminated)
            engine.add_event_handler(ignite.engine.Events.EXCEPTION_RAISED, self.push_exception)

    def push(self, message: str, priority: int = -1):
        "Send message to device"
        _ = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.app_token,
                "user": self.user_key,
                "message": message,
                "priority": priority,
                "html": 1,  # enable html formatting
            },
            proxies=self.proxies,
        )

    def _get_metrics(self, engine: ignite.engine.Engine) -> str:
        "Extract metrics from engine.state"
        message = ""
        metric_names = list(engine.state.metrics.keys())

        key_metric = engine.state.metrics[metric_names[0]]
        self.improvement = key_metric > self.key_metric
        self.key_metric = max(key_metric, self.key_metric)

        for mn in metric_names:
            message += f"{mn}: {engine.state.metrics[mn]:.5f}\n"
        return message

    def start_training(self, engine: ignite.engine.Engine) -> None:
        "Collect basic data for run"
        self.start_time = time.time()
        self.number_of_epochs = self.config.training.max_epochs
        self.run_id = self.config.run_id

    def push_metrics(self, engine: ignite.engine.Engine) -> None:
        epoch = engine.state.epoch
        message = f"<b>{self.run_id}:</b>\n"
        message += f"Metrics after epoch {epoch}/{self.number_of_epochs}:\n"
        message += self._get_metrics(engine)
        if self.improvement:
            self.push(message)

    def push_terminated(self, engine: ignite.engine.Engine) -> None:
        end_time = time.time()
        seconds = self.start_time - end_time
        minutes = seconds // 60
        hours = minutes // 60
        minutes = minutes % 60
        duration = f"{hours}:{minutes}:{seconds}"
        epoch = engine.state.epoch
        message = f"<b>{self.run_id}:</b>\n"
        message += f"Training ended {epoch}/{self.number_of_epochs} epochs\n"
        message += f"Duration {duration} \n"
        message += self._get_metrics(engine)
        self.push(message)

    def push_exception(self, engine: ignite.engine.Engine) -> None:
        epoch = engine.state.epoch
        message = f"<b>{self.run_id}:</b>\n"
        message += f"Exception raise after {epoch}/{self.number_of_epochs} epochs\n"
        self.push(message, 0)


class DebugHandler:
    "Send summary statistics about batch as debugging information to engine logger"

    def __init__(self, config: munch.Munch) -> None:
        self.config = config
        self.debug_on = self.config.debug
        self.logger = logger

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        if self.debug_on:
            engine.add_event_handler(ignite.engine.Events.GET_BATCH_COMPLETED, self.batch_statistics)
            engine.add_event_handler(ignite.engine.Events.GET_BATCH_COMPLETED, self.check_loss_and_n_classes)

    def batch_statistics(self, engine: ignite.engine.Engine) -> None:
        image_keys = self.config.data.image_cols
        label_keys = self.config.data.label_cols
        keys = image_keys + label_keys
        # If multiple images/labels are used, they are concatenated at the end of transforms
        # new labels CommonKeys.LABEL (`label`) and CommonKeys.IMAGE (`image`)
        if "image" not in keys:
            keys.append("image")
        if "label" not in keys:
            keys.append("label")

        message: str = self._table_row()
        for key in keys:
            for items in engine.state.batch[key]:  # type: ignore
                items = convert_to_tensor(items)
                message += self._table_row([key] + self._extract_statisics(items))

        self.logger.info("\nBatch Statistics:")
        self.logger.info(message + "\n")

    def check_loss_and_n_classes(self, engine: ignite.engine.Engine):

        n_classes = get_n_classes_of_model_from_config(self.config)

        if n_classes is None:
            self.logger.info(
                "Cannot derive number of classes from model. "
                "Therefore cannot check if model output fits to loss function"
            )
            return
        labels = convert_to_tensor(engine.state.batch["label"])  # type: ignore
        unique = torch.unique(labels)
        if len(unique) > n_classes:
            self.logger.error(
                "There are more unique values in the labels than there are `out_channels`. "
                f"Found {len(unique)} but expected {n_classes} or less"
            )
        if max(unique) > n_classes:
            self.logger.error(
                "The maximum value of labels is higher than `out_channels`. "
                "This will lead to issues with one-hot conversion and/or your loss function. "
                f"Found max value of {max(unique)} but expected {n_classes}"
            )

    def _extract_statisics(self, x: torch.Tensor) -> List:
        x = x.float()
        shape = tuple(x.shape)
        mean = torch.mean(x).item()
        std = torch.std(x).item()
        min = torch.min(x).item()
        max = torch.max(x).item()
        unique = len(torch.unique(x))
        return [shape, mean, std, min, max, unique]

    def _table_row(self, items: Optional[List] = None) -> str:
        "Create table row with colwidth of 18 and colspacing of 2"
        if items is None:  # print header
            items = ["item", "shape", "mean", "std", "min", "max", "unique val"]
        items = [str(i)[:18] for i in items]
        format_row = "{:>20}" * (len(items) + 1)
        return "\n" + format_row.format("", *items)


class EnsureTensor:
    """In ClassificationTrainer labels are converted from torch.Tensor -> int.
    This leads to multiple downstream problems. This handler is a (temporary) workaround,
    that forces labels to always be torch.Tensor.
    """

    def __init__(self, config: munch.Munch) -> None:
        self.config = config

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self.ensure_tensor)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.ensure_tensor)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.ensure_tensor)

    def ensure_tensor(self, engine: ignite.engine.Engine) -> None:
        "Forces label to be torch.Tensor"
        label = engine.state.output[0]["label"]  # type: ignore
        if not isinstance(label, torch.Tensor):
            if isinstance(label, int):
                label = [label]
            engine.state.output[0]["label"] = torch.tensor(label).to(self.config.device)  # type: ignore

        label = engine.state.batch[0]["label"]  # type: ignore
        if not isinstance(label, torch.Tensor):
            if isinstance(label, int):
                label = [label]
            engine.state.batch[0]["label"] = torch.tensor(label).to(self.config.device)  # type: ignore
