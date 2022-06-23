import time
import ignite
import yaml
import requests
from typing import Dict, List

import torch
from monai.utils.type_conversion import convert_to_tensor


class PushnotificationHandler:
    """Send push notifications with pushover for remote monitoring
    Args:
        config: Global configuration file. Should contain `pushover_credentials` entry,
            where the path to a YAML file is supplied, containing the `app_token`,
            `user_key` and, if you are behind a proxy `proxies`.
        For more information on pushover visit: https://support.pushover.net/
    """

    def __init__(self, config: dict) -> None:
        self.config = config

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
            self.proxies = credentials["proxies"]
            self.enable_notifications = True

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        self.logger = engine.logger
        if self.enable_notifications:
            engine.add_event_handler(ignite.engine.Events.STARTED, self.start_training)
            engine.add_event_handler(
                ignite.engine.Events.EPOCH_COMPLETED, self.push_metrics
            )
            engine.add_event_handler(
                ignite.engine.Events.TERMINATE, self.push_completed
            )
            engine.add_event_handler(
                ignite.engine.Events.EXCEPTION_RAISED, self.push_exception
            )

    def push(self, message: str, priority: int = -1):
        "Send message to device"
        r = requests.post(
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

    def _get_metrics(self, engine: ignite.engine.Engine) -> None:
        "Extract metrics from engine.state"
        message = ""
        metric_names = list(engine.state.metrics.keys())
        for mn in metric_names:
            message += f"{mn}: {engine.state.metrics[mn]}\n"
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
        self.push(message)

    def push_completed(self, engine: ignite.engine.Engine) -> None:
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

    def __init__(self, config: dict) -> None:
        self.config = config
        self.debug_on = self.config.debug

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        self.logger = engine.logger
        if self.debug_on:
            engine.add_event_handler(
                ignite.engine.Events.GET_BATCH_COMPLETED, self.batch_statistics
            )

    def batch_statistics(self, engine: ignite.engine.Engine) -> None:
        image_keys = self.config.data.image_cols
        label_keys = self.config.data.label_cols

        message: str = self._table_row()
        for key in image_keys + label_keys + ["image", "label"]:
            for items in engine.state.batch[key]:
                items = convert_to_tensor(items)
                message += self._table_row([key] + self._extract_statisics(items))

        self.logger.info("\nBatch Statistics:")
        self.logger.info(message + "\n")

    def _extract_statisics(self, x: torch.Tensor) -> List:
        shape = tuple(x.shape)
        mean = torch.mean(x).item()
        std = torch.std(x).item()
        min = torch.min(x).item()
        max = torch.max(x).item()
        unique = len(torch.unique(x))
        return [shape, mean, std, min, max, unique]

    def _table_row(self, items: List = None) -> None:
        "Create table row with colwidth of 18 and colspacing of 2"
        if items is None:  # print header
            items = ["item", "shape", "mean", "std", "min", "max", "unique val"]
        items = [str(i)[:18] for i in items]
        format_row = "{:>20}" * (len(items) + 1)
        return "\n" + format_row.format("", *items)
