import time
import ignite
import yaml
import requests

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

    def get_credentials(self):
        if "pushover_credentials" not in self.config.keys():
            self.logger.warning(
                  "No pushover credentials file submitted, "
                  "will not try to push trainings progress to pushover device. "
                  "If you want to receive status updated via pushover, provide the "
                  "path to a yaml file, containing the `app_token`, `user_key` and `proxies` "
                  "(optional) in the config at `pushover_credentials`"
            )
            return False
        credentials =  self.config.pushover_credentials
        with open(credentials, "r") as stream:
            credentials = yaml.safe_load(stream)
        self.app_token = credentials["app_token"]
        self.user_key = credentials["user_key"]
        self.proxies = credentials["proxies"]
        return True

    def attach(self, engine: ignite.engine.Engine) -> None:
        """
        Args:
            engine: Ignite Engine, should be an evaluator with metrics.
        """
        self.logger = engine.logger
        if self.get_credentials():
            engine.add_event_handler(ignite.engine.Events.STARTED, self.start_training)
            engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.push_metrics)
            engine.add_event_handler(ignite.engine.Events.TERMINATE, self.push_completed)
            engine.add_event_handler(ignite.engine.Events.EXCEPTION_RAISED, self.push_exception)

    def push(self, message: str, priority: int=-1):
        "Send message to device"
        r = requests.post("https://api.pushover.net/1/messages.json",
            data = {
              "token": self.app_token,
              "user": self.user_key,
              "message": message,
              "priority": priority,
              "html": 1, # enable html formatting
            },
            proxies=self.proxies
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
        message += F"Metrics after epoch {epoch}/{self.number_of_epochs}:\n"
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
