import tempfile
import unittest

import yaml
from ignite.engine import Events
from test_utils import TEST_CONFIG, TEST_ENGINE, TEST_IMAGE, TEST_LABEL

from trainlib.handlers import DebugHandler, PushnotificationHandler


class TestPushNotificationHandler(unittest.TestCase):
    config = TEST_CONFIG
    engine = TEST_ENGINE

    def test_init(self):
        # without credentials
        try:
            self.config.pop("pushover_credentials")
        except AttributeError:
            pass

        with self.assertLogs("trainlib", level="WARNING") as _:
            handler = PushnotificationHandler(self.config)
            self.assertFalse(handler.enable_notifications)

    def test_attach(self):
        self.config.debug = False
        with tempfile.TemporaryDirectory() as tempdir:
            credentials = {"app_token": "foo", "user_key": "bar"}
            with open(f"{tempdir}/credentials.yaml", "w+") as outfile:
                yaml.dump(credentials, outfile, default_flow_style=False)
                self.config.pushover_credentials = f"{tempdir}/credentials.yaml"
                handler = PushnotificationHandler(self.config)
        self.assertTrue(handler.enable_notifications)
        handler.attach(self.engine)

        self.assertTrue(self.engine.has_event_handler(handler.push_metrics, Events.COMPLETED))
        self.assertTrue(self.engine.has_event_handler(handler.start_training, Events.STARTED))
        self.assertTrue(self.engine.has_event_handler(handler.push_terminated, Events.TERMINATE))
        self.assertTrue(
            self.engine.has_event_handler(
                handler.push_exception,
                Events.EXCEPTION_RAISED,
            )
        )


class TestDebugHandler(unittest.TestCase):
    config = TEST_CONFIG
    engine = TEST_ENGINE
    image = TEST_IMAGE
    label = TEST_LABEL

    def _prepare_engine_state(self):
        self.engine.state.batch = {"image": self.image, "label": self.label}

    def test_init(self):
        self.config.debug = False
        handler = DebugHandler(self.config)
        self.assertFalse(handler.debug_on)

        self.config.debug = True
        handler = DebugHandler(self.config)
        self.assertTrue(handler.debug_on)

    def test_attach(self):
        self.config.debug = True
        handler = DebugHandler(self.config)
        handler.attach(self.engine)
        self.assertTrue(self.engine.has_event_handler(handler.batch_statistics, Events.GET_BATCH_COMPLETED))
        self.assertTrue(self.engine.has_event_handler(handler.check_loss_and_n_classes, Events.GET_BATCH_COMPLETED))

    def test_check_loss_and_n_classes(self):
        self.config.debug = True
        self._prepare_engine_state()
        # out_channels and number of classes do not fit
        handler = DebugHandler(self.config)
        with self.assertLogs("trainlib", level="ERROR") as cm:
            handler.check_loss_and_n_classes(self.engine)
            self.assertTrue(
                "There are more unique values in the labels than there are `out_channels`." in str(cm.output)
            )

        # value of classes do not fit -> will make problems with one-hot conversion
        self.config.model.out_classes = 3
        self.engine.state.batch["label"] *= 2
        with self.assertLogs("trainlib", level="ERROR") as cm:
            handler.check_loss_and_n_classes(self.engine)
            self.assertTrue("The maximum value of labels is higher than `out_channels`." in str(cm.output))

    def test_batch_statistics(self):
        self.config.debug = True
        self.config.model.out_classes = 3
        self._prepare_engine_state()
        # out_channels and number of classes do not fit
        handler = DebugHandler(self.config)
        with self.assertLogs("trainlib", level="INFO") as cm:
            handler.batch_statistics(self.engine)
            keys_or_metrics = [
                "image",
                "label",
                "item",
                "shape",
                "mean",
                "std",
                "min",
                "max",
                "unique val",
            ]
            for km in keys_or_metrics:
                self.assertTrue(km in str(cm.output), f"{km} not in logging message")


if __name__ == "__main__":
    unittest.main()
