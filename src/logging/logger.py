"""
logger.py

PyTorch Lightning Logger for logging plaintext/JSON events to file (for evaluation/later visualizations).
"""
import json
import os

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


class MetricLogger(LightningLoggerBase):
    def __init__(self, name, save_dir):
        super(MetricLogger, self).__init__()
        self._name, self._save_dir = name, os.path.join(save_dir, "metrics")

        # Create Massive Dictionary to JSONify
        self.events = {}

    @property
    def name(self):
        return self._name

    @property
    def experiment(self):
        return None

    @property
    def version(self):
        return 1.0

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        self.events["hyperparams"] = vars(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        for metric in metrics:
            if metric in self.events:
                self.events[metric].append(metrics[metric])
                self.events["%s_step" % metric].append(step)
            else:
                self.events[metric] = [metrics[metric]]
                self.events["%s_step" % metric] = [step]

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        self.events["status"] = status

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        with open(os.path.join(self._save_dir, "%s-metrics.json" % self._name), "w") as f:
            json.dump(self.events, f, indent=4)
