# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HyperTune class to work with Google CloudML Engine Hyperparameter tuning.

"""

import collections
import json
import os
import time

_DEFAULT_HYPERPARAMETER_METRIC_TAG = 'training/hptuning/metric'
_DEFAULT_METRIC_PATH = '/tmp/hypertune/output.metrics'
# TODO(0olwzo0): consider to make it configurable
_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE = 100


class HyperTune(object):
    """Main class for HyperTune."""

    def __init__(self):
        """Constructor."""
        self.metric_path = os.environ.get('CLOUD_ML_HP_METRIC_FILE',
                                          _DEFAULT_METRIC_PATH)
        if not os.path.exists(os.path.dirname(self.metric_path)):
            os.makedirs(os.path.dirname(self.metric_path))

        self.trial_id = os.environ.get('CLOUD_ML_TRIAL_ID', 0)
        self.metrics_queue = collections.deque(
            maxlen=_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE)

    def _dump_metrics_to_file(self):
        with open(self.metric_path, 'w') as metric_file:
            for metric in self.metrics_queue:
                metric_file.write(json.dumps(metric, sort_keys=True) + '\n')

    def report_hyperparameter_tuning_metric(self,
                                            hyperparameter_metric_tag,
                                            metric_value,
                                            global_step=None):
        """Method to report hyperparameter tuning metric.

        Args:
          hyperparameter_metric_tag: The hyperparameter metric name this metric
            value is associated with. Should keep consistent with the tag
            specified in HyperparameterSpec.
          metric_value: float, the values for the hyperparameter metric to report.
          global_step: int, the global step this metric value is associated with.
        """
        metric_value = float(metric_value)
        metric_tag = _DEFAULT_HYPERPARAMETER_METRIC_TAG
        if hyperparameter_metric_tag:
            metric_tag = hyperparameter_metric_tag
        metric_body = {
            'timestamp': time.time(),
            'trial': str(self.trial_id),
            metric_tag: str(metric_value),
            'global_step': str(int(global_step) if global_step else 0)
        }
        self.metrics_queue.append(metric_body)
        self._dump_metrics_to_file()
