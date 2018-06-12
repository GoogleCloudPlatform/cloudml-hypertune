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

"""Unit Tests for hypertune.

"""

import json
import os
import shutil
import tempfile
import unittest

from hypertune import HyperTune


class TestHypertune(unittest.TestCase):
    """ TestHypertune class.

    """
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_report_metric(self):
        """ Test writing 1 metric record.

        """
        os.environ['CLOUD_ML_HP_METRIC_FILE'] = os.path.join(self.test_dir, 'metric.output')
        os.environ['CLOUD_ML_TRIAL_ID'] = '1'
        hpt = HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='my_metric_tag',
            metric_value=0.987,
            global_step=1000)
        with open(os.path.join(self.test_dir, 'metric.output')) as metric_output:
            metric = json.loads(metric_output.readlines()[-1].strip())
            self.assertAlmostEqual('0.987', metric['my_metric_tag'])
            self.assertEqual('1', metric['trial'])
            self.assertEqual('1000', metric['global_step'])

    def test_report_metric_circular(self):
        """ Test that the metric file will only store most recent 100 records.

        """
        os.environ['CLOUD_ML_HP_METRIC_FILE'] = os.path.join(self.test_dir, 'metric.output')
        os.environ['CLOUD_ML_TRIAL_ID'] = '1'
        hpt = HyperTune()
        for step in range(0, 200):
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='my_metric_tag',
                metric_value=0.1 * step,
                global_step=step)
        with open(os.path.join(self.test_dir, 'metric.output')) as metric_output:
            metric_content = metric_output.readlines()
            metric = json.loads(metric_content[0].strip())
            self.assertAlmostEqual('10.0', metric['my_metric_tag'])
            self.assertEqual('1', metric['trial'])
            self.assertEqual('100', metric['global_step'])
            metric = json.loads(metric_content[-1].strip())
            self.assertAlmostEqual('19.9', metric['my_metric_tag'])
            self.assertEqual('1', metric['trial'])
            self.assertEqual('199', metric['global_step'])

if __name__ == '__main__':
    unittest.main()
        