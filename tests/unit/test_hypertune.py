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

"""Unit Tests for hypertune."""

import json
import os
import shutil
import tempfile
import unittest

from hypertune import HyperTune


class TestHypertune(unittest.TestCase):
    """TestHypertune class."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self._orig_metric_env = os.environ.get('CLOUD_ML_HP_METRIC_FILE')
        self._orig_trial_env = os.environ.get('CLOUD_ML_TRIAL_ID')
        self._cleanup_dirs = [self.test_dir]

    def tearDown(self):
        if self._orig_metric_env is None:
            os.environ.pop('CLOUD_ML_HP_METRIC_FILE', None)
        else:
            os.environ['CLOUD_ML_HP_METRIC_FILE'] = self._orig_metric_env
        if self._orig_trial_env is None:
            os.environ.pop('CLOUD_ML_TRIAL_ID', None)
        else:
            os.environ['CLOUD_ML_TRIAL_ID'] = self._orig_trial_env
        for d in self._cleanup_dirs:
            shutil.rmtree(d, ignore_errors=True)

    def test_report_metric(self):
        """Test writing 1 metric record."""
        os.environ['CLOUD_ML_HP_METRIC_FILE'] = os.path.join(self.test_dir, 'metric.output')
        os.environ['CLOUD_ML_TRIAL_ID'] = '1'
        hpt = HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='my_metric_tag',
            metric_value=0.987,
            global_step=1000)
        with open(os.path.join(self.test_dir, 'metric.output')) as metric_output:
            metric = json.loads(metric_output.readlines()[-1].strip())
            self.assertAlmostEqual(0.987, float(metric['my_metric_tag']))
            self.assertEqual('1', metric['trial'])
            self.assertEqual('1000', metric['global_step'])

    def test_report_metric_with_ckpt(self):
        """Test writing 1 metric record with checkpoint."""
        os.environ['CLOUD_ML_HP_METRIC_FILE'] = os.path.join(self.test_dir, 'metric.output')
        os.environ['CLOUD_ML_TRIAL_ID'] = '1'
        hpt = HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='my_metric_tag',
            metric_value=0.987,
            global_step=1000,
            checkpoint_path='gs://my_bucket/ckpt/')
        with open(os.path.join(self.test_dir, 'metric.output')) as metric_output:
            metric = json.loads(metric_output.readlines()[-1].strip())
            self.assertAlmostEqual(0.987, float(metric['my_metric_tag']))
            self.assertEqual('1', metric['trial'])
            self.assertEqual('1000', metric['global_step'])
            self.assertEqual('gs://my_bucket/ckpt/', metric['checkpoint_path'])

    def test_report_metric_circular(self):
        """Test that the metric file will only store most recent 100 records."""
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
            self.assertAlmostEqual(10.0, float(metric['my_metric_tag']))
            self.assertEqual('1', metric['trial'])
            self.assertEqual('100', metric['global_step'])
            metric = json.loads(metric_content[-1].strip())
            self.assertAlmostEqual(19.9, float(metric['my_metric_tag']))
            self.assertEqual('1', metric['trial'])
            self.assertEqual('199', metric['global_step'])

    def test_default_path_uses_unique_private_tempdir(self):
        """Default path avoids predictable /tmp/hypertune symlink attacks."""
        os.environ.pop('CLOUD_ML_HP_METRIC_FILE', None)
        hpt = HyperTune()
        parent_dir = os.path.dirname(hpt.metric_path)
        self._cleanup_dirs.append(parent_dir)
        self.assertNotEqual('/tmp/hypertune', parent_dir)
        self.assertEqual('output.metrics', os.path.basename(hpt.metric_path))
        hpt.report_hyperparameter_tuning_metric('loss', 0.25, global_step=5)
        with open(hpt.metric_path) as metric_output:
            metric = json.loads(metric_output.read().strip())
        self.assertAlmostEqual(0.25, float(metric['loss']))

    def test_rejects_symlink_target_file(self):
        """Refuses to clobber a victim file through a symlinked metric path."""
        victim_file = os.path.join(self.test_dir, 'victim_bashrc')
        with open(victim_file, 'w') as f:
            f.write('ORIGINAL_SENSITIVE_CONTENT')
        symlink_metric = os.path.join(self.test_dir, 'output.metrics')
        os.symlink(victim_file, symlink_metric)

        os.environ['CLOUD_ML_HP_METRIC_FILE'] = symlink_metric
        hpt = HyperTune()
        with self.assertRaises(OSError):
            hpt.report_hyperparameter_tuning_metric('m', 1.0)
        with open(victim_file) as f:
            self.assertEqual('ORIGINAL_SENSITIVE_CONTENT', f.read())

    def test_rejects_symlink_parent_directory(self):
        """Refuses parent-directory symlink bypass (/tmp/hypertune -> victim_dir)."""
        victim_dir = os.path.join(self.test_dir, 'victim_dir')
        os.makedirs(victim_dir)
        victim_metrics = os.path.join(victim_dir, 'output.metrics')
        with open(victim_metrics, 'w') as f:
            f.write('VICTIM_METRICS_CONTENT')

        symlink_parent = os.path.join(self.test_dir, 'symlink_hypertune')
        os.symlink(victim_dir, symlink_parent)
        os.environ['CLOUD_ML_HP_METRIC_FILE'] = os.path.join(
            symlink_parent, 'output.metrics'
        )

        with self.assertRaises(OSError):
            HyperTune()
        with open(victim_metrics) as f:
            self.assertEqual('VICTIM_METRICS_CONTENT', f.read())


if __name__ == '__main__':
    unittest.main()
