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

"""HyperTune class to work with Google CloudML Engine Hyperparameter tuning."""

import collections
import errno
import json
import os
import stat
import tempfile
import time

_DEFAULT_HYPERPARAMETER_METRIC_TAG = 'training/hptuning/metric'
_DEFAULT_METRIC_FILENAME = 'output.metrics'
_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE = 100
_MACOS_ROOT_SYMLINKS = frozenset(('/tmp', '/var', '/etc'))


def _ensure_safe_parent_dir(metric_path):
    """Ensures parent directories exist without traversing untrusted symlinks."""
    abs_path = os.path.abspath(metric_path)
    parent_dir = os.path.dirname(abs_path)
    parts = [p for p in parent_dir.split(os.sep) if p]
    current = os.sep
    for part in parts:
        current = os.path.join(current, part)
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o700)
            st = os.lstat(current)
        if stat.S_ISLNK(st.st_mode):
            if current in _MACOS_ROOT_SYMLINKS and st.st_uid == 0:
                continue
            raise OSError(
                errno.ELOOP,
                'Refusing to traverse symlink directory component',
                current,
            )
        if not stat.S_ISDIR(st.st_mode):
            raise OSError(
                errno.ENOTDIR,
                'Metric path component is not a directory',
                current,
            )
    return parent_dir, os.path.basename(abs_path)


def _open_metric_file_safely(metric_path):
    """Opens metric_path for writing without following symlinks."""
    parent_dir, filename = _ensure_safe_parent_dir(metric_path)
    if not filename:
        raise ValueError(f'Invalid metric file path: {metric_path!r}')

    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
        return open(os.path.join(parent_dir, filename), 'w', encoding='utf-8')

    dir_fd = os.open(parent_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        dir_stat = os.fstat(dir_fd)
        if hasattr(os, 'geteuid') and dir_stat.st_uid not in (os.geteuid(), 0):
            raise PermissionError(
                f'Refusing to write metrics into directory owned by uid '
                f'{dir_stat.st_uid}: {parent_dir}'
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        fd = os.open(filename, flags, 0o600, dir_fd=dir_fd)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(
                    errno.EINVAL,
                    'Refusing to write metrics to non-regular file',
                    metric_path,
                )
            if hasattr(os, 'geteuid') and file_stat.st_uid != os.geteuid():
                raise PermissionError(
                    f'Refusing to overwrite metric file owned by uid '
                    f'{file_stat.st_uid}: {metric_path}'
                )
        except Exception:
            os.close(fd)
            raise
    finally:
        os.close(dir_fd)

    return os.fdopen(fd, 'w', encoding='utf-8')


class HyperTune(object):
    """Main class for HyperTune."""

    def __init__(self):
        """Constructor."""
        env_metric_path = os.environ.get('CLOUD_ML_HP_METRIC_FILE')
        if env_metric_path:
            self.metric_path = env_metric_path
            _ensure_safe_parent_dir(self.metric_path)
        else:
            temp_dir = tempfile.mkdtemp(prefix='hypertune_')
            self.metric_path = os.path.join(temp_dir, _DEFAULT_METRIC_FILENAME)

        self.trial_id = os.environ.get('CLOUD_ML_TRIAL_ID', 0)
        self.metrics_queue = collections.deque(
            maxlen=_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE)

    def _dump_metrics_to_file(self):
        with _open_metric_file_safely(self.metric_path) as metric_file:
            for metric in self.metrics_queue:
                metric_file.write(json.dumps(metric, sort_keys=True) + '\n')

    def report_hyperparameter_tuning_metric(self,
                                            hyperparameter_metric_tag,
                                            metric_value,
                                            global_step=None,
                                            checkpoint_path=''):
        """Method to report hyperparameter tuning metric.

        Args:
          hyperparameter_metric_tag: The hyperparameter metric name this metric
            value is associated with. Should keep consistent with the tag
            specified in HyperparameterSpec.
          metric_value: float, the values for the hyperparameter metric to report.
          global_step: int, the global step this metric value is associated with.
          checkpoint_path: The checkpoint path which can be used to warmstart from.
        """
        metric_value = float(metric_value)
        metric_tag = _DEFAULT_HYPERPARAMETER_METRIC_TAG
        if hyperparameter_metric_tag:
            metric_tag = hyperparameter_metric_tag
        metric_body = {
            'timestamp': time.time(),
            'trial': str(self.trial_id),
            metric_tag: str(metric_value),
            'global_step': str(int(global_step) if global_step else 0),
            'checkpoint_path': checkpoint_path
        }
        self.metrics_queue.append(metric_body)
        self._dump_metrics_to_file()
