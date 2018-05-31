cloudml-hypertune - Helper Functions for CloudML Engine Hypertune Services.
===========================================================================

``cloudml-hypertune`` provides functionalities to report metrics for `Google CloudML
Engine Hyperparameter Tuning Service <https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview>`__.

Installation
------------

Install via `pip <https://pypi.python.org/pypi/pip>`__:

::

    pip install cloudml-hypertune


Prerequisites
-------------

-  Google CloudML Engine `Hyperparameter Tuning
   Overview <https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview>`__.


Usage
-----

.. code:: python

    import hypertune

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='my_metric_tag',
        metric_value=0.987,
        global_step=1000)

By default, the metric entries will be stored to ``/var/hypertune/outout.metric`` in json format:

::

    {"global_step": "1000", "my_metric_tag": "0.987", "timestamp": 1525851440.123456, "trial": "0"}


Licensing
---------

- Apache 2.0
