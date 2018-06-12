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

"""setup.py for hypertune.

"""
from codecs import open

from setuptools import setup


long_description = open('README.rst', 'r', encoding='utf-8').read()


setup(
    name='cloudml-hypertune',
    version='0.1.0.dev4',

    description='A library to report Google CloudML Engine HyperTune metrics.',
    long_description=long_description,

    author='Google CloudML Engine',
    author_email='cloudml-feedback@google.com',
    license='Apache Software License',
    url='http://github.com/GoogleCloudPlatform/cloudml-hypertune',

    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Internet',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Distributed Computing',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',

        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
    ],

    keywords='ml hyperparameter tuning',

    packages=['hypertune'],
)
