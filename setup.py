"""Installer for the word-rnn-tensorflow."""

from setuptools import find_packages
from setuptools import setup


setup(
    name='word-rnn-tensorflow',
    version='0.1',
    description=(
        'Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level '
        'language models in Python using TensorFlow.'),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'License :: Other/Proprietary License',
    ],
    url='http://github.com/karantan/word-rnn-tensorflow',
    keywords='tensorflow, text generation',
    license='Proprietary',
    packages=find_packages('src', exclude=['ez_setup']),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'click',
        'h5py',
        'pyyaml',
        'tensorflow',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'pdbpp',
        ],
    },
    # entry_points={
    #     'console_scripts': [
    #         'charmodel-train=train:train',
    #         'charmodel-generate=run:generate_text'
    #     ]
    # },
)
