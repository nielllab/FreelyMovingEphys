"""
setup.py

Setup file for fmEphys package.

Written by DMM, 2022
"""


import setuptools


setuptools.setup(
    name = 'fmEphys',
    packages = setuptools.find_packages(),
    description = 'Analysis for electrophysiology in freely moving mice.',
    author = 'DMM',
    version = 0.1,
)