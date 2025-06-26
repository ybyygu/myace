"""
MyACE Tools - A collection of utilities for the ACE/pacemaker workflow.
"""

# Import high-level API functions for easy access
from .data_processing import build_dataset
from .active_learning import evaluate_and_select

# Import the I/O module under its own namespace
from . import io
from . import utils

# Define what gets imported with 'from myace import *'
__all__ = [
    'build_dataset',
    'evaluate_and_select',
    'utils',
    'io'
]
