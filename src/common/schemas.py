"""
Schemas for the project.
"""
from enum import Enum


class Mode(str, Enum):
    paper = "paper"
    live = "live"
