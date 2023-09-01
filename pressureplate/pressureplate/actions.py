from enum import IntEnum

class GridActions(IntEnum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Noop = 4

class IPDActions(IntEnum):
    Lie = 0
    Confess = 1