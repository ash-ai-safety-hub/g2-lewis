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

class MarketActions(IntEnum):
    SetPrice1 = 0
    SetPrice2 = 1
    SetPrice3 = 2
    SetPrice4 = 3
    SetPrice5 = 4