from enum import Enum


class MathSign(Enum):
    MINUS = 1
    FRACTION_HOR = 2
    FRACTION_VERT = 3
    MULTIPLY = 4
    COMMA = 5
    EQUAL = 6


class Position(Enum):
    BASIS = 1
    EXPONENT = 2
