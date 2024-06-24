from enum import Enum, StrEnum
from functools import partial, reduce
from typing import Annotated, Callable, Iterable, Literal, TypeVar

import numpy as np
import numpy.typing as npt

Vec2 = tuple[int, int]
Position = Vec2
Pixel = tuple[int, int, int]

DType = TypeVar("DType", bound=np.generic)
ArrayNxN = Annotated[npt.NDArray[DType], Literal["N", "N"]]
ArrayNxNx3 = Annotated[npt.NDArray[DType], Literal["N", "N", 3]]


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


DIRECTION_DELTAS: dict[Direction, Vec2] = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


def opposite(direction: Direction) -> Direction:
    match direction:
        case Direction.UP:
            return Direction.DOWN
        case Direction.DOWN:
            return Direction.UP
        case Direction.LEFT:
            return Direction.RIGHT
        case Direction.RIGHT:
            return Direction.LEFT


def sum_vecs(*vec2s: Vec2) -> Vec2:
    def add(v1: Vec2, v2: Vec2) -> Vec2:
        return (v1[0] + v2[0], v1[1] + v2[1])

    return reduce(add, vec2s)


def is_in_bounds(bounds: Vec2, position: Position):
    return 0 <= position[0] < bounds[0] and 0 <= position[1] < bounds[1]


def position_in_direction(position: Position, direction: Direction) -> Position:
    return sum_vecs(position, DIRECTION_DELTAS[direction])


def get_neighbouring_positions(bounds: Vec2, position: Position) -> Iterable[Position]:
    is_valid = partial(is_in_bounds, bounds)
    result = []
    for direction in Direction:
        next_position = position_in_direction(position, direction)
        if is_valid(next_position):
            result.append(next_position)

    return result
