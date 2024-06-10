from enum import Enum, StrEnum
from typing import Annotated, Callable, Iterable, Literal, Optional, Tuple, TypeVar
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import random

Vec2 = tuple[int, int]
Position = Vec2
Pixel = tuple[int, int, int]

DType = TypeVar("DType", bound=np.generic)
ArrayNxN = Annotated[npt.NDArray[DType], Literal["N", "N"]]
ArrayNxNx3 = Annotated[npt.NDArray[DType], Literal["N", "N", 3]]


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class PixelData:
    data: ArrayNxNx3[np.uint8]

    def map(self, fn: Callable[[ArrayNxNx3[np.uint8]], ArrayNxNx3[np.uint8]]):
        self.data = fn(self.data)
        return self

    @classmethod
    def from_image(cls, image: Image.Image):
        pixels: list[Pixel] = list(image.getdata())  # type: ignore
        width, height = image.size
        result: ArrayNxNx3[np.uint8] = np.empty((width, height, 3), np.uint8)

        for x in range(width):
            for y in range(height):
                position = x, y
                pixel = pixels[width * y + x]
                result[position] = pixel

        return PixelData(result)

    def generate_tiles(self, kernel: Vec2):
        n, m = kernel
        width, height, _ = self.data.shape

        # pass kernel over data and extract
        for x in range(0, width - n, n):
            for y in range(0, height - m, m):
                start = x, y
                yield self.crop(start, kernel)

    def crop(self, position: Position, kernel: Vec2) -> "PixelData":
        x, y = position
        width, height = kernel

        data = self.data[x : x + width, y : y + height]
        return PixelData(data)

    def intersects(
        self, other: "PixelData", from_direction: Direction, offset: Vec2
    ) -> bool:
        return True

    def __hash__(self):
        return hash(self.data.tobytes())

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, PixelData):
            return False

        return np.array_equal(value.data, self.data)


@dataclass
class WFCConfig:
    kernel: Vec2


def wfc(config: WFCConfig, image_path: Path):
    image = Image.open(image_path)
    bitmap = PixelData.from_image(image)

    tiles = set(bitmap.generate_tiles(config.kernel))


T = TypeVar("T")
Update = Callable[[T], T]
Weighted = tuple[T, int]


def wavefunction_collapse[
    State, Element
](
    state: State,
    get_elements: Callable[[State], Iterable[Element]],
    entropy: Callable[[State, Element], int],
    actions: Callable[[State, Element], list[Weighted[Update[State]]]],
    propagate: Callable[[State, Element], State],
):

    while True:
        non_collapsed = list(
            filter(lambda element: entropy(state, element) > 0, get_elements(state))
        )
        if (len(non_collapsed)) == 0:
            return state

        next_element = min(non_collapsed, key=lambda element: entropy(state, element))

        if (entropy(state, next_element)) == 0:
            return state

        weighted_actions = actions(state, next_element)
        # Contradiction Encountered
        if len(weighted_actions) == 0:
            return state

        just_actions, weights = zip(*weighted_actions)
        action = random.choices(just_actions, weights=weights)[0]

        state = propagate(action(state), next_element)


Wave = ArrayNxN[np.bool_]


def get_wave_elements(wave: Wave) -> Iterable[Position]:
    width, height = wave.shape
    for y in range(height):
        for x in range(width):
            position = x, y
            yield position


def wave_element_entropy(wave: Wave, position: Position):
    return np.count_nonzero(wave[position])


def wave_actions(wave: Wave, position: Position) -> list[Weighted[Update[Wave]]]:
    pass
