from dataclasses import dataclass
from typing import Callable

from PIL import Image
import numpy as np

from utils import ArrayNxNx3, Direction, Pixel, opposite, Vec2, Position


@dataclass
class PixelData:
    data: ArrayNxNx3[np.uint8]

    def map(self, fn: Callable[[ArrayNxNx3[np.uint8]], ArrayNxNx3[np.uint8]]):
        self.data = fn(self.data)
        return self

    @classmethod
    def from_image(cls, image: Image.Image):
        pixels: list[Pixel] = list(image.convert("RGB").getdata())  # type: ignore
        width, height = image.size
        result: ArrayNxNx3[np.uint8] = np.empty((width, height, 3), np.uint8)

        for x in range(width):
            for y in range(height):
                position = x, y
                pixel = pixels[width * y + x]
                result[position] = pixel

        return PixelData(result)

    @classmethod
    def from_dimensions(cls, dimensions: Vec2):
        width, height = dimensions
        data = np.zeros((width, height, 3), dtype=np.uint8)
        return cls(data)

    @classmethod
    def from_colour(cls, dimensions: Vec2, colour: Pixel):
        width, height = dimensions
        data = np.zeros((width, height, 3), dtype=np.uint8)
        data[:, :] = colour
        return cls(data)

    @classmethod
    def blend(cls, data: "PixelData", *other: "PixelData"):
        size = 1 + len(other)
        if size == 1:
            return data

        combined = [data, *other]
        nps = [x.data for x in combined]
        result = np.mean(np.array(nps), axis=0)
        return PixelData(result)

    def overlay(self, position: Position, data: "PixelData") -> "PixelData":
        width, height, _ = self.data.shape
        other_width, other_height, _ = data.data.shape

        row, col = position
        result = self.data.copy()

        # If we would place the data outside our own bounds, do nothing.
        if row >= height and col >= width:
            return PixelData(result)

        end_row = min(row + other_height, height)
        end_col = min(col + other_width, width)

        overlay_width = end_col - col
        overlay_height = end_row - row

        if overlay_width == 0 or overlay_height == 0:
            return PixelData(result)

        data = data.crop((0, 0), (overlay_width, overlay_height))
        result[col : col + overlay_width, row : row + overlay_height] = data.data

        return PixelData(result)

    def combine(self, position: Position, data: "PixelData") -> "PixelData":
        width, height, _ = self.data.shape
        overlay_width, overlay_height, _ = data.data.shape

        row, col = position
        result_width = max(width, col + overlay_width)
        result_height = max(height, row + overlay_height)

        result = PixelData.from_dimensions((result_width, result_height))
        result.overlay((0, 0), self)
        result.overlay(position, data)
        return result

    def to_image(self) -> Image.Image:
        return Image.fromarray(self.data, mode="RGB")

    def generate_tiles(self, kernel: Vec2, step: Vec2 = (1, 1)):
        n, m = kernel
        width, height, _ = self.data.shape

        dx, dy = step

        # pass kernel over data and extract
        for x in range(0, width - n, dx):
            for y in range(0, height - m, dy):
                start = x, y
                yield self.crop(start, kernel)

    def crop(self, position: Position, kernel: Vec2) -> "PixelData":
        x, y = position
        width, height = kernel

        data = self.data[x : x + width, y : y + height]
        return PixelData(data)

    def edge(self, direction: Direction, depth: int = 1) -> "PixelData":
        match direction:
            case Direction.UP:
                data = self.data[:depth]
            case Direction.RIGHT:
                data = self.data[:, -depth:]
            case Direction.LEFT:
                data = self.data[:, :depth]
            case Direction.DOWN:
                data = self.data[-depth:]

        return PixelData(data)

    def overlaps(
        self, other: "PixelData", from_direction: Direction, offset: int
    ) -> bool:
        """Would this data intersect the other if this data is overlayed from the given direction"""
        if other.data.shape != self.data.shape:
            return False

        return self.edge(from_direction, offset) == other.edge(
            opposite(from_direction), offset
        )

    def __hash__(self):
        return hash(self.data.tobytes())

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, PixelData):
            return False

        return np.array_equal(value.data, self.data)
