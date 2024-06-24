from functools import partial, reduce
import os
from typing import Annotated, Callable, Iterable, Literal, Optional, TypeVar, override
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import math
from tqdm import tqdm

from utils import (
    DType,
    Direction,
    Position,
    Vec2,
    is_in_bounds,
    opposite,
    position_in_direction,
)
from pixeldata import PixelData
from wfc import WFC, Update, Weighted, wavefunction_collapse
from collections import Counter, deque

DType = TypeVar("DType", bound=np.generic)

Array3D = Annotated[npt.NDArray[DType], Literal["X", "Y", "Z"]]
Wave = Array3D[np.bool_]

ArrayNx4xN = Annotated[npt.NDArray[DType], Literal["N", 4, "N"]]

# A type which represents Original Tile -> Direction -> Other Tile -> <data>
Neighbours = ArrayNx4xN[np.bool_]


@dataclass
class TileGenerationConfig:
    kernel: Vec2
    sampling_step: int
    max_attempts: int
    output_pixels: Vec2
    neighbour_overlap: int
    output_path: Optional[Path] = None
    gui: bool = False


@dataclass
class TileInfo:
    tile_map: dict[int, PixelData]
    frequencies: dict[int, int]
    neighbours: Neighbours

    @classmethod
    def from_image(cls, config: TileGenerationConfig, image: Image.Image):
        bitmap = PixelData.from_image(image)

        # Reduce to unique list of tiles and frequencies
        raw_frequencies = Counter(bitmap.generate_tiles(config.kernel))

        # Assign each tile a number for memory purposes
        tile_map = dict(enumerate(raw_frequencies.keys()))

        frequencies = {id: raw_frequencies[tile_map[id]] for id in tile_map.keys()}

        # Generate neighbour map from unique tiles
        neighbours = generate_neighbours(tile_map, overlap=config.neighbour_overlap)

        return cls(tile_map, frequencies, neighbours)

    @classmethod
    def from_tileset(cls, config: TileGenerationConfig, dir: Path):
        tiles: list[PixelData] = []
        for file in dir.iterdir():
            if not file.is_file():
                continue
            tiles.append(PixelData.from_image(Image.open(file)))

        tile_map = dict(enumerate(tiles))
        frequencies = {id: 1 for id in tile_map.keys()}

        neighbours = generate_neighbours(tile_map, overlap=1)
        return cls(tile_map, frequencies, neighbours)

    @property
    def ids(self) -> list[int]:
        return list(self.tile_map.keys())

    def __len__(self) -> int:
        return len(self.ids)


class TileGenerator(WFC[Wave, Position]):

    def __init__(self, config: TileGenerationConfig, tile_info: TileInfo) -> None:
        self._status = "Pending"
        self.config = config
        self.tile_info = tile_info

        output_width, output_height = config.output_pixels
        kernel_width, kernel_height = config.kernel

        row_step = kernel_height - config.neighbour_overlap
        col_step = kernel_width - config.neighbour_overlap
        self.kernel_step = col_step, row_step

        rows = math.ceil(output_height / row_step)
        cols = math.ceil(output_width / col_step)
        self.tiled_bounds = rows, cols

        self.invalid_tile = PixelData.from_colour(config.kernel, (255, 0, 0))
        self.reset()

    def reset(self):
        rows, cols = self.tiled_bounds
        self.state = np.ones((cols, rows, len(self.tile_info)), dtype=np.bool_)

    @property
    def wave(self):
        return self.state

    @override
    def get_elements(self) -> Iterable[Position]:
        rows, cols, _ = self.state.shape
        for row in range(rows):
            for col in range(cols):
                position = row, col
                yield position

    @override
    def entropy(self, element: Position) -> int:
        return np.count_nonzero(self.state[element]) - 1

    @override
    def actions(self, element: Position) -> list[Weighted[Update[Wave]]]:
        row, col = element
        result: list[Weighted[Update[Wave]]] = []
        for id in self.tile_info.ids:
            # If that tile is still possible
            if self.state[row, col, id]:
                # Add the action which would collapse it, with the attached probability
                result.append(
                    (partial(collapse, id, element), self.tile_info.frequencies[id])
                )

        return result

    @override
    def propagate(self, last_collapsed_element: Position) -> None:
        changed: deque[Position] = deque()
        changed.append(last_collapsed_element)

        while len(changed):
            position = changed.popleft()

            for direction in Direction:
                neighbour = position_in_direction(position, direction)
                if not is_in_bounds(self.tiled_bounds, neighbour):
                    continue

                did_change = self._update_possibilities(neighbour, position, direction)
                if did_change:
                    changed.append(neighbour)

    def _update_possibilities(
        self, position: Position, neighbour: Position, direction: Direction
    ):
        """Update the current positions state based off the neighbours state.

        Parameters:
            position: The position whose possibilities are to be updated.
            neighbour: The "oracle" position to update from.
            direction: Direction from position to neighbour
        """

        current_possibilities = self.state[position]

        result = np.logical_and(
            current_possibilities,
            self._possible_tile_ids_in_direction(neighbour, opposite(direction)),
        )

        did_change = not np.array_equal(current_possibilities, result)
        self.state[position] = result

        return did_change

    def _possible_tile_ids_in_direction(self, position: Position, direction: Direction):
        # Start with none being possible
        result = np.zeros(len(self.tile_info.ids), dtype=np.bool_)

        def possibilities(id: int):
            return self.tile_info.neighbours[id, direction.value]

        return reduce(
            np.logical_or, map(possibilities, self._possible_tile_ids(position)), result
        )

    def _possible_tile_ids(self, position: Position) -> list[int]:
        # Get a list of lists of the indices that are nonzero
        possible = np.argwhere(self.state[position])

        # Get the possible ids for this position
        return [arg[0] for arg in possible.tolist()]

    def _display_tile(self, position: Position) -> PixelData:
        # Get the possible tiles for this position
        tiles = [
            self.tile_info.tile_map[id] for id in self._possible_tile_ids(position)
        ]

        # Check for invalid tiles
        if len(tiles) == 0:
            return self.invalid_tile

        return PixelData.blend(*tiles)

    def build_image(self) -> Image.Image:
        result = PixelData.from_dimensions(self.config.output_pixels)
        rows, cols = self.tiled_bounds
        row_step, col_step = self.kernel_step

        for row in range(rows):
            for col in range(cols):
                position = row, col
                pixel_position = row * row_step, col * col_step
                tile = self._display_tile(position)
                result = result.overlay(pixel_position, tile)

        return result.to_image()


def collapse(id: int, position: Position, wave: Wave) -> Wave:
    row, col = position
    wave[row, col, :] = False
    wave[row, col, id] = True
    return wave


def build_wave_actions(tile_info: TileInfo, wave: Wave, position: Position):
    row, col = position
    result: list[Weighted[Update[Wave]]] = []
    for id in tile_info.ids:
        # If that tile is still possible
        if wave[row, col, id]:
            # Add the action which would collapse it, with the attached probability
            result.append((partial(collapse, id, position), tile_info.frequencies[id]))

    return result


def generate_neighbours(tiles: dict[int, PixelData], overlap: int = 1) -> Neighbours:
    """Neighbours should be used like id -> direction_idx -> <all ids that can be there>"""
    count = len(tiles)
    result = np.zeros((count, 4, count), dtype=np.bool_)

    for id, tile in tiles.items():
        for direction in Direction:
            for other_id, other_tile in tiles.items():
                result[other_id][direction.value][id] = tile.overlaps(
                    other_tile, direction, overlap
                )

    return result
