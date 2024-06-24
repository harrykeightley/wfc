import argparse
from functools import partial
from pathlib import Path

from gui import run_tile_generation
from tile_generation import TileGenerationConfig, generate_tiles
from utils import Vec2


def as_path(path):
    return Path(path)


def dir_path(dir):
    """Helper type for ArgParse"""
    path = Path(dir)
    if path.exists() and path.is_dir():
        return dir
    else:
        raise NotADirectoryError(dir)


def file_path(raw_path: str):
    path = Path(raw_path)
    if path.is_dir():
        raise FileExistsError("Path points to a directory")
    else:
        return path


class InvalidDimensionsException(Exception):
    pass


def parse_dimensions(n: int, raw_dimensions: str):
    """Helper type for ArgParse to extract some dimensions"""
    sections = raw_dimensions.lower().split("x")

    if len(sections) != n:
        raise InvalidDimensionsException(
            f"Expected {n} dimensions (e.g. {'x'.join(['N'] * n)}) but found {len(sections)}"
        )

    try:
        return tuple([int(section) for section in sections])
    except ValueError:
        raise InvalidDimensionsException(
            f"Error converting dimensions {raw_dimensions} to tuples"
        )


def setup_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        title="Program",
        help="The WFC program to run",
        required=True,
        metavar="program",
        dest="program",
    )

    sudoku_parser = subparsers.add_parser("sudoku", help="Generate a sudoku map")

    # TILE GENERATION
    tile_generation_parser = subparsers.add_parser(
        "tiles", help="Generate tiles from an input image"
    )

    tile_generation_parser.add_argument(
        "-k",
        "--kernel",
        help="Dimensions which represent the NxM kernel (width x height) to use in the algorithm",
        type=partial(parse_dimensions, 2),
        required=True,
    )

    tile_generation_parser.add_argument(
        "-i",
        "--input-path",
        help="The path to an image to generate samples from",
        type=as_path,
        required=True,
    )

    tile_generation_parser.add_argument(
        "-d",
        "--output-dimensions",
        help="Dimensions which represent the pixels in the output image (width x height)",
        type=partial(parse_dimensions, 2),
        required=True,
    )

    tile_generation_parser.add_argument(
        "-o",
        "--output-path",
        help="The path to save the resulting image (if we could make one)",
        type=file_path,
        default="./results.jpg",
    )

    tile_generation_parser.add_argument(
        "--overlap",
        help="Overlapping amount to use when finding neighbours",
        type=int,
        default=1,
    )

    tile_generation_parser.add_argument(
        "--max-attempts",
        help="Maximum amount of attempts to try before giving up",
        type=int,
        default=20,
    )

    tile_generation_parser.add_argument(
        "--sampling-step",
        help="Delta to add to the kernel each sampling step to extract a tile",
        type=partial(parse_dimensions, 2),
        default=(1, 1),
    )

    tile_generation_parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to show a gui",
        default=False,
    )

    return parser.parse_args()


def main(args):
    match args.program:
        case "tiles":
            config = TileGenerationConfig(
                kernel=args.kernel,
                output_pixels=args.output_dimensions,
                output_path=args.output_path,
                neighbour_overlap=args.overlap,
                sampling_step=args.sampling_step,
                max_attempts=args.max_attempts,
                gui=args.gui,
            )
            image_path = args.input_path
            print(config, image_path)

            if config.gui:
                run_tile_generation(config, image_path)

            else:
                generate_tiles(config, image_path)

        case "sudoku":
            pass

        case action:
            print(f"unknown option {action}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
