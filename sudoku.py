from functools import partial
from typing import Any, Callable, Iterable, Optional

from wfc import Position, Update, Weighted, wavefunction_collapse

BOARD_SIZE = 9
ALL_NUMBERS = tuple(range(1, 10))

Square = dict[int, bool]
Board = dict[Position, Square]
BoardResult = dict[Position, int]


def initial_square_state():
    return {number: True for number in ALL_NUMBERS}


def square_potential(square: Square):
    return len([v for v in square.values() if v])


def initial_board_state() -> Board:
    result: Board = {}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            position = row, col
            result[position] = initial_square_state()

    return result


def get_positions(board: Board) -> Iterable[Position]:
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            yield row, col


def positions_in_row(row: int):
    return [(row, col) for col in range(BOARD_SIZE)]


def positions_in_column(col: int):
    return [(row, col) for row in range(BOARD_SIZE)]


def positions_in_quadrant(position: Position) -> list[Position]:
    start_row, start_col = position
    row_section = start_row // 3
    col_section = start_col // 3

    section_start = row_section * 3, col_section * 3
    result = []
    for row_offset in range(3):
        for col_offset in range(3):
            row, col = section_start
            section_position = row + row_offset, col + col_offset
            result.append(section_position)

    return result


def square_entropy(board: Board, position: Position):
    return max(0, (square_potential(board[position])) - 1)


def is_square_solved(square: Square):
    return square_potential(square) == 1


def get_square_solution(square: Square):
    for number, is_valid in square.items():
        if is_valid:
            return number

    return -1


def filter_by_fixed(board: Board, positions: Iterable[Position]) -> Iterable[Position]:
    return filter(lambda position: is_square_solved(board[position]), positions)


def get_fixed_squares(board: Board) -> BoardResult:
    return {
        pos: get_square_solution(board[pos])
        for pos in filter_by_fixed(board, get_positions(board))
    }


def is_solved(board: Board) -> bool:
    return all(
        square_potential(board[position]) == 1 for position in get_positions(board)
    )


def intersecting_positions(position: Position) -> list[Position]:
    row, col = position
    return [
        *positions_in_column(col),
        *positions_in_row(row),
        *positions_in_quadrant(position),
    ]


def potential_numbers(board: Board, position: Position) -> set[int]:
    positions_to_check = intersecting_positions(position)
    numbers = set(
        map(
            lambda pos: get_square_solution(board[pos]),
            filter_by_fixed(board, positions_to_check),
        )
    )
    return set(ALL_NUMBERS).difference(numbers)


def update_square(board: Board, position: Position, square: Square) -> Board:
    board[position] = square
    return board


def collapse_square(number: int) -> Square:
    result = {number: False for number in ALL_NUMBERS}
    result[number] = True
    return result


def propagate_number(square: Square, number: int) -> Square:
    square[number] = False
    return square


def sudoku_actions(board: Board, position: Position) -> list[Weighted[Update[Board]]]:
    set_square = partial(update_square, board, position)
    possible_numbers = potential_numbers(board, position)

    def collapse_to(number: int) -> Update[Board]:
        return lambda board: set_square(collapse_square(number))

    return [(collapse_to(i), 1) for i in possible_numbers]


def print_board(board: Board, display: Callable[[Board, Position], Optional[Any]]):
    print()
    for row in range(BOARD_SIZE):
        output = ""
        for col in range(BOARD_SIZE):
            position = row, col
            char = str(display(board, position))
            char = char if char is not None else "#"
            output += char
        print(output)


def get_fixed(board: Board, position: Position):
    square = board[position]
    if is_square_solved(square):
        return str(get_square_solution(square))
    else:
        return " "


def propagate_collapse(board: Board, last_position: Position) -> Board:
    print_board(board, get_fixed)

    last_number = get_square_solution(board[last_position])
    relevant_positions = intersecting_positions(last_position)
    for position in relevant_positions:
        if position != last_position:
            update_square(
                board, position, propagate_number(board[position], last_number)
            )

    return board


def sudoku_generator(tries=10):
    attempts = 0
    solution = None
    while attempts < tries:
        board = initial_board_state()
        solution = wavefunction_collapse(
            board, get_positions, square_entropy, sudoku_actions, propagate_collapse
        )
        if is_solved(solution):
            break

        attempts += 1

    if solution and is_solved(solution):
        print("SOLVED!")
        print_board(solution, get_fixed)
    else:
        print("FAILED")