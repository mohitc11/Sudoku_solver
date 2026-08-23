"""
Solve a Sudoku puzzle from a JSON file produced by sudoku_ocr.py.

Input JSON format:
    {"grid": [[0,0,3,...], ...9 rows of 9 ints, 0 = empty...]}

Usage:
    python sudoku_solver.py output.json solved.json
"""

import json
import sys
import time


def _find_empty(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def _is_valid(grid, row, col, num):
    # Row and column check
    if num in grid[row]:
        return False
    if num in (grid[r][col] for r in range(9)):
        return False

    # 3x3 box check
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num:
                return False

    return True


def solve(grid) -> bool:
    """Solve `grid` in place via backtracking. Returns True if solvable."""
    empty = _find_empty(grid)
    if not empty:
        return True  # no empty cells left -> solved

    row, col = empty
    for num in range(1, 10):
        if _is_valid(grid, row, col, num):
            grid[row][col] = num
            if solve(grid):
                return True
            grid[row][col] = 0  # backtrack

    return False


def _validate_input_grid(grid):
    """Basic sanity checks: 9x9, values 0-9, no duplicate given digits."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("Grid must be 9x9.")

    for row in grid:
        for val in row:
            if not isinstance(val, int) or not (0 <= val <= 9):
                raise ValueError(f"Invalid cell value: {val!r} (must be int 0-9).")

    # Check no row/col/box already has a duplicate among the given (non-zero) digits
    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            if val == 0:
                continue
            grid[r][c] = 0  # temporarily clear to test against the rest
            ok = _is_valid(grid, r, c, val)
            grid[r][c] = val
            if not ok:
                raise ValueError(
                    f"Invalid puzzle: digit {val} at row {r}, col {c} "
                    f"conflicts with another given digit."
                )


def solve_sudoku_json(input_path: str, output_path: str = None) -> dict:
    """
    Load a grid JSON (as produced by sudoku_ocr.py), solve it, and
    return/save the solved grid as JSON: {"grid": [[...]], "solved": true}.
    """
    start_total = time.perf_counter()

    with open(input_path) as f:
        data = json.load(f)

    t = time.perf_counter()
    grid = data["grid"]
    _validate_input_grid(grid)
    print(f"\n\nValidate input grid: {(time.perf_counter() - t) * 1000:.2f} ms")

    t = time.perf_counter()
    solved = solve(grid)
    print(f"Solved puzzle: {(time.perf_counter() - t) * 1000:.2f} ms")

    if not solved:
        result = {"grid": grid, "solved": False, "error": "No solution exists for this puzzle."}
    else:
        result = {"grid": grid, "solved": True}

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"TOTAL: {(time.perf_counter() - start_total) * 1000:.2f} ms")
    return result


def _print_grid(grid):
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            row_str += f"{grid[r][c] if grid[r][c] != 0 else '.'} "
        print(row_str)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sudoku_solver.py <input.json> [output.json]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "solved.json"

    result = solve_sudoku_json(in_path, out_path)
    print(result)
    if result["solved"]:
        print("Solved puzzle:\n")
        _print_grid(result["grid"])
        print(f"\nSaved to {out_path}")
    else:
        print(result["error"])
