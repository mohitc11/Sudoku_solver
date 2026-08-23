"""
Extract a Sudoku grid from an image into a 9x9 JSON structure using PaddleOCR.

Pipeline:
1. Preprocess the image and locate the largest square-ish contour (the Sudoku board).
2. Warp-perspective it into a flat top-down square image.
3. Slice that square into an 81-cell grid.
4. Run PaddleOCR on each cell (numbers only) to read the digit, if any.
5. Assemble the results into a 9x9 JSON grid (0 for empty cells).

Install deps first:
    pip install paddleocr paddlepaddle opencv-python numpy --break-system-packages
"""

import json
import time

import cv2
import numpy as np
from paddleocr import TextRecognition

from sudoku_solver import _print_grid

# Each Sudoku cell is already cropped down to a single isolated digit, so we
# don't need the full OCR pipeline (text detection + classification +
# recognition) — just the recognition model. This also sidesteps a Paddle/
# oneDNN text-detection kernel bug some environments hit
# (NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...).
# _rec = TextRecognition(model_name="PP-OCRv6_medium_rec")      # heavier model, takes about 50 - 60 seconds, high accuracy, multi-languages
_rec = TextRecognition(model_name="PP-OCRv5_mobile_rec")        # light model, takes a couple of seconds, medium accuracy, multi-languages
# _rec = TextRecognition(model_name="en_PP-OCRv5_mobile_rec")   # light model, faster than PP-OCRv5_mobile_rec model, medium accuracy, english

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_grid_contour(gray: np.ndarray) -> np.ndarray:
    """Find the largest 4-point contour, assumed to be the Sudoku board border."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    raise ValueError("Could not find a 4-cornered Sudoku grid in the image.")


def _warp_grid(image: np.ndarray, corners: np.ndarray, size: int = 900) -> np.ndarray:
    """Perspective-warp the detected board into a size x size square."""
    rect = _order_points(corners.astype("float32"))
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def _split_cells(warped_gray: np.ndarray, size: int = 900):
    """Split the warped square into 81 cell images, each cropped to trim grid lines."""
    step = size // 9
    margin = int(step * 0.12)  # trim border lines / neighboring digits
    cells = []
    for row in range(9):
        for col in range(9):
            y0, y1 = row * step + margin, (row + 1) * step - margin
            x0, x1 = col * step + margin, (col + 1) * step - margin
            cell = warped_gray[y0:y1, x0:x1]
            cells.append(cell)
    return cells


def _cell_has_digit(cell: np.ndarray) -> bool:
    """Heuristic: check if a cell has enough dark pixels to contain a digit."""
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    filled_ratio = np.count_nonzero(thresh) / thresh.size
    return filled_ratio > 0.03


def _read_digit(cell: np.ndarray) -> int:
    """Run PaddleOCR on a single cell image and return the recognized digit (0 if none)."""
    if not _cell_has_digit(cell):
        return 0

    # Upscale + binarize to help OCR on small, low-res cell crops
    cell_big = cv2.resize(cell, (128, 128), interpolation=cv2.INTER_CUBIC)
    _, cell_bin = cv2.threshold(cell_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cell_rgb = cv2.cvtColor(cell_bin, cv2.COLOR_GRAY2RGB)

    result = _rec.predict(cell_rgb)

    if not result:
        return 0

    res = result[0]
    # Standalone TextRecognition returns a single text/score per image
    # (dict-like or attribute-style depending on version), unlike the full
    # pipeline's per-region `rec_texts`/`rec_scores` lists.
    rec_text = res.get("rec_text") if hasattr(res, "get") else getattr(res, "rec_text", "")
    rec_score = res.get("rec_score") if hasattr(res, "get") else getattr(res, "rec_score", 0.0)

    if not rec_text or (rec_score is not None and rec_score < 0.5):
        return 0

    text = "".join(ch for ch in rec_text if ch.isdigit())
    if text:
        return int(text[0])
    return 0


def extract_sudoku_to_json(image_path: str, output_path: str = None) -> dict:
    """
    Extract a Sudoku grid from `image_path` and return/save it as JSON.

    Returns a dict: {"grid": [[int, ...9...], ...9 rows...]}
    Empty cells are represented as 0.
    """
    start_total = time.perf_counter()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    t = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"\n\nGrayscale: {(time.perf_counter() - t) * 1000:.2f} ms")

    t = time.perf_counter()
    corners = _find_grid_contour(gray)
    print(f"Find grid: {(time.perf_counter() - t) * 1000:.2f} ms")

    t = time.perf_counter()
    warped_color = _warp_grid(image, corners)
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    print(f"Warp: {(time.perf_counter() - t) * 1000:.2f} ms")

    t = time.perf_counter()
    cells = _split_cells(warped_gray)
    print(f"Split cells: {(time.perf_counter() - t) * 1000:.2f} ms")

    t = time.perf_counter()
    digits = [_read_digit(cell) for cell in cells]
    print(f"Read digits: {(time.perf_counter() - t) * 1000:.2f} ms")

    grid = [digits[r * 9:(r + 1) * 9] for r in range(9)]
    result = {"grid": grid}

    if output_path:
        t = time.perf_counter()
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
            print(f"JSON save: {(time.perf_counter() - t) * 1000:.2f} ms")

    print(f"TOTAL: {(time.perf_counter() - start_total) * 1000:.2f} ms")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sudoku_ocr.py <image_path> [output.json]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "sudoku.json"

    data = extract_sudoku_to_json(img_path, out_path)
    # print(json.dumps(data, indent=2))
    print("\nUnsolved puzzle:")
    _print_grid(data["grid"])
    print(f"\nSaved to {out_path}")