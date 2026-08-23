"""
Flask app: upload a Sudoku photo, extract the grid with PaddleOCR, solve it,
and display the result.

Run:
    pip install flask paddleocr paddlepaddle opencv-python numpy --break-system-packages
    python app.py

Then open http://localhost:5000
"""

import os
import tempfile

from flask import Flask, jsonify, render_template, request, send_from_directory

from sudoku_ocr import extract_sudoku_to_json
from sudoku_solver import _validate_input_grid, solve

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )

@app.route("/api/solve", methods=["POST"])
def api_solve():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify(
            {"error": "Unsupported file type. Use PNG, JPG, JPEG, WEBP, or BMP."}
        ), 400

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        print("File stored at:", tmp_path)
        extracted = extract_sudoku_to_json(tmp_path)
        original_grid = extracted["grid"]

        # Solve on a copy so we can return both the extracted and solved grids.
        solved_grid = [row[:] for row in original_grid]
        try:
            _validate_input_grid(solved_grid)
        except ValueError as e:
            return jsonify(
                {"error": str(e), "extracted_grid": original_grid}
            ), 422

        was_solved = solve(solved_grid)

        return jsonify(
            {
                "extracted_grid": original_grid,
                "solved_grid": solved_grid if was_solved else None,
                "solved": was_solved,
                "error": None
                if was_solved
                else "No solution exists for the extracted puzzle — likely an OCR misread.",
            }
        )

    except ValueError as e:
        # Raised by extract_sudoku_to_json when the grid border can't be found.
        return jsonify({"error": str(e)}), 422
    except Exception as e:  # noqa: BLE001 - surfaced to the user as-is
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
