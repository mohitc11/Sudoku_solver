# Sudoku Solver (Flask + PaddleOCR)

Upload a photo of a Sudoku puzzle, and the app extracts the grid with
PaddleOCR, solves it with a backtracking solver, and displays the result.

## Project layout

```
sudoku_app/
├── app.py              # Flask routes: page + /api/solve
├── sudoku_ocr.py        # Image -> 9x9 grid JSON (OpenCV + PaddleOCR)
├── sudoku_solver.py      # 9x9 grid JSON -> solved grid JSON (backtracking)
├── templates/
│   └── index.html        # Upload UI + animated result grid
├── static/                # (reserved for any extra assets)
└── requirements.txt
```

## Setup

```bash
cd Sudoku_solver
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt --break-system-packages
```

## Run

```bash
python app.py
```

for local testing use  
```bash
python sudoku_ocr.py sample_sudoku.jpg output.json
```

```bash
python sudoku_solver.py output.json solved.json
```


Then open **http://localhost:5000** in your browser.

## How it works

1. **Upload** — drag/drop or click to select an image (PNG/JPG/WEBP/BMP, up to 8 MB).
2. **Solve puzzle** button sends the file to `POST /api/solve` as `multipart/form-data`.
3. The backend:
   - saves the upload to a temp file,
   - calls `extract_sudoku_to_json()` (grid detection → perspective warp → per-cell PaddleOCR) to get the 9×9 grid,
   - validates it (catches duplicate digits from OCR misreads),
   - runs `solve()` (backtracking) on a copy,
   - returns JSON: `{"extracted_grid": [...], "solved_grid": [...] | null, "solved": bool, "error": string | null}`.
4. The page renders the grid: originally-given digits in ink, solved digits in amber, animated in cell-by-cell.

## Notes

- If `solved` comes back `false`, the extracted grid likely has an OCR misread — an unsolvable Sudoku almost always means the input digits are wrong, not a bug in the solver.
- The 8 MB upload cap is set in `app.py` (`MAX_CONTENT_LENGTH`); raise it if needed.
- `sudoku_ocr.py` loads the PaddleOCR recognition model once at server startup, so the first request after launch may be a bit slower while the model loads/warms up.
-  for more accurate ocr model change model in `sudoku_ocr.py` from `PP-OCRv5_mobile_rec` to `PP-OCRv6_medium_rec`
