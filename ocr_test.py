import os

# Fix PaddlePaddle CPU / oneDNN PIR issue
os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

result = ocr.predict("sudoku_warped.jp")

for res in result:
    res.print()