import sys
import logging
from ocr_processor import OCRProcessor

logging.basicConfig(level=logging.ERROR)

def test():
    processor = OCRProcessor()
    
    # Real-world test cases from the quiz app
    tests = [
        ("Karkaroff, 4 1-", "Karkaroff, 4"),
        ("Karkaroff, 4 V", "Karkaroff, 4"),
        ("Dumbledore, 8 a", "Dumbledore, 8"),
        ("Maxime, 7 -", "Maxime, 7"),
        ("Karkaroff, 0", "Karkaroff, 0"),
        ("Dumbledore, -8", "Dumbledore, -8"),  # Negative numbers
    ]
    
    print("Real-World Quiz Answer Cleaning Tests")
    print("=" * 70)
    
    all_pass = True
    for inp, exp in tests:
        out = processor.clean_answer_text(inp)
        ok = (out == exp)
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] '{inp}' -> '{out}' (expected '{exp}')")
    
    print("=" * 70)
    print("RESULT:", "ALL PASSED" if all_pass else "SOME FAILED")
    return all_pass

if __name__ == "__main__":
    sys.exit(0 if test() else 1)
