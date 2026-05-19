from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / 'outputs'

MODEL_DIR = OUTPUT_DIR / 'model'
RESULT_DIR = OUTPUT_DIR / 'results'