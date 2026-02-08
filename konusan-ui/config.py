from pathlib import Path

# konusan-ui/ altından repo kökünü bul
BASE_DIR = Path(__file__).resolve().parents[1]

SADTALKER_PYTHON = str(BASE_DIR / "python" / "python.exe")
SADTALKER_DIR = str(BASE_DIR / "sadtalker")
FFMPEG_PATH = str(BASE_DIR / "ffmpeg" / "ffmpeg.exe")

DEFAULT_OUTPUT_DIR = str(BASE_DIR / "output_ui")
