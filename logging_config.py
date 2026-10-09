import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


class _Tee:
    """Mirror a stream (stdout/stderr) into the log file, so the file holds
    exactly what the terminal showed: prints, logging lines and warnings."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, name):  # isatty(), encoding, fileno()...
        return getattr(self._stream, name)


def setup_logging(level: int = logging.INFO) -> Path:
    """Configure root logging once and tee all terminal output to logs/run_<ts>.log.

    The tee must be installed *before* basicConfig: its StreamHandler binds to
    whatever sys.stderr is at that moment, so logging lines reach the file too.
    Returns the log file path.
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return log_path
