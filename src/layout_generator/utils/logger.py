import os
import sys
from datetime import datetime


class Logger:
    COLORS = {
        "INFO": "\033[96m",
        "DEBUG": "\033[94m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }

    def __init__(self, debug=False, info=True, log_file=None, source=None):
        self.debug_mode = debug
        self.info_mode = info
        self.log_file = log_file
        self.source = source or "Logger"
        self.log_handle = None

        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Open the file once and keep it open
            self.log_handle = open(self.log_file, "a", encoding='utf-8')
            self.log_handle.write(f"# Log start for {self.source} at {self._timestamp()}\n")
            self.log_handle.flush()

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, level: str, msg: str):
        timestamp = self._timestamp()
        tag = f"[{level}] [{self.source}] [{timestamp}]"
        
        # Log to console
        color = self.COLORS.get(level, "")
        reset = self.COLORS["RESET"]
        console_msg = f"{color}{tag}{reset} {msg}"

        if (level == "INFO" and self.info_mode) or \
           (level == "DEBUG" and self.debug_mode) or \
           level in {"WARNING", "ERROR"}:
            print(console_msg, file=sys.stderr if level == "ERROR" else sys.stdout)

        # Log to file
        if self.log_handle:
            file_msg = f"{tag} {msg}\n"
            self.log_handle.write(file_msg)
            self.log_handle.flush()  # Ensure message is written to disk immediately

    def close(self):
        """Closes the log file handle."""
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None

    def __del__(self):
        """Destructor to ensure the file is closed when the object is destroyed."""
        self.close()

    def info(self, msg: str):
        self._log("INFO", msg)

    def debug(self, msg: str):
        self._log("DEBUG", msg)

    def warning(self, msg: str):
        self._log("WARNING", msg)

    def error(self, msg: str):
        self._log("ERROR", msg)
