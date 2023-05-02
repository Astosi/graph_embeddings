import logging
import time

class CoolFormatter(logging.Formatter):
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    GREY = "\033[37m"

    def format(self, record):
        level = record.levelname
        if level == "DEBUG":
            level = f"{self.BLUE}{level}{self.END}"
        elif level == "INFO":
            level = f"{self.GREEN}{level}{self.END}"
        elif level == "WARNING":
            level = f"{self.YELLOW}{level}{self.END}"
        elif level == "ERROR":
            level = f"{self.RED}{level}{self.END}"
        elif level == "CRITICAL":
            level = f"{self.BOLD}{self.RED}{level}{self.END}"

        message = super().format(record)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        return f"{self.GREY}{timestamp}{self.END} - {level} - {message}"
