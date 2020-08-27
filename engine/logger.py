import logging
import sys

from colorlog import ColoredFormatter
from tqdm import tqdm


class PBarHandler(logging.Handler):
    def __init__(self, fp):
        logging.Handler.__init__(self)
        self.pbars = []
        self.pbar_offset = 0
        self.fp = fp

    def emit(self, record):
        line = self.format(record)
        if self.pbars:
            self.pbars[-1].write(line)
            self.pbar_offset = len(self.pbars) - 1
        elif self.pbar_offset:
            for _ in range(self.pbar_offset):
                self.fp.write("\n")
            self.pbar_offset = 0
            self.fp.write(line + "\n")
        else:
            self.fp.write(line + "\n")

    def register(self, pbar):
        self.pbars.append(pbar)
        _pbar_close = pbar.close

        def pbar_close():
            if pbar.disable:
                return
            _pbar_close()
            self.pbars.pop()

        pbar.close = pbar_close


class Logger:
    colored_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s %(log_color)s%(levelname).4s%(reset)s %(log_color)s[%(name)s:%(lineno)d]%(reset)s %(message_log_color)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
                    "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={"message": {
            "ERROR": "red", "CRITICAL": "red,bg_white"}},
        style="%",
    )

    pbar_handler = PBarHandler(sys.stdout)
    pbar_handler.setFormatter(colored_formatter)

    file_handler = None

    def __init__(self, name, level):
        self._logger = logging.getLogger(name)
        if level is not None:
            self._logger.setLevel(level)
        self._logger.addHandler(self.pbar_handler)
        self._logger.propagate = False
        self.debug = self._logger.debug
        self.info = self._logger.info
        self.warn = self._logger.warning
        self.warning = self._logger.warning
        self.error = self._logger.error
        self.critical = self._logger.critical
        self.exception = self._logger.exception

    def pbar(self, *args, **kwargs):
        kwargs["ascii"] = kwargs.get("ascii", True)

        pbar = tqdm(*args, **kwargs)
        self.pbar_handler.register(pbar)
        return pbar


def get_logger(name=None):
    return Logger(name, logging.DEBUG)
