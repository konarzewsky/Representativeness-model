import json
import logging
import sys
from pathlib import Path

from api.env import LOG_LEVEL


def prepare_logger():
    logger = logging.getLogger()
    logger.setLevel(level=LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s"
    )
    if not logger.handlers:
        lh = logging.StreamHandler(sys.stdout)
        lh.setFormatter(formatter)
        logger.addHandler(lh)
    return logger


class Process:
    status_path: Path = Path(__file__).resolve().parents[1] / "status.json"
    details_dict: dict = {
        "started": "New training started",
        "in_progress": "Training in progress",
        "ended": "Training successfully completed",
        "failed": "Error occurred during training",
    }

    @classmethod
    def save_status(cls, data: dict) -> None:
        with open(cls.status_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    @classmethod
    def load_status(cls) -> dict:
        try:
            with open(cls.status_path, "r") as json_file:
                status = json.load(json_file)
        except FileNotFoundError:
            status = {"details": "No training recorded so far", "in_progress": False}
            cls.save_status(status)
        return status

    @classmethod
    def details(cls, event: str) -> str:
        return cls.details_dict.get(event, "")


def map_training_errors(error: str):
    match error:
        case x if "could not convert" in x:
            return "Provided data could not be converted to numeric format"
        case x if "incompatible" in x:
            return "Shapes or formats of provided data are incompatible"
        case x if "at least one array" in x:
            return "No data provided"
        case _:
            return error
