import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Union

from fastapi import BackgroundTasks, Depends, FastAPI

from api.dependencies import verify_auth_token
from api.schemas import ModelInput, ModelPrediction, ModelSpec, ProcessDetails
from api.utils import Process, map_training_errors, prepare_logger
from model.model import RepresentativenessModel

logger = prepare_logger()


MAIN_PATH = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Creates 'models' directory before the app starts up
    Removes 'models' directory with its contents before shutdown
    Removes 'status.json' before shutdown
    """
    os.makedirs(MAIN_PATH / "models", exist_ok=True)
    yield
    shutil.rmtree(MAIN_PATH / "models")
    if os.path.exists(MAIN_PATH / "status.json"):
        os.remove(MAIN_PATH / "status.json")


app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(verify_auth_token)],
)

model = RepresentativenessModel(models_path=MAIN_PATH / "models")

process = Process.load_status()


def update_process_info(
    event: str, error: str | None = None, model_token: str | None = None
) -> None:
    """
    Updates current process details and removes redundant models
    - event: event name
    - error: error message if erro occurred
    - model_token: model identification token
    """
    process["details"] = Process.details(event)

    def drop_items(keys: list):
        for k in keys:
            process.pop(k, None)

    match event:
        case "started":
            process["start_time"] = datetime.now().isoformat()
            process["in_progress"] = True
            drop_items(["end_time", "error_time", "error"])
        case "ended":
            process["end_time"] = datetime.now().isoformat()
            process["in_progress"] = False
            process["prod_model"] = model_token
            drop_items(["error_time", "error"])
        case "failed":
            process["in_progress"] = False
            process["error_time"] = datetime.now().isoformat()
            process["error"] = error
            drop_items(["end_time"])
        case "in_progress":
            process["in_progress"] = True
            drop_items(["end_time", "error_time", "error"])
        case _:
            pass

    Process.save_status(process)
    model.delete(keep=[process.get("prod_model")])


def train_model_task(model_spec: ModelSpec) -> None:
    """
    Starts new model training in the background
    - model_spec: new model specification
    """
    try:
        token = model.make_ensemble(
            data=model_spec.data,
            n_split=model_spec.n_split,
            n_nearest=model_spec.n_nearest,
        )
    except Exception as e:
        error_msg = map_training_errors(error=e.__str__())
        logger.error(f"Training failed. {error_msg}")
        update_process_info(event="failed", error=error_msg)
    else:
        update_process_info(event="ended", model_token=token)


@app.get("/")
async def root():
    return {"message": "Welcome to 'Representativeness model' service."}


@app.post("/train", response_model=ProcessDetails, response_model_exclude_unset=True)
async def train(model_spec: ModelSpec, tasks: BackgroundTasks):
    if not process.get("in_progress"):
        update_process_info(event="started")
        tasks.add_task(train_model_task, model_spec)
    else:
        update_process_info(event="in_progress")
    return ProcessDetails(**process)


@app.get("/status", response_model=ProcessDetails, response_model_exclude_unset=True)
async def check_status():
    if process.get("in_progress"):
        update_process_info(event="in_progress")
    return ProcessDetails(**process)


@app.post(
    "/predict",
    response_model=Union[ModelPrediction, ProcessDetails],
    response_model_exclude_unset=True,
)
async def predict(input: ModelInput):
    if not (token := process.get("prod_model")):
        return ProcessDetails(details="No models trained yet")
    if not model.models:
        model.load_models(token=token)
    return ModelPrediction(
        model=token,
        prediction=model.predict(input.data),
    )
