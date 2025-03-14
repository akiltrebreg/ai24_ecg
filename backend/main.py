import os
import shutil
import warnings
from typing import List
from typing_extensions import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import utils
from utils import ALLOWED_EXTENSIONS
from training import (
    train_model,
    list_experiments,
    get_experiment_metrics,
    get_experiment_curves,
)
from logger_config import logger
import training


warnings.filterwarnings("ignore")


app = FastAPI()


@app.post("/upload_dataset")
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> dict:
    """
        Handles the upload of a dataset file
        Args:
            file (Annotated[UploadFile, File(...)]): The uploaded dataset file
        Returns:
            dict
        """
    filename = file.filename
    if not any(filename.endswith(ext)
               for ext in ALLOWED_EXTENSIONS):
        logger.error("Попытка загрузить неправильный формат файла: "
                     "%s", filename)
        raise HTTPException(status_code=400,
                            detail="Неверный формат файла. Допустим: csv")
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        logger.error("Размер файла превышает 200 МБ")
        raise HTTPException(status_code=400, detail="Файл слишком большой")
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info("Файл %s успешно загружен", filename)
    folder_path = file_path[:file_path.rfind(".")]
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    logger.info(file_path + "|" + folder_path)
    shutil.unpack_archive(file_path, folder_path)
    logger.info("EDA...")
    utils.create_eda(folder_path)
    logger.info("...EDA")
    logger.info("Файл %s успешно разархивирован", filename)
    return {"message": "Файл успешно загружен", "filepath": file_path}


@app.post("/upload_inference")
async def upload_inference(
        file: Annotated[UploadFile, File(...)],
        model_name: Annotated[str, Form(...)]
) -> dict:
    """
        Handles file upload and performs inference using the specified model.

        Args:
            file: Inference file (csv or allowed archive).
            model_name: Model name for predictions.

        Returns:
            dict: Success message and predictions.
    """
    filename = file.filename
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        logger.error("Попытка загрузить неправильный формат файла: %s",
                     filename)
        raise HTTPException(status_code=400,
                            detail="Неверный формат файла. Допустим: csv")
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        logger.error("Размер файла превышает 200 МБ")
        raise HTTPException(status_code=400, detail="Файл слишком большой")
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info("Файл %s успешно загружен", filename)
    folder_path = file_path[:file_path.rfind(".")]
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    logger.info(file_path + "|" + folder_path)
    shutil.unpack_archive(file_path, folder_path)
    logger.info("Файл %s успешно разархивирован", filename)
    predicts = training.get_inference(folder_path, model_name)
    return {"message": "Предсказание успешно завершено", "predicts": predicts}


class TrainRequest(BaseModel):
    model_type: str
    params: dict
    dataset_name: str


@app.post("/train_model")
def train_model_endpoint(req: TrainRequest):
    """
    Endpoint for training a model.

    Args:
        req (TrainRequest): The training request containing model type,
                            parameters, and dataset name.

    Returns:
        dict: A success message and the name of the experiment.
    """
    try:
        exp_name = train_model(req.model_type, req.params, req.dataset_name)
        logger.info(
            "Модель %s успешно обучена. Эксперимент: %s",
            req.model_type,
            exp_name,
        )
        return {
            "message": "Модель успешно обучена",
            "experiment_name": exp_name
        }
    except Exception as e:
        logger.error("Ошибка обучения модели: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/experiments")
def get_experiments() -> dict:
    """
        Fetches a list of experiments with their metrics and parameters.

        Returns:
            dict: A dictionary containing a list of experiments with details
                  like metrics and model parameters.
    """
    res = list_experiments()
    result = []
    for elem, details in res.items():
        entry = {"id": elem, "metrics": {}, "params": {}}
        for el in details:
            if isinstance(el, dict):
                entry["metrics"] = el
            elif isinstance(el, LogisticRegression):
                entry["params"]["model"] = "Logistic Regression"
                entry["params"]["solver"] = str(el.solver)
                entry["params"]["penalty"] = str(el.penalty)
                entry["params"]["C"] = str(el.C)
                entry["params"]["class_weight"] = str(el.class_weight)
                if hasattr(el, 'l1_ratio') and el.l1_ratio is not None:
                    entry["params"]["l1_ratio"] = str(el.l1_ratio)
            elif isinstance(el, SVC):
                entry["params"]["model"] = "SVC"
                entry["params"]["C"] = str(el.C)
                entry["params"]["kernel"] = str(el.kernel)
                entry["params"]["gamma"] = str(el.gamma)
                entry["params"]["class_weight"] = str(el.class_weight)

        result.append(entry)

    logger.info("Запрошен список экспериментов")
    return {"experiments": result}


@app.get("/experiment_metrics")
def experiment_metrics(name: Annotated[str, Form(...)]) -> dict:
    """Get metrics for an experiment."""
    return get_experiment_metrics(name)


@app.get("/experiment_curves")
def experiment_curves(names: Annotated[List[str], Form(...)]) -> dict:
    """Get learning curves for experiments."""
    return get_experiment_curves(names)


@app.get("/get_eda_info")
def get_eda_info(dataset_name: Annotated[str, Form(...)]) -> dict:
    """
    Fetches exploratory data analysis information for the specified dataset.

    Args:
        dataset_name (Annotated[str, Form(...)]):
        The name of the dataset to analyze.

    Returns:
        dict: A dictionary containing the following keys:
            - "df3":
            A list of records representing the summarized dataset (df3).
            - "df_exploded":
            A list of records representing the exploded dataset (df_exploded).
            - "top_diseases":
            A list of the top diseases based on the analysis.
            - "top_2_diseases":
            A list of the top 2 diseases based on the analysis.
    """
    df3, df_exploded, top_diseases, top_2_diseases = \
        training.get_eda_info(dataset_name)
    return {
        "df3": df3.to_dict(orient="records"),
        "df_exploded": df_exploded.to_dict(orient="records"),
        "top_diseases": [
            int(x) if isinstance(x, np.integer) else x
            for x in top_diseases
        ],
        "top_2_diseases": [
            int(x) if isinstance(x, np.integer) else x
            for x in top_2_diseases
        ],
    }
