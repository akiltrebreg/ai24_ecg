import json
import os
from pathlib import Path
import uuid
import warnings
from datetime import datetime, timedelta
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import utils
from logger_config import logger
import CNN_model
import torch


warnings.filterwarnings("ignore")

RAND = 42

def get_eda_info(dataset_name: str):
    """
        Retrieves exploratory data analysis (EDA) information for a dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            Any: EDA information from the utility function.
    """
    return utils.get_eda_info(dataset_name)


def train_model(model_type: str, params: dict, dataset_name: str):
    """
        Trains a machine learning model with the specified type and parameters.

        Args:
            model_type (str): The type of the model to train. Supported types:
                              "Logistic Regression", "SVC".
            params (dict): The hyperparameters for the model.
            dataset_name (str): The name of the dataset to use for training.

        Returns:
            str: The name of the experiment directory where model, scaler,
                 metrics, and learning curves are saved.

        Raises:
            ValueError: If the dataset is not found or the model type
            is invalid.
    """
    data_path = os.path.join("data", dataset_name)
    if not os.path.exists(data_path):
        raise ValueError("Датасет не найден")
    logger.info("Converting_mat_files_to_df...")
    df = utils.make_df_from_mat_files(data_path)
    logger.info("Preprocess is started")
    df_for_cnn, top_diseases = utils.preprocess_dataset(df)
    logger.info("Preprocess is finished. Start fitting")
    formatted_time = (datetime.now() + timedelta(hours=3)) \
        .strftime("%H_%M_%d_%m_%Y")
    exp_dir = os.path.join("experiments",
                           f"{formatted_time}_{str(uuid.uuid4())[:8]}")
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, f"top_diseases.txt"), 'w', encoding='utf-8') as f:
        for item in top_diseases:
            f.write(f"{item}\n")
    if model_type == "CNN":
        train_loader, val_loader, test_loader, pos_weight = \
            CNN_model.prepare_loaders(
            df_for_cnn,
            batch_size=32
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CNN_model.ECG1DCNN(n_leads=12,
                         num_classes=len(df_for_cnn['labels_target'][0]))
        CNN_model.train_model(model, train_loader, val_loader, device,
                              epochs=params['epochs'], lr=params['lr'],
                              pos_weight=pos_weight,
                              save_path=os.path.join(exp_dir, "model.pth"),
                              average=params['average'],
                              path_for_hp = os.path.join(exp_dir, "hyperparams.hp"))
    else:
        raise ValueError("Неизвестный тип модели")
    logger.info("Fitting is finished. Start predicting")
    metrics = CNN_model.test_model(model, test_loader, device,
                         checkpoint=os.path.join(exp_dir, "model.pth"),
                         average=params['average'])
    logger.info("Predicting is finished. Start count metrics")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(exp_dir, "metrics.csv"), index=False)

    joblib.dump(model, os.path.join(exp_dir, "model.joblib"))
    return f"{formatted_time}_{str(uuid.uuid4())[:8]}"


def list_experiments():
    """
        Lists all experiments with their models and metrics.

        Returns:
            dict: A dictionary where keys are experiment names, and values are
                  lists containing the model and metrics.
    """
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        return []
    list_of_experiments = [d for d in os.listdir(exp_dir)
                           if os.path.isdir(os.path.join(exp_dir, d))
                           and os.listdir(os.path.join(exp_dir, d))]
    result_dict = {}
    list_of_metrics = ['Loss', 'Acc', 'AUC', 'Prec', 'Rec', 'F1']
    list_of_metrics_last_version = ['accuracy', 'f1']
    for exp in list_of_experiments:
        experiment_path = os.path.join(exp_dir, exp)
        model_file = os.path.join(experiment_path, "model.joblib")
        metrics_file = os.path.join(experiment_path, "metrics.csv")
        model = joblib.load(model_file)
        params = {}
        if isinstance(model, LogisticRegression):
            params["model"] = "Logistic Regression"
            params["solver"] = str(model.solver)
            params["penalty"] = str(model.penalty)
            params["C"] = str(model.C)
            params["class_weight"] = str(model.class_weight)
            if hasattr(model, 'l1_ratio') and model.l1_ratio is not None:
                params["l1_ratio"] = str(model.l1_ratio)
        elif isinstance(model, SVC):
            params["model"] = "SVC"
            params["C"] = str(model.C)
            params["kernel"] = str(model.kernel)
            params["gamma"] = str(model.gamma)
            params["class_weight"] = str(model.class_weight)
        else:
            logger.info(os.path.join(experiment_path, "hyperparams.hp"))
            with open(os.path.join(experiment_path, "hyperparams.hp"), "r", encoding="utf-8") as fp:
                params = json.load(fp)
            params["model"] = "CNN"
        metrics_df = pd.read_csv(metrics_file)
        result_metrics = {}
        if list_of_metrics[0] in list(metrics_df.columns):
            for metric in list_of_metrics:
                result_metrics[metric] = metrics_df[metric][0]
        else:
            for metric in list_of_metrics_last_version:
                result_metrics[metric] = metrics_df[metric][0]

        result_dict[exp] = [result_metrics, params]
    return result_dict


def get_inference(folder_path: str, model_name: str, ):
    """
        Performs inference using a specified model on preprocessed data.

        Args:
            folder_path (str): Path to the folder containing input data files.
            model_name (str): Name of the model to use for inference.

        Returns:
            list: A list of predicted disease names.
    """
    df = utils.make_df_from_mat_files(folder_path)
    X = utils.preprocess_dataset(df, for_inference = True)[0]
    max_len = max(len(sig[0]) for sig in X['signal'])
    loader = DataLoader(CNN_model.ECGInferenceDataset(X, max_len), batch_size=32)
    model_path = os.path.join(os.path.join(
        Path(folder_path).parent.parent, "experiments"),
        model_name,
        "model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_model.ECG1DCNN(n_leads=12, num_classes=30).to(device)
    predictions_ohe = CNN_model.predict(model, loader, model_path, device)[1]
    top_diseases_path = os.path.join(os.path.join(
        Path(folder_path).parent.parent, "experiments"),
        model_name,
        "top_diseases.txt")
    with open(top_diseases_path, 'r', encoding='utf-8') as f:
        top_diseases = [line.rstrip('\n') for line in f]
    df_snomed = pd.read_csv(Path(__file__).parent / "snomed-ct.csv", sep=',')
    df_snomed = df_snomed.rename(columns={
        'Dx': 'disease_name',
        'SNOMED CT Code': 'labels'
    })
    disease_dict = dict(zip(df_snomed['labels'], df_snomed['disease_name']))

    results: list[list] = []
    for row in predictions_ohe:
        codes = [top_diseases[i] for i, flag in enumerate(row) if flag == 1]
        names = [disease_dict.get(int(code)) for code in codes]
        results.append(names)
    logger.info(results)
    return results

