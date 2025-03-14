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
import utils
from logger_config import logger


warnings.filterwarnings("ignore")

RAND = 42


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
    X_train_std, X_test_std, y_train, y_test, sc = \
        utils.preprocess_dataset(df)
    logger.info("Preprocess is finished. Start fitting")
    if model_type == "Logistic Regression":
        model = LogisticRegression(multi_class='ovr',
                                   random_state=RAND,
                                   max_iter=1000,
                                   **params)
    elif model_type == "SVC":
        model = SVC(probability=True,
                    random_state=RAND,
                    **params)
    else:
        raise ValueError("Неизвестный тип модели")
    model.fit(X_train_std, y_train)
    logger.info("Fitting is finished. Start predicting")
    y_pred = model.predict(X_test_std)
    logger.info("Predicting is finished. Start count metrics")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train_std,
        y_train,
        cv=3,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=RAND
    )
    formatted_time = (datetime.now() + timedelta(hours=3))\
        .strftime("%H_%M_%d_%m_%Y")
    exp_dir = os.path.join("experiments",
                           f"{formatted_time}_{str(uuid.uuid4())[:8]}")
    os.makedirs(exp_dir, exist_ok=True)
    joblib.dump(model, os.path.join(exp_dir, "model.joblib"))
    joblib.dump(sc, os.path.join(exp_dir, "scaler.joblib"))
    pd.DataFrame([metrics]).to_csv(
        os.path.join(exp_dir, "metrics.csv"), index=False)
    curves = pd.DataFrame({
        "train_sizes": train_sizes.tolist(),
        "train_scores": np.mean(train_scores, axis=1).tolist(),
        "validation_scores": np.mean(val_scores, axis=1).tolist()
    })
    curves.to_csv(os.path.join(exp_dir, "curves.csv"), index=False)
    return f"{formatted_time}_{str(uuid.uuid4())[:8]}"


def get_inference(folder_path: str, model_name: str):
    """
        Performs inference using a specified model on preprocessed data.

        Args:
            folder_path (str): Path to the folder containing input data files.
            model_name (str): Name of the model to use for inference.

        Returns:
            list: A list of predicted disease names.
    """
    df = utils.make_df_from_mat_files(folder_path)
    X = utils.preprocess_dataset(df, True)[0]
    model_path = os.path.join(os.path.join(
        Path(folder_path).parent.parent, "experiments"),
                              model_name,
                              "model.joblib")
    model = joblib.load(model_path)
    predictions = model.predict(X)
    predictions = predictions.tolist()

    file_path = os.path.abspath(__file__)
    path = Path(file_path).parent
    file_above = path / "snomed-ct.csv"
    df_snomed = pd.read_csv(file_above, sep=',')
    df_snomed = df_snomed.rename(
        columns={'Dx': 'disease_name',
                 'SNOMED CT Code': 'labels',
                 'Abbreviation': 'short_disease_name'})
    disease_dict = dict(zip(df_snomed['labels'], df_snomed['disease_name']))
    disease_names = [disease_dict[disease_id] for disease_id in predictions]
    return disease_names


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
    list_of_metrics = ["accuracy", "f1"]
    for exp in list_of_experiments:
        experiment_path = os.path.join(exp_dir, exp)
        model_file = os.path.join(experiment_path, "model.joblib")
        metrics_file = os.path.join(experiment_path, "metrics.csv")
        model = joblib.load(model_file)
        metrics_df = pd.read_csv(metrics_file)
        result_metrics = {}
        for metric in list_of_metrics:
            result_metrics[metric] = metrics_df[metric][0]
        result_dict[exp] = [model, result_metrics]
        print(result_dict)
    return result_dict
    # return [d for d in os.listdir(exp_dir)
    #         if os.path.isdir(os.path.join(exp_dir, d))]


def get_experiment_metrics(name: str):
    """
        Retrieves metrics for a specific experiment.

        Args:
            name (str): Name of the experiment.

        Returns:
            dict: A dictionary of metrics for the experiment.
    """
    exp_dir = os.path.join("experiments", name)
    if not os.path.exists(exp_dir):
        raise ValueError("Эксперимент не найден")
    metrics_path = os.path.join(exp_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)
    return df.to_dict(orient="records")[0]


def get_experiment_curves(names: list):
    """
        Retrieves learning curves for a list of experiments.

        Args:
            names (list): List of experiment names.

        Returns:
            dict: A dictionary where keys are experiment names and values are
                  learning curve data.
    """
    curves_data = {}
    for n in names:
        exp_dir = os.path.join("experiments", n)
        if not os.path.exists(exp_dir):
            continue
        curves_path = os.path.join(exp_dir, "curves.csv")
        df = pd.read_csv(curves_path)
        curves_data[n] = {
            "train_sizes": df["train_sizes"].tolist(),
            "train_scores": df["train_scores"].tolist(),
            "validation_scores": df["validation_scores"].tolist()
        }
    return curves_data


def get_eda_info(dataset_name: str):
    """
        Retrieves exploratory data analysis (EDA) information for a dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            Any: EDA information from the utility function.
    """
    return utils.get_eda_info(dataset_name)
