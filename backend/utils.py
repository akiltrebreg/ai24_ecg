import logging
import os
import shutil
import warnings
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tsfel import feature_extraction
from logger_config import logger
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

warnings.filterwarnings("ignore")
RAND = 42
ALLOWED_EXTENSIONS = ['.zip']
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def make_df_from_mat_files(path: str):
    """
        Processes .mat files in the given directory and returns a DataFrame
        containing ECG signal data and associated metadata.
    """
    def import_key_data(path):
        signal = []
        id_s = []
        time = []
        prefix = []
        one = []
        two = []
        three = []
        aVR = []
        aVL = []
        aVF = []
        V1 = []
        V2 = []
        V3 = []
        V4 = []
        V5 = []
        V6 = []
        age = []
        gender = []
        labels = []
        ecg_filenames = []
        r = []
        h = []
        x = []
        for subdir, _, files in sorted(os.walk(path)):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".mat"):
                    data, header_data = load_challenge_data(filepath)

                    signal.append(data)

                    if len(header_data[0][:-1]) > 0:
                        id_s.append(header_data[0][0:6])
                    else:
                        id_s.append(np.nan)

                    if len(header_data[0][:-1]) > 0:
                        time.append(header_data[0][11:-1])
                    else:
                        time.append(np.nan)

                    if len(header_data[1][:-1]) > 0:
                        prefix.append(header_data[1][11:29])
                    else:
                        prefix.append(np.nan)

                    if len(header_data[1][:-1]) > 0:
                        one.append(header_data[1][29:-1].replace(' I', ''))
                    else:
                        one.append(np.nan)

                    if len(header_data[2][:-1]) > 0:
                        two.append(header_data[2][29:-1].replace(' II', ''))
                    else:
                        two.append(np.nan)

                    if len(header_data[3][:-1]) > 0:
                        three.append(header_data[3][29:-1].replace(' III', ''))
                    else:
                        three.append(np.nan)

                    if len(header_data[4][:-1]) > 0:
                        aVR.append(header_data[4][29:-1].replace(' aVR', ''))
                    else:
                        aVR.append(np.nan)

                    if len(header_data[5][:-1]) > 0:
                        aVL.append(header_data[5][29:-1].replace(' aVL', ''))
                    else:
                        aVL.append(np.nan)

                    if len(header_data[6][:-1]) > 0:
                        aVF.append(header_data[6][29:-1].replace(' aVF', ''))
                    else:
                        aVF.append(np.nan)

                    if len(header_data[7][:-1]) > 0:
                        V1.append(header_data[7][29:-1].replace(' V1', ''))
                    else:
                        V1.append(np.nan)

                    if len(header_data[8][:-1]) > 0:
                        V2.append(header_data[8][29:-1].replace(' V2', ''))
                    else:
                        V2.append(np.nan)

                    if len(header_data[9][:-1]) > 0:
                        V3.append(header_data[9][29:-1].replace(' V3', ''))
                    else:
                        V3.append(np.nan)

                    if len(header_data[10][:-1]) > 0:
                        V4.append(header_data[10][29:-1].replace(' V4', ''))
                    else:
                        V4.append(np.nan)

                    if len(header_data[11][:-1]) > 0:
                        V5.append(header_data[11][29:-1].replace(' V5', ''))
                    else:
                        V5.append(np.nan)

                    if len(header_data[12][:-1]) > 0:
                        V6.append(header_data[12][29:-1].replace(' V6', ''))
                    else:
                        V6.append(np.nan)

                    if len(header_data[15][5:-1]) > 0:
                        labels.append(header_data[15][5:-1])
                    else:
                        labels.append(np.nan)

                    ecg_filenames.append(filepath)

                    if len(header_data[14][6:-1]) > 0:
                        gender.append(header_data[14][6:-1])
                    else:
                        gender.append(np.nan)

                    if len(header_data[13][6:-1]) > 0:
                        age.append(header_data[13][6:-1])
                    else:
                        age.append(np.nan)

                    if len(header_data[16][:-1].split(' ')[1]) > 0:
                        r.append(header_data[16][:-1].split(' ')[1])
                    else:
                        r.append(np.nan)

                    if len(header_data[17][:-1].split(' ')[1]) > 0:
                        h.append(header_data[17][:-1].split(' ')[1])
                    else:
                        h.append(np.nan)

                    if len(header_data[18][:-1].split(' ')[1]) > 0:
                        x.append(header_data[18][:-1].split(' ')[1])
                    else:
                        x.append(np.nan)

        return signal, id, time, prefix, one, two, three, aVR, aVL, aVF, V1, \
            V2, V3, V4, V5, V6, gender, age, labels, ecg_filenames, r, h, x

    def load_challenge_data(filename):
        x = loadmat(filename)
        data = np.asarray(x['val'], dtype=np.float64)
        new_file = filename.replace('.mat', '.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r', encoding='utf-8') as f:
            header_data = f.readlines()
        return data, header_data

    signal, id_s, time, prefix, one, two, three, aVR, aVL, aVF, V1, V2, V3, V4, \
        V5, V6, gender, age, labels, ecg_filenames, r, h, x = \
        import_key_data(path)

    df = pd.DataFrame(
        {
            'id': id_s,
            'time': time,
            'prefix': prefix,
            'one': one,
            'two': two,
            'three': three,
            'aVR': aVR,
            'aVL': aVL,
            'aVF': aVF,
            'V1': V1,
            'V2': V2,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'V6': V6,
            'gender': gender,
            'age': age,
            'labels': labels,
            'signal': signal,
            'ecg_filename': ecg_filenames,
            'r': r,
            'h': h,
            'x': x
        }
    )
    return df


def create_eda(folder_path: str):
    """
       Performs exploratory data analysis (EDA) on ECG signal data stored in .mat files and exports processed insights.
    """

    logger.info("EDA is_starting. path = %s", folder_path)
    eda_path = folder_path + "_eda"
    if os.path.exists(eda_path):
        shutil.rmtree(eda_path)
    os.mkdir(eda_path)
    df = make_df_from_mat_files(folder_path)
    logger.info("shape = %s", df.shape)
    df2 = df.drop(['time', 'prefix'], axis=1)
    col_names = ['one', 'two', 'three', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3',
                 'V4', 'V5', 'V6']
    for col_name in col_names:
        info_first = []
        for el in df[col_name]:
            info_first.append(list(map(int, el.split()))[0])
        df2[col_name] = info_first
    df2 = df2.drop(['r', 'h', 'x'], axis=1)  # Всегда unknown
    df2['labels'] = df2['labels'].apply(lambda x: list(map(int, x.split(','))))
    df2['labels'] = df2['labels']\
        .apply(lambda x: list(set(x)) if isinstance(x, list) else x)
    df2 = df2.drop(columns=['ecg_filename'])
    file_path = os.path.abspath(__file__)
    path = Path(file_path).parent
    file_above = path / "snomed-ct.csv"
    df_diseases = pd.read_csv(file_above, sep=',')
    df_diseases_code = df_diseases.rename(
        columns={'Dx': 'disease_name',
                 'SNOMED CT Code': 'labels',
                 'Abbreviation': 'short_disease_name'})
    df_exploded = df2.explode('labels', ignore_index=True)
    df_exploded = df_exploded.merge(df_diseases_code, how='left', on='labels')
    df_exploded['age'] = df_exploded['age'].replace({'NaN': 0, np.nan: 0})
    df_exploded.age = df_exploded.age.apply(int)
    median_value_male = df_exploded[(df_exploded.gender == 'Male')
                                    & (df_exploded.age != 0)]['age'].median()
    df_exploded.loc[df_exploded.gender == 'Male', 'age'] = \
        df_exploded.loc[df_exploded.gender == 'Male', 'age'].replace(
        0, median_value_male)
    df_exploded.loc[df_exploded.gender == 'Female', 'age'] = df_exploded.loc[
        df_exploded.gender == 'Female', 'age'].replace(0, median_value_male)
    df_exploded.groupby(['disease_name', 'short_disease_name'])\
        .agg({'id': 'nunique'}).sort_values(by='id', ascending=False)
    df3 = df2
    df3['len_disease'] = df3['labels'].apply(len)

    logger.info("EDA PART 1")
    df3.drop(columns=['signal', 'age'], axis=1)\
        .to_csv(os.path.join(eda_path, "df3.csv"), index=False)
    logger.info("EDA PART 2")
    df_exploded.drop('signal', axis=1)\
        .to_csv(os.path.join(eda_path, "df_exploded.csv"), index=False)
    logger.info("EDA PART 3")
    df = df_exploded
    top = 20
    top_2 = 15
    top_diseases = (
        df_exploded.groupby('disease_name')
        .id.nunique()
        .sort_values(ascending=False)[:top]
        .reset_index()
        .disease_name
        .tolist()
    )

    list_top_diseases = pd.DataFrame({'ListValues': top_diseases})
    logger.info("EDA PART 4")
    list_top_diseases.to_csv(os.path.join(eda_path, "top_diseases.csv"),
                             index=False)
    logger.info("EDA PART 5")
    df_preprocessed = df.drop(['id'], axis=1)
    df_preprocessed['gender'] = df_preprocessed['gender'].replace({'Female': 1,
                                                                   'Male': 0})
    df_preprocessed = df_preprocessed.dropna()
    df_final = df_preprocessed[df_preprocessed.disease_name.isin(top_diseases)]
    df_cropped = df_final[df_final.disease_name.isin(top_diseases)]
    subset = list(df_cropped.columns)
    subset.remove('labels')
    subset.remove('short_disease_name')
    subset.remove('signal')
    subset.remove('disease_name')
    df_cropped_2 = df_cropped.drop_duplicates(subset=subset, keep='first')
    top_2_diseases = (
        df_cropped_2.groupby('disease_name')
        .one
        .count()
        .sort_values(ascending=False)[:top_2]
        .reset_index()
        .disease_name
        .tolist()
    )

    list_top_2_diseases = pd.DataFrame({'ListValues': top_2_diseases})
    logger.info("EDA PART 6")
    list_top_2_diseases.to_csv(os.path.join(eda_path, "top_2_diseases.csv"),
                               index=False)
    logger.info("EDA ended")


def get_eda_info(dataset_name: str):
    """
        Retrieves preprocessed EDA information for a specified dataset.
    """
    file_path = os.path.abspath(__file__)
    path_part = "data/" + dataset_name + "_eda"
    path_parent = Path(file_path).parent
    path = path_parent / path_part
    logger.info("FIND_EDA... PATH = %s", path)
    df3 = pd.read_csv(path / 'df3.csv')
    df_exploded = pd.read_csv(path / 'df_exploded.csv')
    top_diseases = pd.read_csv(path / 'top_diseases.csv')['ListValues']\
        .tolist()
    top_2_diseases = pd.read_csv(path / 'top_2_diseases.csv')['ListValues']\
        .tolist()
    return [df3, df_exploded, top_diseases, top_2_diseases]


def preprocess_dataset(df: pd.DataFrame, for_inference = False):
    """
            Preprocesses a given dataset of ECG signals and prepares it for
             machine learning tasks.
    """
    logging.info("Размер предобрабатываемого датасета: %s", df.shape)

    df_full = pd.DataFrame(
        {
            'id': df['id'],
            'labels': df['labels'],
            'signal': df['signal']
        }
    )
    valid_mask = df_full['signal'].apply(
        lambda sig: isinstance(sig, (list, np.ndarray))
                    and len(sig) == 12
                    and all(
            isinstance(ch, (list, np.ndarray)) and len(ch) == 5000 for ch in
            sig)
    )
    df_full = df_full[valid_mask].reset_index(drop=True)
    top = []
    if not for_inference:
        df_full['labels_list'] = (
            df_full['labels']
            .apply(lambda s: [int(x) for x in s.split(',') if x])
        )
        TOP = 30
        all_labels = Counter(
            label for lst in df_full['labels_list'] for label in lst)
        top = [label for label, _ in all_labels.most_common(TOP)]

        df_full['labels_top'] = (
            df_full['labels_list']
            .apply(lambda lst: [x for x in lst if x in top])
        )

        mlb = MultiLabelBinarizer(classes=top)
        ohe = mlb.fit_transform(df_full['labels_top'])

        ohe_df = pd.DataFrame(ohe, columns=[f"lbl_{c}" for c in mlb.classes_],
                              index=df_full.index)
        df_full = pd.concat([df_full, ohe_df], axis=1)

        lbl_cols = [col for col in df_full.columns if col.startswith('lbl_')]

        df_full['labels_target'] = df_full[lbl_cols].values.tolist()

        df_for_cnn = df_full[['signal', 'labels_target']]
    else:
        df_for_cnn = df_full[['signal']]
    return df_for_cnn, top
