import warnings
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from logger_config import logger


warnings.filterwarnings("ignore")

# URL backend
BACKEND_URL = "http://backend:8000"

# Заголовок приложения
st.title("Приложение для анализа сигналов ЭКГ и определения состояния "
         "здоровья человека")
logger.info("Приложение запущено")

# Сохранение состояния загруженного файла для обучения
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None
if "eda_performed" not in st.session_state:
    st.session_state["eda_performed"] = False
if "df_exploded" not in st.session_state:
    st.session_state["df_exploded"] = None
if "df3" not in st.session_state:
    st.session_state["df3"] = None
if "top20_diseases" not in st.session_state:
    st.session_state["top20_diseases"] = None
if "top15_diseases" not in st.session_state:
    st.session_state["top15_diseases"] = None
if "dataset_name_prediction" not in st.session_state:
    st.session_state["dataset_name_prediction"] = None


# Part 1: Загрузка файла
uploaded_file = st.file_uploader("Загрузите ZIP-файл для обучения",
                                 type=["zip"])
if uploaded_file is not None and st.session_state["dataset_name"] is None:
    logger.info("Файл загружен пользователем: %s", uploaded_file.name)
    # Отправляем файл на сервер только один раз
    files = {"file": (uploaded_file.name,
                      uploaded_file.read(),
                      "application/zip")}
    upload_response = requests.post(f"{BACKEND_URL}/upload_dataset",
                                    files=files, timeout=10000)

    if upload_response.status_code == 200:
        logger.info("Файл успешно отправлен на сервер: %s", uploaded_file.name)
        st.success("Файл успешно загружен!")
        st.session_state["dataset_name"] = (
            upload_response
            .json()["filepath"]
            .split("/")[-1]  # Сохраняем имя файла
        )

    else:
        logger.error("Ошибка загрузки файла на сервер: %s",
                     upload_response.text)
        st.error(f"Ошибка загрузки файла: {upload_response.text}")

# Part 2: EDA
st.divider()
st.subheader("Разведочный анализ данных")
dataset_name = st.session_state["dataset_name"]

# Проверка наличия загруженного файла
if st.session_state["dataset_name"]:
    st.write(f"Загруженный файл: {st.session_state['dataset_name']}")

    # Отображение кнопки только после загрузки файла
    if not st.session_state["eda_performed"]:
        if st.button("Провести EDA"):
            logger.info("Пользователь инициировал выполнение EDA для файла: "
                        "%s", dataset_name)
            with st.spinner("Обработка данных, пожалуйста, подождите..."):
                response = requests.get(
                    f"{BACKEND_URL}/get_eda_info",
                    data={
                        "dataset_name": st.session_state["dataset_name"]
                        .replace(".zip", "")
                    }, timeout=10000

                )
                if response.status_code == 200:
                    logger.info("EDA успешно выполнен для файла: %s",
                                dataset_name)
                    result = response.json()

                    # Сохраняем данные в session_state
                    st.session_state["df_exploded"] = \
                        pd.DataFrame(result["df_exploded"])
                    st.session_state["df3"] = pd.DataFrame(result["df3"])
                    st.session_state["top20_diseases"] = result["top_diseases"]
                    st.session_state["top15_diseases"] = \
                        result["top_2_diseases"]
                    st.session_state["eda_performed"] = True
                    st.success("EDA успешно выполнен!")
                else:
                    logger.error("Ошибка выполнения EDA: %s", response.text)
                    st.error(f"Ошибка: {response.status_code} — "
                             f"{response.text}")
    else:
        logger.info("EDA уже выполнен для файла: %s", dataset_name)
        st.success("EDA уже выполнен.")
else:
    logger.warning("Попытка выполнения EDA без загрузки файла")
    st.info("Пожалуйста, загрузите файл, чтобы произвести анализ.")

# Проверяем, загружены ли данные
if st.session_state["df_exploded"] is not None and st.session_state["df3"] \
        is not None and st.session_state["top20_diseases"] is not None \
        and st.session_state["top15_diseases"] is not None:

    df_exploded = st.session_state["df_exploded"]
    df3 = st.session_state["df3"]
    top20_diseases = st.session_state["top20_diseases"]
    top15_diseases = st.session_state["top15_diseases"]

    # Отображение описательной статистики
    st.markdown("Описательная статистика:")
    st.write(df_exploded.describe())

    # Построение гистограммы c количеством заболеваний на одного пациента
    hist = go.Histogram(
        x=df3['len_disease'],
        nbinsx=10,  # Количество бинов
        marker_color='#00CC96',
        opacity=0.75
    )

    fig1 = go.Figure(data=[hist])
    fig1.update_layout(
        title="Гистограмма распределения количества заболеваний у "
              "одного пациента",
        xaxis_title="Количество заболеваний",
        yaxis_title="Количество пациентов",
        bargap=0.2,  # Промежуток между барами
    )

    st.plotly_chart(fig1)

    # Группировка данных
    st.write('Количество женщин и мужчин среди пациентов:')
    st.write(df_exploded.groupby('gender').agg({'id': 'nunique'}))

    # Построение гистограммы распределения возраста в датасете
    fig2 = go.Figure()

    for gender in df_exploded['gender'].unique():
        if gender == 'Female':
            color = '#FFA15A'
        else:
            color = '#6FA5C9'

        fig2.add_trace(
            go.Histogram(
                x=df_exploded[df_exploded['gender'] == gender]['age'],
                name=f"{gender}",
                marker_color=color,
                opacity=0.75
            )
        )

    fig2.update_layout(
        title={"text": 'Гистограмма распределения возраста',
               "font": {"size": 16}},
        xaxis={"title": 'Возраст',
               "showgrid": False,
               "zeroline": False},
        yaxis={"title": 'Количество пациентов',
               "showgrid": True,
               "zeroline": False},
        bargap=0.2,  # Промежуток между столбцами
        barmode='overlay',  # Наложение столбцов
        plot_bgcolor='rgba(0,0,0,0)'  # Прозрачный фон
    )

    # Отображение графика в Streamlit
    st.plotly_chart(fig2)

    # Интерфейс для выбора заболевания и пола
    selected_disease = st.selectbox("Выберите заболевание", top20_diseases)
    gender_filter = st.radio("Выберите пол", ["Оба", "Мужчины", "Женщины"])

    # Фильтрация данных
    @st.cache_data
    def filter_data(disease, gender_filter_sp, df):
        """
        Filters the dataset based on disease name and gender.
        """
        df_filtered = df[df.disease_name == disease]
        if gender_filter != "Оба":
            gender_sp = "Male" if gender_filter_sp == "Мужчины" else "Female"
            df_filtered = df_filtered[df_filtered.gender == gender_sp]
        return df_filtered

    df_to_plot = filter_data(selected_disease, gender_filter, df_exploded)

    # Построение графика
    hist = go.Histogram(
        x=df_to_plot['age'],
        nbinsx=15,  # Количество бинов
        marker_color='#AB63FA',
        opacity=0.75
    )

    fig3 = go.Figure(data=[hist])
    fig3.update_layout(
        title=f'Гистограмма распределения возраста для {selected_disease} '
              f'({gender_filter})',
        xaxis_title="Возраст",
        yaxis_title="Количество пациентов",
        bargap=0.01,  # Промежуток между барами
    )

    st.plotly_chart(fig3)

    @st.cache_data
    def filter_data_disease(disease, df):
        """
        Filters the dataset based on disease name.
        """
        df_filtered = df[df.disease_name == disease]
        return df_filtered

    df_to_plot_gender = filter_data_disease(selected_disease, df_exploded)

    total_men_count = df_exploded[df_exploded.gender == 'Male'].shape[0]
    total_women_count = df_exploded[df_exploded.gender == 'Female'].shape[0]

    men_count = df_to_plot_gender[
        (df_to_plot_gender.gender == 'Male')
        & (df_exploded.disease_name == selected_disease)].shape[0]
    women_count = df_to_plot_gender[
        (df_to_plot_gender.gender == 'Female')
        & (df_exploded.disease_name == selected_disease)].shape[0]

    total_count = men_count + women_count

    men_ratio = men_count / total_count
    women_ratio = women_count / total_count

    fig4 = go.Figure()

    fig4.add_trace(
        go.Bar(
            x=['Мужчины', 'Женщины'],
            y=[men_ratio, women_ratio],
            marker_color=['#6FA5C9', '#FFA15A'],
            name='Доля пациентов'
        )
    )

    fig4.update_layout(
        title=f'Доля мужчин и женщин для {selected_disease}',  # Заголовок
        xaxis={"title": 'Пол', "tickmode": 'array'},
        yaxis={"title": 'Доля пациентов', "range": [0, 1]},
        plot_bgcolor='rgba(0,0,0,0)',  # Прозрачный фон
    )

    # Отображение графика в Streamlit
    st.plotly_chart(fig4)

    st.info("**На основе загруженных данных возможен прогноз следующих "
            "диагнозов:** " + ", ".join(top15_diseases))

# Part 3: Выбор модели и параметров
st.divider()
st.subheader("Обучение модели")
if uploaded_file is not None and st.session_state["dataset_name"] is not None:
    st.write("Выберите модель:")
    model_type = st.selectbox("Модель", ["SVC", "Logistic Regression"])

    params = {}
    if model_type == "SVC":
        # Гиперпараметры для SVC
        params["C"] = st.slider("C", 0.001, 100.0, 1.0)
        params["kernel"] = st.selectbox("Kernel", ["linear",
                                                   "rbf",
                                                   "poly",
                                                   "sigmoid"])
        params["gamma"] = st.selectbox("Gamma", ["scale",
                                                 "auto"])
        params["class_weight"] = st.selectbox("Class Weight", [None,
                                                               "balanced"])
    elif model_type == "Logistic Regression":
        # Гиперпараметры для Logistic Regression
        params["solver"] = st.selectbox("Solver", ["liblinear",
                                                   "saga",
                                                   "lbfgs"])
        params["penalty"] = st.selectbox("Penalty", ["l1",
                                                     "l2",
                                                     "elasticnet",
                                                     "none"])
        params["C"] = st.selectbox("C", [0.001, 0.01, 0.1, 1, 10, 100])
        params["class_weight"] = st.selectbox("Class Weight", [None,
                                                               "balanced"])

        # Параметр l1_ratio отображается, только если penalty = elasticnet
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        else:
            params["l1_ratio"] = None  # Если penalty != elasticnet, None

    # Кнопка для обучения модели
    if st.button("Обучить модель"):
        logger.info("Запуск обучения модели %s с параметрами %s",
                    model_type,
                    params)
        # Формируем запрос на обучение
        train_data = {
            "model_type": model_type,
            "params": params,
            "dataset_name": dataset_name.replace(".zip", "")
        }
        train_response = requests.post(f"{BACKEND_URL}/train_model",
                                       json=train_data, timeout=10000)

        if train_response.status_code == 200:
            logger.info("Модель %s успешно обучена", model_type)
            st.success("Модель успешно обучена")
            result = train_response.json()
            exp_name = result["experiment_name"]
            st.write(f"Имя эксперимента: {exp_name}")
        else:
            logger.error("Ошибка обучения модели: %s", train_response.text)
            st.error(f"Ошибка обучения модели: {train_response.text}")
else:
    logger.warning("Попытка обучения модели без загрузки файла")
    st.info("Пожалуйста, загрузите файл, чтобы обучить модель.")


# Part 4: Выбор модели и прогноз
# Сохранение состояния загруженного файла для прогноза
if "dataset_name_prediction" not in st.session_state:
    st.session_state["dataset_name_prediction"] = None
    logger.info("Инициализировано состояние для хранения имени "
                "файла прогноза.")

if "prediction_triggered" not in st.session_state:
    st.session_state["prediction_triggered"] = False

st.divider()
st.subheader("Прогноз по анализам ЭКГ")
exps_resp = requests.get(f"{BACKEND_URL}/experiments", timeout=10000)
if exps_resp.status_code == 200:
    logger.info("Успешно получен список обученных моделей с сервера.")
    if "experiments" in exps_resp.json():
        exps = exps_resp.json()["experiments"]
        print(exps)
        logger.info("Найдено %s обученных моделей.", str(len(exps)))
    else:
        logger.warning("Список обученных моделей пуст.")
        st.error("Обученные модели отсутствуют.")

    # Выбор модели
    ids = [item["id"] for item in exps]
    selected_model = st.selectbox("Выберите модель для предсказания", ids)
    logger.info("Выбрана модель %s для прогноза", selected_model)

    # Находим данные о выбранной модели
    model_data = next(item for item in exps if item["id"] == selected_model)

    # Отображение данных о модели
    st.write("#### Информация о выбранной модели")
    st.write("**Метрики:**")
    st.json(model_data["metrics"])
    st.write("**Параметры:**")
    st.json(model_data["params"])

    st.write("#### Получение прогноза о состоянии здоровья")
    uploaded_file_prediction = st\
        .file_uploader("Загрузите ZIP-файл для прогноза диагноза",
                       type=["zip"])
    if uploaded_file_prediction is not None \
            and st.session_state["dataset_name_prediction"] is None:
        logger.info("Файл для прогноза загружен: %s",
                    uploaded_file_prediction.name)
        st.session_state["dataset_name_prediction"] = \
            uploaded_file_prediction.name

    if uploaded_file_prediction is not None and st.button("Получить прогноз"):
        st.session_state["prediction_triggered"] = True

    if st.session_state["prediction_triggered"]:
        logger.info("Начат процесс получения прогноза.")
        # Подготовка данных для запроса
        files = {"file": (uploaded_file_prediction.name,
                          uploaded_file_prediction.read(),
                          "application/zip")}
        response = requests.post(f"{BACKEND_URL}/upload_inference",
                                 files=files,
                                 data={"model_name": selected_model},
                                 timeout=10000)

        if response.status_code == 200:
            logger.info("Предсказание завершено успешно.")
            result = response.json()
            disease_names = result["predicts"]
            disease_names_str = ', '.join(disease_names)
            st.success(f"Ваш прогноз: {disease_names_str}. "
                       f"Требуется консультация с врачом.")
        else:
            logger.error("Ошибка при выполнении предсказания: %s",
                         response.text)
            st.error(f"Ошибка при выполнении предсказания: {response.text}")
else:
    st.error("Ошибка получения списка обученных моделей")
