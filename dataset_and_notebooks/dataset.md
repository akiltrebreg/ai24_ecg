### Описание данных 
Набор данных ЭКГ был получен из [датасета](https://drive.google.com/drive/folders/1EI8XNLSLe8Hs2_HM0K_aQ5C5h2focnC5?usp=sharing) для соревнования PhysioNet 2020 года. <br>
Представленные анализы принадлежат жителям штата Джорджия, США. <br>
Набор содержит 10 344 показания ЭКГ в 12 отведениях длиной 10 секунд с частотой дискретизации 500 Гц. <br>
Заключения профессиональных экспертов по данным электрокардиограммам содержат как различные сердечные заболевания, так и вердикт о хорошей деятельности сердечной мышцы (синусовый ритм). <br>

Результат по каждому пациенту хранится в виде двух файлов следующих форматов:
1) «.hea», который содержит короткий текст, описывающий физиологический сигнал
2) «.mat», содержащий сигнал в виде массива значений и метаданные анализа

Используемый в проекте датасет был собран из предварительной обработанных «.mat» файлов.

Датасет содержал в себе исключительно кодировки заболеваний и был дополнен расшифровкой SNOMED CT. Это один из стандартов США, предназначенный для электронного обмена клинической медицинской информацией. Каждому коду заболевания соответствует установленные полное и краткое наименования.

### Полученный датафрейм с ЭКГ пациентов содержит следующие колонки:
 1. id – уникальный идентификатор ЭКГ пациента, строковый тип
 2. one – первое значение сигнала первого отведения, числовой тип
 3. two – первое значение сигнала второго отведения, числовой тип
 4. three – первое значение сигнала третьего отведения, числовой тип
 5. aVR – первое значение сигнала четвертого отведения, числовой тип
 6. aVL – первое значение сигнала пятого отведения, числовой тип
 7. aVF – первое значение сигнала шестого отведения, числовой тип
 8. V1 – первое значение сигнала седьмого отведения, числовой тип
 9. V2 – первое значение сигнала восьмого отведения, числовой тип
 10. V3 – первое значение сигнала девятого отведения, числовой тип
 11. V4 – первое значение сигнала десятого отведения, числовой тип
 12. V5 – первое значение сигнала одиннадцатого отведения, числовой тип
 13. V6 – первое значение сигнала двенадцатого отведения, числовой тип
 14. gender – пол пациента, строковый тип
 15. age – возраст пациента, числовой тип
 16. labels – код заболевания, числовой тип
 17. signal – массив с численными значениями сигнала
 18. disease_name – наименование заболевания, строковый тип
 19. short_disease_name – аббревиатура заболевания, строковый тип

<br><br>
Также были выделены характеристики сигналов для каждого отведения:

**Спектральная энтропия**

one_spectral_entropy – спектральная энтропия для первого отведения <br>
two_spectral_entropy – спектральная энтропия для второго отведения <br>
three_spectral_entropy – спектральная энтропия для третьего отведения <br>
aVR_spectral_entropy – спектральная энтропия для четвертого отведения <br>
aVL_spectral_entropy – спектральная энтропия для пятого отведения <br>
aVF_spectral_entropy – спектральная энтропия для шестого отведения <br>
V1_spectral_entropy – спектральная энтропия для седьмого отведения <br>
V2_spectral_entropy – спектральная энтропия для восьмого отведения <br>
V3_spectral_entropy – спектральная энтропия для девятого отведения <br>
V4_spectral_entropy – спектральная энтропия для десятого отведения <br>
V5_spectral_entropy – спектральная энтропия для одиннадцатого отведения <br>
V6_spectral_entropy – спектральная энтропия для двенадцатого отведения <br>

**Спектральная дисперсия**

one_spectral_variation <br>
two_spectral_variation <br>
three_spectral_variation <br>
aVR_spectral_variation <br>
aVL_spectral_variation <br>
aVF_spectral_variation <br>
V1_spectral_variation <br>
V2_spectral_variation <br>
V3_spectral_variation <br>
V4_spectral_variation <br>
V5_spectral_variation <br>
V6_spectral_variation <br>

**Mel-частотные кепстральные коэффициенты**

one_mfcc <br>
two_mfcc <br>
three_mfcc <br>
aVR_mfcc <br>
aVL_mfcc <br>
aVF_mfcc <br>
V1_mfcc <br>
V2_mfcc <br>
V3_mfcc <br>
V4_mfcc <br>
V5_mfcc <br>
V6_mfcc <br>

**Уменьшение амплитуды**

one_spectral_decrease <br>
two_spectral_decrease <br>
three_spectral_decrease <br>
aVR_spectral_decrease <br>
aVL_spectral_decrease <br>
aVF_spectral_decrease <br>
V1_spectral_decrease <br>
V2_spectral_decrease <br>
V3_spectral_decrease <br>
V4_spectral_decrease <br>
V5_spectral_decrease <br>
V6_spectral_decrease <br>

**Среднее абсолютное отклонение**

one_mean_abs_diff <br>
two_mean_abs_diff <br>
three_mean_abs_diff <br>
aVR_mean_abs_diff <br>
aVL_mean_abs_diff <br>
aVF_mean_abs_diff <br>
V1_mean_abs_diff <br>
V2_mean_abs_diff <br>
V3_mean_abs_diff <br>
V4_mean_abs_diff <br>
V5_mean_abs_diff <br>
V6_mean_abs_diff <br>

**Среднее значение разностей** 

one_mean_diff <br>
two_mean_diff <br>
three_mean_diff <br>
aVR_mean_diff <br>
aVL_mean_diff <br>
aVF_mean_diff <br>
V1_mean_diff <br>
V2_mean_diff <br>
V3_mean_diff <br>
V4_mean_diff <br>
V5_mean_diff <br>
V6_mean_diff <br>

**Абсолютная энергия**

one_abs_energy <br>
two_abs_energy <br>
three_abs_energy <br>
aVR_abs_energy <br>
aVL_abs_energy <br>
aVF_abs_energy <br>
V1_abs_energy <br>
V2_abs_energy <br>
V3_abs_energy <br>
V4_abs_energy <br>
V5_abs_energy <br>
V6_abs_energy <br>

**Энтропия**

one_enthropy <br>
two_enthropy <br>
three_enthropy <br>
aVR_enthropy <br>
aVL_enthropy <br>
aVF_enthropy <br>
V1_enthropy <br>
V2_enthropy <br>
V3_enthropy <br>
V4_enthropy <br>
V5_enthropy <br>
V6_enthropy <br>

**Коэффициент асимметрии**

one_skewness <br>
two_skewness <br>
three_skewness <br>
aVR_skewness <br>
aVL_skewness <br>
aVF_skewness <br>
V1_skewness <br>
V2_skewness <br>
V3_skewness <br>
V4_skewness <br>
V5_skewness <br>
V6_skewness <br>

**Коэффициент эксцесса/островершинности**
 
one_kurtosis <br>
two_kurtosis <br>
three_kurtosis <br>
aVR_kurtosis <br>
aVL_kurtosis <br>
aVF_kurtosis <br>
V1_kurtosis <br>
V2_kurtosis <br>
V3_kurtosis <br>
V4_kurtosis <br>
V5_kurtosis <br>
V6_kurtosis <br>


Всего в заключениях по результат электрокардиограмм содержится 67 заболеваний. <br>
При изучении метаданных было обнаружено 150 пропущенных значений возраста (колонка age). Пропуски были заполненными медианными значениями возраста отдельно для мужчин и отдельно для женщин.
