import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


# Загрузка данных из файла boston.csv
data = pd.read_csv("boston.csv")

print(data.info(), '\n\n\n') # общая информация о DataFrame



# Проверим типы данных каждого столбца
print(data.dtypes, '\n')


# Проверим, есть ли нечисловые данные
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns # метод .select_dtypes() используется для выбора столбцов в DataFrame на основе их типа данных, метод .columns возвращает имена этих столбцов

print(non_numeric_columns, '\n') # список пустой

if len(non_numeric_columns) == 0:
    print("\nВсе столбцы имеют числовой тип.")
else:
    print(f"\nНайдены нечисловые столбцы: {list(non_numeric_columns)}")

print('\n')

# Проверим, есть ли пропущенные данные в данных
missing_data = data.isnull().sum() # с помощью sum суммируем значения по каждому столбцу (True - 1, False - 0)
print("Пропущенные данные:\n", missing_data)

# Рассчитываем корреляцию для всех числовых столбцов
correlation_matrix = data.corr() # этот метод автоматически рассчитывает корреляцию Пирсона между всеми числовыми столбцами в DataFrame.

# Выводим корреляционную матрицу
print("\nКоэффициент корреляции для всех пар признаков:")
print(correlation_matrix)

# Строим тепловую карту
plt.figure(figsize=(10, 8))  # Устанавливаем размер изображения
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=3) # heatmap - построение тепловой карты
# annot = True - в каждой ячейке тепловой карты будут отображаться числовые значения коэффициентов корреляции
# .2f - числа будут отображаться с двумя знаками после запятой, linewidths - толщина линии, разделяющей ячейки карты

# Показываем график
plt.title("Тепловая карта корреляции признаков\n")
plt.show()

# Целевой признак
target = "MEDV"

# Рассчитываем корреляцию всех признаков с целевым признаком
correlations_with_target = data.corr()[target].abs().sort_values(ascending=False) # Сортируем коэффициенты корреляции по убыванию (самая сильная положительная корреляция будет первой)

# Отображаем отсортированные коэффициенты корреляции
print("\nКорреляция с целевым признаком (MEDV):\n")
print(correlations_with_target, '\n')

# Выбираем от 4 до 6 признаков с наибольшей корреляцией (по модулю, исключая целевой признак)
top_features = correlations_with_target.iloc[1:7]  # Пропускаем первый элемент (это сам MEDV)

# Отображаем выбранные признаки
print("Наиболее коррелирующие признаки:\n")
print(top_features)

# Сохраняем эти признаки для дальнейшего анализа
selected_features = top_features.index.tolist() # преобразуем в обычный список
print("\nВыбранные признаки для дальнейшего анализа:", selected_features)

# Для каждого выбранного признака строим точечную диаграмму
for feature in selected_features:
    plt.figure(figsize=(8, 6))  # Устанавливаем размер графика
    # Строим точечную диаграмму для целевого признака и выбранного признака
    sns.scatterplot(x=data[feature], y=data[target], color='blue')

    # Заголовок графика
    plt.title(f'Точечная диаграмма: {feature} vs {target}')

    # Подписи осей
    plt.xlabel(feature)
    plt.ylabel(target)

    # Показываем график
    plt.show()


# 'LSTAT', 'RM', 'PTRATIO'

# Список факторных признаков
factor_features = ["LSTAT", "RM", "PTRATIO"]

# Целевая переменная
target = "MEDV"

print("\nФакторные признаки:", factor_features)
print("Целевая переменная:", target)

# Признаки
X = data[factor_features]

# Целевая переменная
y = data[target]

# Разбиение на обучающую и тестовую выборки в соотношении 8:2 (80% для обучения, 20% для тестирования)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # ф-я разделяет исходные данные на обучающую и тестовую выборки
# X - признаки или входные данные, y - это целевая переменная (выходные данные), test_size=0.2 означает, что 20% всех данных будет использоваться для тестовой выборки
# random_state=42 гарантирует, что каждый раз функция возвращает одно и то же разбиение данных независимо от того, сколько раз вы запускаете программу, генератор случайных чисел


# Выводим размер обучающей и тестовой выборки
print(f"\nРазмер обучающей выборки X_train: {X_train.shape}")
print(f"Размер тестовой выборки X_test: {X_test.shape}")
print(f"Размер обучающей выборки y_train: {y_train.shape}")
print(f"Размер тестовой выборки y_test: {y_test.shape}")



# Создаём объект модели линейной регрессии
model = LinearRegression()

# Обучаем модель на обучающих данных
model.fit(X_train, y_train) # X_train - обучающая выборка, y_train - целевая переменная для обучающей выборки

# Передаем обученной модели данные X_train и получаем прогнозные значения для обучающей выборки
y_train_pred = model.predict(X_train)

# Передаем обученной модели данные X_test и получаем прогнозные значения для тестовой выборки
y_test_pred = model.predict(X_test)


print("\nПрогнозные значения на обучающей выборке:\n", y_train_pred[:5])
print("Прогнозные значения на тестовой выборке:\n", y_test_pred[:5])



# Рассчитываем коэффициент детерминации (R²) для обучающей выборки
r2_train = r2_score(y_train, y_train_pred)

# Рассчитываем коэффициент детерминации (R²) для тестовой выборки
r2_test = r2_score(y_test, y_test_pred)

# Рассчитываем среднеквадратичную ошибку (MSE) для обучающей выборки
mse_train = mean_squared_error(y_train, y_train_pred)

# Рассчитываем среднеквадратичную ошибку (MSE) для тестовой выборки
mse_test = mean_squared_error(y_test, y_test_pred)

# Рассчитываем корень из среднеквадратичной ошибки (RMSE) для обучающей выборки
rmse_train = np.sqrt(mse_train)

# Рассчитываем корень из среднеквадратичной ошибки (RMSE) для тестовой выборки
rmse_test = np.sqrt(mse_test)

# Выводим результаты
print(f"\nКоэффициент детерминации (R²) на обучающей выборке: {r2_train:.4f}")
print(f"Коэффициент детерминации (R²) на тестовой выборке: {r2_test:.4f}")
print(f"Корень из среднеквадратичной ошибки (RMSE) на обучающей выборке: {rmse_train:.4f}")
print(f"Корень из среднеквадратичной ошибки (RMSE) на тестовой выборке: {rmse_test:.4f}")

# 14 задание

# Строим boxplot для целевого признака (MEDV)
plt.figure(figsize=(8, 6))
sns.boxplot(x=data[target])

# Заголовок
plt.title('Boxplot для целевого признака MEDV')

# Показываем график
plt.show()

# Для вычисления выбросов по методу IQR
Q1 = data[target].quantile(0.25)  # Первый квартиль
Q3 = data[target].quantile(0.75)  # Третий квартиль
IQR = Q3 - Q1  # Межквартильное расстояние

# Вычисляем границы для выбросов
lower_bound = Q1 - 1.5 * IQR  # Нижняя граница
upper_bound = Q3 + 1.5 * IQR  # Верхняя граница

print(f"Нижняя граница: {lower_bound}")
print(f"Верхняя граница: {upper_bound}")

# Определяем выбросы
outliers = data[target][(data[target] < lower_bound) | (data[target] > upper_bound)] # отбираем те строки, для которых условие выброса (меньше или больше границы) выполняется

# Выводим выбросы
print("\nВыбросы в целевом признаке (MEDV):")
print(outliers)

# Если по графику не удалось точно определить выбросы, считаем, что 50.0 является выбросом
# outliers_manual = data[target][data[target] == 50.0]
# print("\nРучные выбросы (MEDV = 50.0):")
# print(outliers_manual)

# 15 задание

# Отфильтровываем выбросы
filtered_data = data[(data[target] >= lower_bound) & (data[target] <= upper_bound)]

# Признаки (факторные признаки)
X_filtered = filtered_data[factor_features]

# Целевая переменная (без выбросов)
y_filtered = filtered_data[target]

# Разбиение на обучающую и тестовую выборки (80%/20%)
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Обучаем модель на очищенных данных
model_filtered = LinearRegression()
model_filtered.fit(X_train_filtered, y_train_filtered)

# Прогнозируем значения на обучающей и тестовой выборках
y_train_pred_filtered = model_filtered.predict(X_train_filtered)
y_test_pred_filtered = model_filtered.predict(X_test_filtered)

# Рассчитываем R2 и RMSE для обучающей выборки
r2_train_filtered = r2_score(y_train_filtered, y_train_pred_filtered)
rmse_train_filtered = np.sqrt(mean_squared_error(y_train_filtered, y_train_pred_filtered))

# Рассчитываем R2 и RMSE для тестовой выборки
r2_test_filtered = r2_score(y_test_filtered, y_test_pred_filtered)
rmse_test_filtered = np.sqrt(mean_squared_error(y_test_filtered, y_test_pred_filtered))


# 8. Выводим результаты до и после удаления выбросов для сравнения
print("Результаты до удаления выбросов:")
print(f"R2 (train): {r2_train:.4f}, RMSE (train): {rmse_train:.4f}")
print(f"R2 (test): {r2_test:.4f}, RMSE (test): {rmse_test:.4f}")
print("\nРезультаты после удаления выбросов:")
print(f"R2 (train): {r2_train_filtered:.4f}, RMSE (train): {rmse_train_filtered:.4f}")
print(f"R2 (test): {r2_test_filtered:.4f}, RMSE (test): {rmse_test_filtered:.4f}")

print("\nПосле удаления выбросов модель хуже объясняет изменения в целевом признаке (R² снизился с 0.56 до 0.43 на тесте).")
print("Это значит, что модель стала менее способной учитывать все факторы, влияющие на целевую переменную.")
print("Однако ошибка предсказания (RMSE) снизилась, что говорит о том, что модель стала точнее, так как выбросы больше не искажают результаты.")

# 16 задание

# Инициализируем модель Ridge
ridge_model = Ridge()

# Обучаем модель на обучающих данных
ridge_model.fit(X_train, y_train)

# Прогнозируем на обучающих данных
y_train_pred_ridge = ridge_model.predict(X_train)

# Прогнозируем на тестовых данных
y_test_pred_ridge = ridge_model.predict(X_test)

# Рассчитываем коэффициент детерминации (R²) для обучающей выборки
r2_train_ridge = r2_score(y_train, y_train_pred_ridge)

# Рассчитываем коэффициент детерминации (R²) для тестовой выборки
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)

# Рассчитываем среднеквадратичную ошибку (MSE) для обучающей выборки
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)

# Рассчитываем среднеквадратичную ошибку (MSE) для тестовой выборки
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)

# Рассчитываем корень из среднеквадратичной ошибки (RMSE) для обучающей выборки
rmse_train_ridge = np.sqrt(mse_train_ridge)

# Рассчитываем корень из среднеквадратичной ошибки (RMSE) для тестовой выборки
rmse_test_ridge = np.sqrt(mse_test_ridge)

print("Результаты для гребневой регрессии (Ridge):")
print(f"R² (train): {r2_train_ridge:.4f}, RMSE (train): {rmse_train_ridge:.4f}")
print(f"R² (test): {r2_test_ridge:.4f}, RMSE (test): {rmse_test_ridge:.4f}")

# 17 задание

# Создаем полиномиальные признаки 3-й степени (  для каждого признака будут добавлены новые признаки (квадратичные) и  (кубические)
poly = PolynomialFeatures(degree=3)

# Преобразуем данные
X_poly = poly.fit_transform(X_train)  # Преобразуем обучающие данные (метод fit_transform сначала обучает модель
# на обучающих данных, затем сразу выполняет преобразование)
X_test_poly = poly.transform(X_test)  # Преобразуем тестовые данные

# Создаем и обучаем модель линейной регрессии на полиномиальных признаках
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)

# Прогнозируем значения на обучающей выборке
y_train_pred_poly = poly_reg.predict(X_poly)

# Прогнозируем значения на тестовой выборке
y_test_pred_poly = poly_reg.predict(X_test_poly)

# Рассчитываем R2 и RMSE для обучающей выборки
r2_train_poly = r2_score(y_train, y_train_pred_poly)
rmse_train_poly = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))

# Рассчитываем R2 и RMSE для тестовой выборки
r2_test_poly = r2_score(y_test, y_test_pred_poly)
rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_test_pred_poly))

# Выводим результаты
print("\nПолиномиальная регрессия (степень 3):")
print(f"R2 (train): {r2_train_poly:.4f}, RMSE (train): {rmse_train_poly:.4f}")
print(f"R2 (test): {r2_test_poly:.4f}, RMSE (test): {rmse_test_poly:.4f}")


print("\nРезультаты линейной регрессии:")
print(f"R2 (train): {r2_train:.4f}, RMSE (train): {rmse_train:.4f}")
print(f"R2 (test): {r2_test:.4f}, RMSE (test): {rmse_test:.4f}")

print("\nРезультаты гребневой регрессии (Ridge):")
print(f"R2 (train): {r2_train_ridge:.4f}, RMSE (train): {rmse_train_ridge:.4f}")
print(f"R2 (test): {r2_test_ridge:.4f}, RMSE (test): {rmse_test_ridge:.4f}")

print("\nРезультаты полиномиальной регрессии (степень 3):")
print(f"R2 (train): {r2_train_poly:.4f}, RMSE (train): {rmse_train_poly:.4f}")
print(f"R2 (test): {r2_test_poly:.4f}, RMSE (test): {rmse_test_poly:.4f}")

print("\nВывод:")
print("R² показывает, насколько хорошо модель объясняет изменения в данных. Чем выше R², тем лучше модель.")
print("RMSE показывает, насколько точно модель предсказывает значения. Чем ниже RMSE, тем точнее модель.")

print("\nСравнение моделей:")
print("1. Линейная регрессия: R² на тестовых данных — 0.4765, что означает, что модель объясняет только 47% данных. "
      "RMSE — 6.1958, что указывает на большую ошибку предсказаний.")
print("2. Гребневая регрессия (Ridge): результаты почти такие же, как у линейной регрессии. Это значит, что регуляризация (метод, используемый в машинном обучении для предотвращения переобучения модели) не улучшила модель.")
print("3. Полиномиальная регрессия (степень 3): она объясняет 66% данных (R² = 0.6656), и ошибка меньше (RMSE = 4.9517), что делает модель точнее.")

print("\nЗаключение: полиномиальная регрессия дает лучшие результаты по точности, но она может переобучиться. "
      "Линейная и гребневая регрессии менее точны, но более стабильны.")