#Импортируем библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Загружаем очищенные данные
df = pd.read_csv('data/cian_final_cleaned.csv')

#Выведем первые 5 строк и кол-во строк и колонок датасета
print(df.head(5))
print("\nКоличество строк и колонок")
print(df.shape)

#Корреляционный анализ Пирсона
#Обозначим за X1 - total_meters(независимая переменная), за Y - price(зависимая переменная)
x1 = df['total_meters']
y = df['price']
#Рассчитаем корреляцию Пирсона
correlation1 = x1.corr(y)
print(f"\nЗависимость цены от общей площади: {correlation1:.4f}")
print("Сильная положительная связь: чем больше площадь, тем выше цена")

#Обозначим за X2 - rooms_count(независимая переменная), за Y - price(зависимая переменная)
x2 = df['rooms_count']
y = df['price']
#Рассчитаем корреляцию Пирсона
correlation2 = x2.corr(y)
print(f"\nЗависимость цены от количества комнат: {correlation2:.4f}")
print("Слабая отрицательная связь: количество комнат почти не влияют на стоимость")

#Обозначим за X3 - floor(независимая переменная), за Y - price(зависимая переменная)
x3 = df['floor']
y = df['price']
#Рассчитаем корреляцию Пирсона
correlation3 = x3.corr(y)
print(f"\nЗависимость цены от этажа: {correlation3:.4f}")
print("Слабая отрицательная связь: чем выше этаж, тем дешевле цена")

#Обозначим за X4 - floors_count(независимая переменная), за Y - price(зависимая переменная)
x4 = df['floors_count']
y = df['price']
#Рассчитаем корреляцию Пирсона
correlation4 = x4.corr(y)
print(f"\nЗависимость цены от этажности дома: {correlation4:.4f}")
print("Умеренная отрицательная связь: чем выше дом, тем дешевле цена")

#Сравним типы этажей
#Классифицируем этажи: Первый, Средний, Последний
conditions = [
    df['floor'] == 1,
    df['floor'] == df['floors_count']
]
choice = ['Первый', 'Последний']

df['floor_type'] = np.select(conditions, choice, default='Средний')
print("\nТипы этажей:")
print(df[['floors_count', 'floor', 'floor_type']])

#Удалим выбросы, квартриры дороже 1.5млн рублей
df_filtered1 = df[df['price_per_m2'] < 1500000]
#Посчитаем среднюю цену за квадратный метр для каждого этажа
count_for_floortytype = df_filtered1.groupby('floor_type')['price_per_m2'].median().round(0)
#Выведем результат
print("\nСредняя цена за квадратный метр для каждого этажа")
print(count_for_floortytype)
#Определим какой тип этажа самый дорогой и самый дешёвый
print(f"\nСамый дешёвый этаж - {count_for_floortytype.idxmin()}: {count_for_floortytype.min():.0f}")
print(f"\nСамый дорогой этаж - {count_for_floortytype.idxmax()}: {count_for_floortytype.max():.0f}")
print("\nПервый этаж - самый доступный сегмент \nСредний этаж - основной объем предложений с базовой ценой \nПоследний этаж не имеет дисконта в данной выборке и даже дороже среднего, что может объясняться наличием улучшенных планировок или пентхаусов")

#Определим влияние высотности дома. Разобъем дома на категории: Малоэтажные(1-5), Среднеэтажные(6-10), Многоэтажные(11-20), Высотные(21+)
floor_category = [
    (df['floors_count'] >=1) & (df['floors_count'] <= 5),
    (df['floors_count'] >=6) & (df['floors_count'] <= 10),
    (df['floors_count'] >=11) & (df['floors_count'] <= 20),
    df['floors_count'] >= 21
]
category_name = ['Малоэтажный', 'Среднеэтажный', 'Многоэтажный', 'Высотный',]

df['floor_height'] = np.select(floor_category, category_name, default='Неизвестно')
print("\nЭтажности домов:")
print(df[['floors_count','floor_height']].head(15))

#Посчитаем среднюю цену за квадратный метр в каждой категории
count_of_floorheight = df.groupby('floor_height')['price_per_m2'].median().round(0)
print("\nСредняя цена за квадратный метр для каждый группы высотности дома")
print(count_of_floorheight)

#Удалим выбросы, квартриры дороже 1.5млн рублей
df_filtered2 = df[df['price_per_m2'] < 1500000]
#Дисконт за последний этаж (цена последних / цена средних - 1) * 100
prices = df_filtered2.groupby('floor_type')['price_per_m2'].median()
last = prices['Последний']
middle = prices['Средний']
discont = (last / middle-1)*100
print(f"\nЦена последних этажей отличается от средних на: {discont:.2f}%")

#Построим столбчатую диаграмму средней цены за квадратный метр для каждого этажа
graph1 = df_filtered1.groupby('floor_type')['price_per_m2'].median().round(0)

plt.figure(figsize=(10, 5))
plt.bar(graph1.index , graph1.values)
plt.title("Средняя цена за квадратный метр для каждого этажа")
plt.xlabel("Тип этажа")
plt.ylabel("Цена за квадратный метр")
plt.grid(True)
plt.show()

#Построим столбчатую диаграмму средний цены за квадратный метр в каждой категории этажности
graph2 = df.groupby('floor_height')['price_per_m2'].median().round(0)

plt.figure(figsize=(10, 5))
plt.bar(graph2.index, graph2.values)
plt.title("Средняя цена за квадратный метр в каждой категории этажности")
plt.xlabel("Этажность")
plt.ylabel("Цена за квадратный метр")
plt.grid(True)
plt.show()

#Построим матрицу корреляции для всех числовых признаков
numeric_df = df[['price', 'total_meters', 'rooms_count', 'floor', 'floors_count', 'price_per_m2']]
plt.figure(figsize=(10,5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Матрица корреляции факторов стоимости квартир')
plt.show()