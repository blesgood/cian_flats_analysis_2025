#Импортируем библиотеки
import pandas as pd
import matplotlib.pyplot as plt

#Импортируем данные из csv (с изменение разделителя на ";")
df = pd.read_csv('cian_flat_sale_1_57_moskva_19_Apr_2025_13_59_22_202458.csv', sep = ';')

#Выводим данные файла (первые 5 строк)
print(df.head(5))

#Выведем кол-во строк и колонок датасета
print("\nКоличество строк и колонок")
print(df.shape)

#Выведем названия всех колонок
print("\nНазвания колонок")
print(df.columns)

#Выведем тип данных каждой колоонки
print("\nТипы данных в каждой колонке")
print(df.dtypes)

#Выведем кол-во пропусков(пустых значений) в каждой колонке
print("\nКолличество пустых значений")
print(df.isnull().sum())

#Выведем основную статистику по числовым колонкам(цена, площадь, команды и т.д.)
print("\nОсновная статистика по числовым колонкам")
print(df.describe())

#Подготовка данных к анализу
#удаляем колонки которые не нужны для анализа
df = df.drop(['url', 'phone', 'object_type', 'heating_type'], axis=1)

#Удаляем объявление без цен
df = df.dropna(subset=['price'])

#Создадим новую колонку цены за квадратный метр
df['price_per_m2'] = df['price'] / df['total_meters']

#Средняя цена за квадртаный метр
average_price = df['price_per_m2'].mean()
print("\nСредняя цена за квадратный метр")
print(average_price)

#Максимальная цена за квадратный метр
max_price = df['price_per_m2'].max()
print("\nМаксимальная цена за квадратный метр")
print(max_price)

#Минимальная цена за квадратный метр
min_price = df['price_per_m2'].min()
print("\nМинимальная цена за квадратный метр")
print(min_price)

#Кол-во комнат в среднем у кваритр
average_rooms = df['rooms_count'].mean()
print("\nКолличество комнат в среднем у квартир")
print(average_rooms)

#Построим гистрограмму распределения цены за квадратный метр
plt.figure(figsize=(10, 5))
plt.hist(df['price_per_m2'], bins=30, edgecolor='black')
plt.title('Распределение цены за квадратный метр')
plt.xlabel('Цена за м² (руб)')
plt.ylabel('Колличество объявлений')
plt.grid(True, alpha=0.3)
plt.show()

#Удаляем выбросы по цене за квадратный метр
df = df[(df['price_per_m2']>200000)&(df['price_per_m2']<3000000)]

#Считаем медиану цены за квадратный метр
median_price = df['price_per_m2'].median()
#Выведим что получилось 
print("\nДанные без выбросов")
print(df.head(5))
print("\nМедиана за квадратный метр")
print(median_price)

#Сгруппируем по кол-ву комнат и посчитаем среднюю цену за м2
group_by_rooms = df.groupby('rooms_count')['price_per_m2'].mean()
#Выведем результат
print("\nГруппы по кол-ву комнат с подсчетом средней цены за квадратный метр")
print(group_by_rooms)

#Сгруппируем по округам и посчитаем срезднюю цену за м2 по районам и выведем топ-5 самых дорогих районов
group_by_district = df.groupby('district')['price_per_m2'].mean()
sorted = group_by_district.sort_values(ascending=False)
print("\nТоп 5 самых дорогих районов")
print(sorted.head(5))

#Очистим district от мусорных значений
df['district_len'] = df['district'].str.len()
df = df[df['district_len'] < 20]
#Удалим вспомогательную колонку district_len
df = df.drop(['district_len'], axis=1)
#Выведим топ-5 самых дорогих районов после чистки данных
group_by_district_len = df.groupby('district')['price_per_m2'].mean()
sorted_len = group_by_district_len.sort_values(ascending=False)
print("\nТоп 5 самых дорогих районов")
print(sorted_len.head(5))

#Построим столбчатую диаграмму средней цена по кол-ву комнат
rooms_price = df.groupby('rooms_count')['price_per_m2'].mean()

plt.figure(figsize=(10, 5))
plt.bar(rooms_price.index, rooms_price.values)
plt.title('Средняя цена по кол-ву комнат')
plt.xlabel('Колличество комнат')
plt.ylabel('Цена за м² (руб)')
plt.grid(True)
plt.tight_layout()
plt.show()

#Построим Горизонтальную столбчатую диаграмму топ-10 самых дорогих районов (после чистки)
top_districts = df.groupby('district')['price_per_m2'].mean().sort_values(ascending=False).head(10)


plt.figure(figsize=(10, 6))
plt.barh(range(len(top_districts)), top_districts.values)
plt.yticks(range(len(top_districts)), top_districts.index)
plt.xlabel('Цена за м² (руб)')
plt.title('Топ-10 самых дорогих районов Москвы')
plt.tight_layout()
plt.show()