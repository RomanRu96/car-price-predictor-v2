# train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib  # Стандарт для сохранения sklearn-объектов
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import r2_score  # ← Добавлено для точного сравнения
from catboost import CatBoostRegressor  # ← Добавлено

# Импортируем модель из model.py                   
from model import CarPredictor


# === Глобальные гиперпараметры ===

HIDDEN_SIZE = 12
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001


# Фиксируем случайность для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)


# === 1. Загрузка данных в DataFrame ===
df = pd.read_csv("car_data.csv")

df['brand'] = df['CarName'].str.split().str[0].str.lower()  # сплитом поделили и взяли строку с индексом [0] взяв только марку без модели

# == 1.1. Исправление опечаток в брендах ===
# словарь.    опечатка: правильное значение
brand_fixes = {
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'porcshce' : 'porsche',
    'maxda': 'mazda',
    'toyouta': 'toyota'
}

# применение замены в колонке brand
df['brand'] = df['brand'].replace(brand_fixes)

# встретив НЕ число в horsepower меняет его на NaN 
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce') #  raise - выброс ошибки, coerce - превратить "aaa" в NaN(пропуск), ignore - оставить как есть 

# убирает полностью всю строку, если в ней есть NaN в horsepower 
df.dropna(subset=['horsepower'], inplace=True)  # True - меняет исходный DataFrame, возвращает None, False - возвращает новый DataFrame, исходный не меняется
# тут использовали True, чтобы не создавать копию. Важно для больших данных
# df = df.dropna(subset=['horsepower'], inplace=True) - так писаль нельзя. вернется None

cols_to_keep = ['brand', 'horsepower', 'curbweight', 'fueltype', 'carbody', 'price']
df = df[cols_to_keep]

df.reset_index(drop=True, inplace=True)
# drop=True - сбросить индексы и не создавать отдельной колонки со старыми индексами. Drop - англ. сброс
# drop=False - сбросить индексы и создать отдельную колонку со старыми индексами


# === 2. Разделение признаков и целевой переменной ===
numeric_cols = ["horsepower", "curbweight"]
categorical_cols = ["brand", "fueltype", "carbody"]
target_col = "price"    # важно!!! строка а не список ["price"] 

'''
for col in categorical_cols:
    unicue_vals = df[col].nunique()
    print(f"{col}: {unicue_vals} уникальных значений --> {df[col].unique()[:50]}")
'''

X = df[numeric_cols + categorical_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)








# === 2.1. Обучение CatBoost (сравнение с PyTorch) ===
# CatBoost работает с "сырыми" DataFrame. Ему не нужны StandardScaler и OneHotEncoder:
# он сам масштабирует числа и умно кодирует категории через Target Encoding.
print("\n🌲 Обучение CatBoost (на сырых данных)...")
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=False,          # отключаем лишний вывод в консоль
    random_state=42,
    cat_features=categorical_cols  # указываем, какие колонки - категории
)
# Обучаем на исходных X_train/y_train (без препроцессинга!)
cat_model.fit(X_train, y_train)

# Предсказания и оценка сразу на тесте
cat_preds = cat_model.predict(X_test)
cat_r2 = r2_score(y_test, cat_preds)
print(f"✅ CatBoost R² Test: {cat_r2:.4f}")








# === 3. Препроцессинг (ColumnTranformer + Pandas) ===

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)  
        # handle_unknown="ignore" это значит что все новые брэнды, которых нет в train будут игнорироваться и браться как [0,0,0,0,0,0] 
        # sparse_output=False - не убирает нули из кодировки, чтобы массив был более читаемым 
    ]

)

# Автоматически: масштабирует числа, кодирует строки, склеивает в матрицу
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print('Размерность признаков после препроцессинга:')
#print(X_train_proc.shape)  #  выводить кортеж (164, 31). 164 - количество тестовых примеров, 31 - количество тестовых признаков 
print(X_train_proc.shape[1])  # нам нужно количество признаков, поэтому указали [1] и вывод: 31 признак


# === 4. Нормализация цены (Target Scaling) ===
price_scaler = StandardScaler()
y_train_scaled = price_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = price_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


# === 5. Тензоры PyTorch ===
X_train_t = torch.FloatTensor(X_train_proc) 
y_train_t = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test_proc)
y_test_t = torch.FloatTensor(y_test_scaled).reshape(-1, 1)


# input_size посчитается автоматически (31 признак)
model = CarPredictor(
    input_size = X_train_proc.shape[1],
    hidden_size = HIDDEN_SIZE,
    dropout_rate = DROPOUT_RATE
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # LEARNING_RATE задается в самом начале
criterion = nn.MSELoss()

# === 6. Обучение ===
print("\nОбучение модели...")
for epoch in range(1000):  
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, loss: {loss.item()}")

print(f"\nФинальный Loss: {loss.item():.4f}")


# === 7. Оценка ===
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).numpy().flatten()
    test_pred = model(X_test_t).numpy().flatten()

def calc_r2(pred, true): 
    return 1 - (np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2))

mlp_r2 = calc_r2(test_pred, y_test_scaled)  # ← сохраняем для сравнения
print(f"\nR² Train: {calc_r2(train_pred, y_train_scaled):.2%}")
print(f"R² Test:  {calc_r2(test_pred, y_test_scaled):.2%}")

# Сравнение алгоритмов
print("\n📊 ИТОГОВОЕ СРАВНЕНИЕ:")
print(f"   PyTorch MLP R²: {mlp_r2:.4f}")
print(f"   CatBoost    R²: {cat_r2:.4f}")
best_model = "catboost" if cat_r2 >= mlp_r2 else "mlp"
print(f"🏆 Победитель: {best_model.upper()}")


# === 8. Сохранение артефактов ===
torch.save(model.state_dict(), "model.pth")           # Веса модели
joblib.dump(preprocessor, "preprocessor.joblib")      # числовые признаки
joblib.dump(price_scaler, "price_scaler.joblib")      # признаки price
cat_model.save_model("catboost_model.cbm")            # ← Сохраняем модель CatBoost
df.to_csv("cleaned_car_data.csv", index=False)        # очищенный датасет


# === 9. Сохранение конфига модели ===
model_config = {
    "input_size": int(X_train_proc.shape[1]),  # Автоматически 31
    "hidden_size": HIDDEN_SIZE,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target_col": target_col,

    "mlp_r2": float(mlp_r2),          # ← Добавлено
    "catboost_r2": float(cat_r2),     # ← Добавлено
    "best_model": best_model          # ← Добавлено
}

with open("model_config.json", "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)


print("\n Файлы успешно сохранены:")
print("\n\n model.pth \n preprocessor.joblib \n price_scaler.joblib \n catboost_model.cbm \n model_config.json \n cleaned_car_data.csv")

