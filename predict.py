# predict.py
import torch
import torch.nn as nn
import pandas as pd
import joblib

# === 1. Архитектура модели (должна совпадать с train.py) ===

class CarPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Dropout(0.1),             # случайно отключает 10% нейронов. Стд. защита от запоминания тренировочных данных 
            nn.Linear(12, 1)
        )
    def forward(self, x):
        return self.net(x)


# === 2. Загрузка артефактов ===

preprocessor = joblib.load("preprocessor.joblib")
price_scaler = joblib.load("price_scaler.joblib")


model = CarPredictor(31) 
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

print("\nАртефакты загружены успешно")


# === 3. Функция предсказания ===
def predict_car(horsepower: int, curbweight: int, fueltype: str, carbody: str, brand: str) -> float:
    # 1. Создаем DataFrame с той же схемой, что при обучении
    input_df = pd.DataFrame([{
        "horsepower": horsepower,
        "curbweight": curbweight,
        "fueltype": fueltype,
        "carbody": carbody, 
        "brand": brand 
    }])

    # 2. Пропускаем через препроцессор (сам масштабирует и кодирует)
    processed = preprocessor.transform(input_df)
    tensor_in = torch.FloatTensor(processed)

    # 3. Предсказание
    with torch.no_grad():
        pred_norm = model(tensor_in)

    # 4. Денормализация цены
    price = price_scaler.inverse_transform(pred_norm.numpy().reshape(-1, 1)).item()

    return price

'''
# тест
price = predict_car(105, 1500, 'gas', 'sedan' ,'mazda')
print(price)
'''
