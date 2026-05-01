# app.py
import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
import json

# Импортируем модель из model.py                   
from model import CarPredictor

# кастомизация страницы
st.set_page_config(
    page_title="Car Price Predictor v2",
    page_icon="🚗",
    layout="centered"
)

# --- 1. Загрузка модели и артефактов ---
@st.cache_resource
def load_resources():
    # загружаем конфиг:
    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Загружаем веса и препроцессоры
    model = CarPredictor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        dropout_rate=config["dropout_rate"]
    )
    
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    
    preprocessor = joblib.load("preprocessor.joblib")
    price_scaler = joblib.load("price_scaler.joblib")
    
    return model, preprocessor, price_scaler, config

# Загружаем один раз при старте
model, preprocessor, price_scaler, config = load_resources()

# --- 2. Достаем список с данными из препрцессора ---

cat_encoder = preprocessor.named_transformers_['cat']
AVAILABLE_BRANDS = sorted(cat_encoder.categories_[0])
AVAILABLE_FUELTYPES = sorted(cat_encoder.categories_[1])
AVAILABLE_CARBODY = sorted(cat_encoder.categories_[2])


# --- 3. Интерфейс (UI) — Адаптивный для ПК и мобильных ---
st.title("🚗 Оценка стоимости авто")
st.caption("Прогноз цены на основе ML-модели (PyTorch)")

# Заголовок вынесен НАД колонками → сетка не ломается
st.markdown("###  Параметры автомобиля")

col1, col2 = st.columns(2)

with col1:
    
    brand = st.selectbox("🏷️ Марка", AVAILABLE_BRANDS)
    
    horsepower = st.number_input("⚡ Мощность (л.с.)", min_value=50, max_value=500, value=150, step=10)
    curbweight = st.number_input(" Вес (фунты)", min_value=1500, max_value=4500, value=1500, step=100)

with col2:
    fueltype = st.selectbox(" Вид топлива", AVAILABLE_FUELTYPES)  
    carbody = st.selectbox(" Тип кузова", AVAILABLE_CARBODY)

# Отступ перед кнопкой
st.markdown("<br>", unsafe_allow_html=True)

# Кнопка по центру
_, btn_col, _ = st.columns([1, 3, 1])
with btn_col:
    if st.button(" Узнать цену", use_container_width=True, type="primary"):
        try:
            with st.spinner(" Считаем прогноз..."):
                input_df = pd.DataFrame([{
                    "horsepower": horsepower,
                    "curbweight": curbweight,
                    "fueltype": fueltype,
                    "carbody": carbody, 
                    "brand": brand 
                }])
                processed = preprocessor.transform(input_df)
                tensor_in = torch.FloatTensor(processed)
                with torch.no_grad():
                    pred_norm = model(tensor_in)
                price = price_scaler.inverse_transform(pred_norm.numpy().reshape(-1, 1)).item()

            # Вывод результата
            st.markdown(f"""
                <div style='background-color: #28a745; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;'>
                    <h2 style='color: white; margin: 0;'> {price:,.0f} $</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>Прогнозируемая стоимость</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")

st.markdown("---")

with st.expander("ℹ️ О модели"):
    st.write("""
        - **Архитектура:** MLP, 1 скрытый слой (16 нейронов)
        - **Признаки:** пробег, мощность, год, марка 
        - **R² Test:** ~98.7%
        - ⚠️ Прогноз на синтетических данных
    """)

st.markdown("---")
st.caption("Проект создан в рамках обучения ML | [GitHub](https://github.com/RomanRu96)")

