# app.py (Адаптирован под CatBoost)
import streamlit as st
import pandas as pd
import json
from catboost import CatBoostRegressor  # ← Заменили torch на catboost

# кастомизация страницы
st.set_page_config(
    page_title="Car Price Predictor v2 (CatBoost)",
    page_icon="🚗",
    layout="centered"
)

# --- 1. Загрузка модели и конфига ---
@st.cache_resource
def load_resources():
    # Загружаем конфиг (метаданные, списки категорий)
    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # CatBoost загружается напрямую из .cbm файла
    # Ему НЕ нужны preprocessor или price_scaler!
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    return model, config

model, config = load_resources()

# --- 2. Интерфейс (UI) ---
st.title("🚗 Оценка стоимости авто")
# Показываем реальную метрику из конфига
r2_val = config.get("catboost_r2", 0.92)
st.caption(f"Прогноз цены на основе CatBoost (R² Test: {r2_val:.2%})")

st.markdown("### 📋 Параметры автомобиля")

col1, col2 = st.columns(2)

with col1:
    # Списки категорий (фиксированы, так как preprocessor убран)
    brand = st.selectbox("🏷️ Марка", sorted(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar',
        'mazda', 'buick', 'mercury', 'mitsubishi', 'nissan', 'peugeot', 'plymouth',
        'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']))
    
    horsepower = st.number_input(" Мощность (л.с.)", min_value=50, max_value=300, value=100, step=5)
    curbweight = st.number_input("⚖️ Вес (фунты)", min_value=1500, max_value=4500, value=2500, step=50)

with col2:
    fueltype = st.selectbox("⛽ Тип топлива", ["gas", "diesel"])
    carbody = st.selectbox("🚗 Тип кузова", ["convertible", "hardtop", "hatchback", "sedan", "wagon"])

# Отступ перед кнопкой
st.markdown("<br>", unsafe_allow_html=True)

# Кнопка по центру
_, btn_col, _ = st.columns([1, 3, 1])
with btn_col:
    if st.button("💰 Узнать цену", use_container_width=True, type="primary"):
        try:
            with st.spinner("🧮 Считаем прогноз..."):
                # 1. Собираем данные в DataFrame (CatBoost работает с сырыми данными!)
                input_df = pd.DataFrame([{
                    "horsepower": horsepower,
                    "curbweight": curbweight,
                    "fueltype": fueltype,
                    "carbody": carbody,
                    "brand": brand
                }])

                # 2. Предсказание (возвращает цену сразу в долларах, без денормализации)
                price = model.predict(input_df)[0]

            # 3. Вывод результата
            st.markdown(f"""
                <div style='background-color: #28a745; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;'>
                    <h2 style='color: white; margin: 0;'>💰 {price:,.0f} $</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>Прогнозируемая стоимость</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")

# Инфо о модели
st.markdown("---")
with st.expander("ℹ️ О модели"):
    st.write(f"""
        - **Алгоритм:** CatBoost Regressor
        - **R² Test:** {r2_val:.2%}
        - **Признаки:** {', '.join(config.get('numeric_cols', []) + config.get('categorical_cols', []))}
        - ⚠️ Прогноз на данных из Kaggle
    """)

# Футер
st.markdown("---")
st.caption("Проект создан в рамках обучения ML | [GitHub](https://github.com/RomanRu96)")

