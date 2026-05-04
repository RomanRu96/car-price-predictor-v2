

# 🚗 Car Price Predictor v2

Машинное обучение для предсказания стоимости автомобиля. Проект проходит полный цикл: от очистки данных и обучения до деплоя веб-приложения.

🔗 **[Попробовать демо](https://car-price-predictor-v2-2026.streamlit.app/)**  
📂 **[Датасет (Kaggle)](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)**

---

## 📊 О проекте

Цель проекта — создать точную модель для оценки стоимости авто на основе характеристик. 

**Ключевая особенность:** реализовано **сравнение двух алгоритмов**:
1. **PyTorch (MLP)** — нейронная сеть
2. **CatBoost** — градиентный бустинг

В продакшен (`app.py`) интегрирована победившая модель **CatBoost** (точность выше при меньшем коде).

### 🏆 Результаты

| Алгоритм | R² Score (Test) | 
|:---|:---:|
| **PyTorch MLP** | 90.07% |
| **CatBoost** | **92.85%** ✅ |

---

## 🛠 Технологии

Python 3.9+ | CatBoost | PyTorch | Pandas | Streamlit

---

## 📁 Структура

*   `train.py` — гибридный скрипт обучения (MLP + CatBoost)
*   `app.py` — веб-интерфейс на Streamlit
*   `model.py` — архитектура нейросети
*   `model_config.json` — метаданные модели

---

## 🚀 Запуск локально

**1. Клонирование**  
`git clone https://github.com/[RomanRu96]/car-price-predictor-v2.git`  
`cd car-price-predictor-v2`

**2. Установка зависимостей**  
`pip install -r requirements.txt`

**3. Запуск**  
`streamlit run app.py`

---

## 🔧 Обработка данных

✅ Извлечение марки из полного названия  
✅ Исправление опечаток (vokswagen → volkswagen, maxda → mazda)  
✅ Обработка пропусков в horsepower  
✅ One-Hot Encoding категорий

---

## 🤝 Контакты

Автор: Роман  
GitHub: [github.com/[RomanRu96]](https://github.com/[RomanRu96])

