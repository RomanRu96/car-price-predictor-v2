# 1. Базовый образ (чистая система с Python)
FROM python:3.9-slim

# 2. Рабочая папка внутри контейнера
WORKDIR /app

# 3. Копируем только файл зависимостей (кеширование слоя)
COPY requirements.txt .

# 4. Устанавливаем библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем весь остальной код приложения
COPY . .

# 6. Открываем порт для Streamlit
EXPOSE 8501

# 7. Команда запуска при старте контейнера
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

