FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1 STOCK_AGENT_MODELS=/app/models_storage

# Por defeito arranca a interface Streamlit; o docker-compose arranca também o Web Service.
EXPOSE 8501 8005
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
