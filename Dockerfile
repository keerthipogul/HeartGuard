
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 \
    -r requirements.txt

RUN pip install --no-cache-dir --default-timeout=300 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
