FROM python:3.13-slim

WORKDIR /homework-1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data
COPY ./data/GasProperties.csv ./data/

CMD ["python", "evaluate_models.py"]
