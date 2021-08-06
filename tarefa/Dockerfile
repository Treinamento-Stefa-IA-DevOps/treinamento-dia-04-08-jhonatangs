FROM python:3.8-slim

COPY ./codigo /app

WORKDIR /app

RUN pip install -r requirements

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
