FROM python:3.9-slim

WORKDIR /app

# Copy the root-level requirements.txt
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api /app/api

COPY model_training/model_training/artifacts /app/model_artifacts
ENV DOCKER_ENV=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]



