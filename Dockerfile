FROM mcr.microsoft.com/playwright/python:v1.51.0-noble

WORKDIR /app/OpenManus

RUN apt-get update && apt-get install -y --no-install-recommends git curl build-essential python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --break-system-packages -r requirements.txt

ENV OPENMANUS_API_HOST=0.0.0.0
ENV OPENMANUS_API_PORT=8000
EXPOSE 8000

CMD ["python", "run_api.py"]
