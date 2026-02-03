FROM mcr.microsoft.com/playwright/python:v1.51.0-noble

WORKDIR /app/OpenManus

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 1️⃣ copy deps only
COPY requirements.txt .

# 2️⃣ install deps (cached unless requirements.txt changes)
RUN pip install --break-system-packages -r requirements.txt

# 3️⃣ copy the rest of the app
COPY . .

ENV OPENMANUS_API_HOST=0.0.0.0
ENV OPENMANUS_API_PORT=8000
EXPOSE 8000

CMD ["python", "run_api.py"]
