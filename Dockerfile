FROM python:3.12-slim

WORKDIR /app/OpenManus

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/* \
    && (command -v uv >/dev/null 2>&1 || pip install --no-cache-dir uv)

COPY . .

RUN uv pip install --system -r requirements.txt

ENV OPENMANUS_API_HOST=0.0.0.0
ENV OPENMANUS_API_PORT=8000
EXPOSE 8000

CMD ["python", "run_api.py"]
