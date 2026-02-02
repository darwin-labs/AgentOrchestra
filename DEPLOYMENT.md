# Deployment Guide

This guide describes how to build and run the AgentOrchestra container on your server.

## 1. Build the Container

You can build the image using the provided `docker-compose.yml`:

```bash
docker compose build
```

Or using Docker directly:

```bash
docker build -t agent-orchestra .
```

## 2. Configuration

The application is configured primarily through environment variables. You should create a `.env` file in the project root:

```env
# Required for LLM access (if config.toml says api_key = "env")
GROQ_API_KEY=your_groq_key_here

# Optional: Set an API key to protect your server
# If set, you must include 'Authorization: Bearer <key>' in your requests
OPENMANUS_API_KEY=your_secure_access_token

# Optional: Required if using Daytona for sandboxed execution
DAYTONA_API_KEY=your_daytona_key_here
```

## 3. Persistent Volumes

To ensure agent data and workspace files are not lost when the container restarts, the `docker-compose.yml` maps a local directory to the container:

- `./workspace`: Stores all generated files, logs, and agent state.

## 4. Running the Server

Start the container in detached mode:

```bash
docker compose up -d
```

Check the logs to ensure everything is running correctly:

```bash
docker compose logs -f
```

## 5. Testing the Deployment

Once running, the API will be available at `http://your-server-ip:8000/v1`. You can test it using the local `api_tester.py` by updating the Base URL in the UI.

### Note on Sandboxing

The current container runs commands directly in the `python:3.12-slim` environment. For enhanced security:

1. **Daytona**: RECOMMENDED. Provide a `DAYTONA_API_KEY` and the agent will use remote sandboxes.
2. **Docker Socket**: If you want to use local Docker sandboxes, you must mount the host's Docker socket in `docker-compose.yml`:
   ```yaml
   volumes:
     - /var/run/docker.sock:/var/run/docker.sock
   ```
   _Note: This has security implications for the host._
