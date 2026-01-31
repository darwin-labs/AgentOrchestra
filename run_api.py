import os

import uvicorn


def main() -> None:
    host = os.environ.get("OPENMANUS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENMANUS_API_PORT", "8000"))
    uvicorn.run("app.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
