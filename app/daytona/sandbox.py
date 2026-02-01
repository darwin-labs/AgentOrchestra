import time

from daytona import (
    CreateSandboxFromImageParams,
    Daytona,
    DaytonaConfig,
    Resources,
    Sandbox,
    SandboxState,
    SessionExecuteRequest,
)

from app.config import config
from app.utils.logger import logger


# load_dotenv()
def get_daytona_client() -> Daytona:
    """Get a Daytona client based on current configuration."""
    settings = config.daytona
    # logger.debug(f"Getting Daytona client with API key: {settings.daytona_api_key[:5] if settings.daytona_api_key else 'None'}...")
    conf = DaytonaConfig(
        api_key=settings.daytona_api_key,
        server_url=settings.daytona_server_url,
        target=settings.daytona_target,
    )
    return Daytona(conf)


async def get_or_start_sandbox(sandbox_id: str):
    """Retrieve a sandbox by ID, check its state, and start it if needed."""

    logger.info(f"Getting or starting sandbox with ID: {sandbox_id}")

    try:
        client = get_daytona_client()
        sandbox = client.get(sandbox_id)

        # Check if sandbox needs to be started
        if (
            sandbox.state == SandboxState.ARCHIVED
            or sandbox.state == SandboxState.STOPPED
        ):
            logger.info(f"Sandbox is in {sandbox.state} state. Starting...")
            try:
                client = get_daytona_client()
                client.start(sandbox)
                # Wait a moment for the sandbox to initialize
                # sleep(5)
                # Refresh sandbox state after starting
                sandbox = client.get(sandbox_id)

                # Start supervisord in a session when restarting
                start_supervisord_session(sandbox)
            except Exception as e:
                logger.error(f"Error starting sandbox: {e}")
                raise e

        logger.info(f"Sandbox {sandbox_id} is ready")
        return sandbox

    except Exception as e:
        logger.error(f"Error retrieving or starting sandbox: {str(e)}")
        raise e


def start_supervisord_session(sandbox: Sandbox):
    """Start supervisord in a session."""
    session_id = "supervisord-session"
    try:
        logger.info(f"Creating session {session_id} for supervisord")
        sandbox.process.create_session(session_id)

        # Execute supervisord command
        sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(
                command="exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf",
                var_async=True,
            ),
        )
        time.sleep(25)  # Wait a bit to ensure supervisord starts properly
        logger.info(f"Supervisord started in session {session_id}")
    except Exception as e:
        logger.error(f"Error starting supervisord session: {str(e)}")
        raise e


def create_sandbox(password: str, project_id: str = None):
    """Create a new sandbox with all required services configured and running."""

    logger.info("Creating new Daytona sandbox environment")
    logger.info("Configuring sandbox with browser-use image and environment variables")

    labels = None
    if project_id:
        logger.info(f"Using sandbox_id as label: {project_id}")
        labels = {"id": project_id}

    daytona_settings = config.daytona
    params = CreateSandboxFromImageParams(
        image=daytona_settings.sandbox_image_name,
        public=True,
        labels=labels,
        env_vars={
            "CHROME_PERSISTENT_SESSION": "true",
            "RESOLUTION": "1024x768x24",
            "RESOLUTION_WIDTH": "1024",
            "RESOLUTION_HEIGHT": "768",
            "VNC_PASSWORD": password,
            "ANONYMIZED_TELEMETRY": "false",
            "CHROME_PATH": "",
            "CHROME_USER_DATA": "",
            "CHROME_DEBUGGING_PORT": "9222",
            "CHROME_DEBUGGING_HOST": "localhost",
            "CHROME_CDP": "",
        },
        resources=Resources(
            cpu=2,
            memory=4,
            disk=5,
        ),
        auto_stop_interval=15,
        auto_archive_interval=24 * 60,
    )

    # Create the sandbox
    client = get_daytona_client()
    sandbox = client.create(params)
    logger.info(f"Sandbox created with ID: {sandbox.id}")

    # Start supervisord in a session for new sandbox
    start_supervisord_session(sandbox)

    logger.info(f"Sandbox environment successfully initialized")
    return sandbox


async def delete_sandbox(sandbox_id: str):
    """Delete a sandbox by its ID."""
    logger.info(f"Deleting sandbox with ID: {sandbox_id}")

    try:
        # Get the sandbox
        client = get_daytona_client()
        sandbox = client.get(sandbox_id)

        # Delete the sandbox
        client.delete(sandbox)

        logger.info(f"Successfully deleted sandbox {sandbox_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting sandbox {sandbox_id}: {str(e)}")
        raise e
