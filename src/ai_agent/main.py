import uvicorn

from src.ai_agent.config import config


def main() -> None:
    uvicorn.run(
        "src.ai_agent.app:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.debug,
    )


if __name__ == "__main__":
    main()
