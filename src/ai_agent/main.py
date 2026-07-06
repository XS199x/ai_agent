import uvicorn

from ai_agent.config import config


def main() -> None:
    uvicorn.run(
        "ai_agent.app:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.debug,
    )


if __name__ == "__main__":
    main()
