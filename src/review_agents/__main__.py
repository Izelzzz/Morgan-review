import logging

import uvicorn

from review_agents.api import create_app
from review_agents.config import Settings
from review_agents.credentials import TokenVault
from review_agents.pipeline.run import review_handler
from review_agents.worker import not_implemented


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    vault = TokenVault()
    # Without a finder model the service still starts, answers /healthz and queues
    # reviews; each review then fails with a message saying what to configure.
    handler = review_handler(settings, vault) if settings.finder_model else not_implemented
    uvicorn.run(create_app(settings, handler, vault), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
