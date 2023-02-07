import asyncio


class ModelOrchestrator:
    """
    The orechstrator will implement the business logic behind training the model and
    present the most updated version for inference.
    """
    def __init__(self):
        pass

    def fetch_latest_model(self):
        pass

    def _init_training(self):
        pass

    def train(self):
        pass

    def update_observation(self):
        pass

    def policy(self, observation):
        return 1