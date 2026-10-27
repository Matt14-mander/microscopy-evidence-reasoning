def create_model(config, inference):
    backend = config["backend"]
    if backend == "mock":
        from src.models.mock import MockModel
        return MockModel()
    if backend == "api":
        from src.models.api import APIModel
        return APIModel(config, inference)
    if backend == "local":
        from src.models.local import LocalVLM
        return LocalVLM(config, inference)
    raise ValueError(f"Unknown model backend: {backend}")
