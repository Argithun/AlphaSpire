from copy import deepcopy

from utils.config_loader import ConfigLoader

DEFAULT_SIMULATION_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


def load_simulation_config():
    configuration = ConfigLoader.get("backtest", {})
    if not isinstance(configuration, dict):
        raise ValueError("config.backtest must be a mapping")
    configured_settings = configuration.get("settings", {})
    if not isinstance(configured_settings, dict):
        raise ValueError("config.backtest.settings must be a mapping")
    simulation_type = configuration.get("type", "REGULAR")
    if simulation_type != "REGULAR":
        raise ValueError("The alpha evaluator only supports backtest.type=REGULAR")
    settings = {**DEFAULT_SIMULATION_SETTINGS, **configured_settings}
    if settings["language"] != "FASTEXPR":
        raise ValueError("Generated alphas require backtest.settings.language=FASTEXPR")
    return {"type": simulation_type, "settings": deepcopy(settings)}


def build_simulation_payload(alpha_expression, simulation_config=None):
    configuration = load_simulation_config() if simulation_config is None else simulation_config
    return {**deepcopy(configuration), "regular": alpha_expression}
