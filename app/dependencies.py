from iduconfig import Config

from app.logic.logger_setup import setup_logger

config = Config()
setup_logger(config, log_level="INFO")
