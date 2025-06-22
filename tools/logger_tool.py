import logging
import os

def setup_logger(name: str, log_file: str = "logs/quiz_generation.log", level=logging.DEBUG):
    """
    Crée et configure un logger réutilisable pour tous les outils ou agents.

    Args:
        name: Nom du logger (ex: "quiz_generator")
        log_file: Chemin du fichier de log
        level: Niveau de logging

    Returns:
        logging.Logger: Instance du logger configuré
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
