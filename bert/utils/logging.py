import logging
from transformers import logging as transformers_logging

def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    transformers_logging.set_verbosity_info()
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()