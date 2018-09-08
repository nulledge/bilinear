import logging
import os
from datetime import datetime


def get_logger(comment=None):
    if comment is None:
        comment = datetime.now().strftime('%b%d_%H-%M-%S')

    log_dir = 'save/{comment}'.format(comment=comment)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'debug.log')

    form = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    file = logging.FileHandler(log_file)
    stream = logging.StreamHandler()

    file.setFormatter(form)
    stream.setFormatter(form)

    logger = logging.getLogger()
    logger.setLevel(10)  # CRIT=50, ERR=40, WARN=30, INFO=20, DEBUG=10
    logger.addHandler(file)
    logger.addHandler(stream)

    return logger, log_dir, comment
