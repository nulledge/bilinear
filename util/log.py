import logging
import os
from datetime import datetime


def get_logger(time_stamp=None):
    if time_stamp is None:
        time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')

    log_dir = 'save/{time_stamp}'.format(time_stamp=time_stamp)
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

    return logger, log_dir, time_stamp
