import logging
import os
import sys


def custom_logger(name, dir, save=True, level='debug') -> logging.Logger:
    """
    make logger
    :param level: level of logging
    :param name: name (scope) of this logger
    :param dir: directory for saving the log file
    :param save: if true, dump log to file
    :return: logger object
    """

    LEVELS = {'CRITICAL': logging.CRITICAL, 'ERROR': logging.ERROR,
              'WARNING': logging.WARNING, 'INFO': logging.INFO,
              'DEBUG': logging.DEBUG}

    logging.captureWarnings(True)
    logger = logging.getLogger(name)
    logger.setLevel(LEVELS[level.upper()])

    if save:
        if not os.path.exists(dir):
            os.makedirs(dir)
        log_dir = os.path.join(dir, 'experiment.log')
        f_handler = logging.FileHandler(log_dir)
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    return logger


def shutdown_logger(logger):
    """
    Close logger.
    :param logger:
    :return:
    """
    # TODO: debug this, not working
    for handler in logger.handlers:
        handler.close()
