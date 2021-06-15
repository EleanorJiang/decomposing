import logging, os

def add_logging(file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

os.chdir('/cluster/work/sachan/ct/data/decomposing')
logger = add_logging('log.txt')
# logger1 = add_logging('log_mi.txt')