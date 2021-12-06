from sqlalchemy import create_engine
import yaml


def getSecrets(filepath):
    """
    Load secret database parameters.
    :param filepath: path to a .yaml file
    Returns dict.
    """
    with open(filepath, 'r') as f:
        secrets = yaml.safe_load(f)
    return secrets


def startEngine(secrets):
    """
    Start postgres database engine.
    :param secrets: dict
    Returns engine.
    """
    engine = create_engine('postgres://{user}:{password}@{host}:{port}/{dbname}'.format(
        host=secrets['host'],
        port=secrets['port'],
        dbname=secrets['dbname'],
        user=secrets['user'],
        password=secrets['password']
    ))
    return engine
