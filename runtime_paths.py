"""Portable application paths; defaults preserve the Windows desktop layout."""
from pathlib import Path
import os,uuid
APP_ROOT=Path(__file__).resolve().parent

def model_dir():return Path(os.environ.get('CLEAN_SHORE_MODELS_DIR',APP_ROOT/'models')).expanduser().resolve()
def data_dir():return Path(os.environ.get('CLEAN_SHORE_DATA_DIR',APP_ROOT)).expanduser().resolve()
def results_dir():return data_dir()/'results'
def server_mode():return os.environ.get('CLEAN_SHORE_SERVER')=='1'
def new_job_dir():
    folder=results_dir()/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    return folder
