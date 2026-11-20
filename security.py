"""Resource boundaries for untrusted photos; no authentication claims."""
from contextlib import contextmanager
from pathlib import Path
import logging, os, shutil, threading, uuid
from runtime_paths import data_dir, results_dir

class InputRejected(ValueError):
    """Only application-authored messages may be displayed to a visitor."""

def limit(name, default):
    value=int(os.environ.get(name,default))
    if value<1:raise ValueError('Invalid positive resource limit: '+name)
    return value

def validate_uploads(uploads):
    if not uploads or len(uploads)>limit('MAX_BATCH_FILES',32):
        raise InputRejected('Недопустимое количество снимков; лимит задаёт администратор.')
    sizes=[u.size for u in uploads]
    if any(type(n) is not int or n<1 or n>limit('MAX_UPLOAD_MB',100)*1024**2 for n in sizes):
        raise InputRejected('Превышен размер файла или файл пуст.')
    if sum(sizes)>limit('MAX_BATCH_MB',256)*1024**2:
        raise InputRejected('Превышен общий размер выбранных файлов.')

_ANALYSIS_LOCK=threading.Lock()
@contextmanager
def analysis_slot():
    if not _ANALYSIS_LOCK.acquire(blocking=False):
        raise InputRejected('Сервер занят обработкой. Повторите после её завершения.')
    try:yield
    finally:_ANALYSIS_LOCK.release()

def check_storage():
    root=data_dir();root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<limit('MIN_FREE_MB',2048)*1024**2:
        raise InputRejected('Недостаточно свободного места. Обратитесь к администратору.')
    # Admission control, not a filesystem quota. Keep a separate hard volume quota.
    used=0
    if results_dir().exists():
        for path in results_dir().rglob('*'):
            if path.is_file() and not path.is_symlink():used+=path.stat().st_size
    if used>=limit('MAX_STORED_MB',10240)*1024**2:
        raise InputRejected('Хранилище результатов заполнено. Обратитесь к администратору.')

def public_error(exc):
    event=uuid.uuid4().hex[:12]
    logging.getLogger('clean_shore.security').warning('processing_error event=%s type=%s',event,type(exc).__name__)
    if isinstance(exc,InputRejected):return str(exc)
    return 'Не удалось обработать запрос. Код ошибки: '+event
