"""Local preferences and the current analysis, excluding test sessions and package archives."""
from pathlib import Path
import os,json
from runtime_paths import data_dir,results_dir,server_mode
ROOT=Path(__file__).resolve().parent
def read_local(name):
    if os.environ.get('CLEAN_SHORE_TEST')=='1' or server_mode():return None
    try:return json.loads((data_dir()/name).read_text(encoding='utf-8'))
    except (OSError,ValueError):return None
def save_local(name,value):
    if os.environ.get('CLEAN_SHORE_TEST')=='1' or server_mode():return
    import tempfile
    path=data_dir()/name;path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False,suffix='.tmp') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);temp=Path(f.name)
    try:temp.replace(path)
    finally:temp.unlink(missing_ok=True)
def remember_job(job):
    if os.environ.get('CLEAN_SHORE_TEST')=='1' or server_mode():return
    folder=Path(job['folder']).resolve()
    if folder.parent!=results_dir():raise ValueError('Analysis folder outside application results')
    save_local('.current_run.json',{'folder':folder.name})
def restore_job():
    pointer=read_local('.current_run.json')
    if not pointer:return None
    folder=(results_dir()/pointer.get('folder','')).resolve()
    if folder.parent!=results_dir():return None
    try:
        manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
        settings={k:manifest['inference'][k] for k in ('tta','min_area','scale','project','allow_invalid_gps')}
        for k in ('high_thresh','low_thresh','min_seed_pixels','max_area','exclude_people_vehicles','color_sensitivity','min_core_ratio','min_core_pixels'):
            if k in manifest['inference']:settings[k]=manifest['inference'][k]
        return dict(folder=str(folder),frames=manifest['frames'],errors=manifest.get('errors',[]),settings=settings,exports_dirty=True)
    except (OSError,ValueError,KeyError):return None
