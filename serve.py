"""Foreground server entry point for Linux, Docker and process supervisors."""
from pathlib import Path
import argparse,hashlib,json,os,sys,tempfile
from runtime_paths import model_dir,data_dir,results_dir

def check_assets():
    folder=model_dir()
    expected={'model.pt':'28f88aaecb1771b61b98180b3b1aaa104ece522000f294fc66d0842856ffa899',
              'yolo11n.pt':'0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'}
    for name,checksum in expected.items():
        path=folder/name
        if not path.is_file():raise ValueError(f'Missing model asset: {path}')
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=checksum:raise ValueError(f'Model checksum mismatch: {name}')
    config=json.loads((folder/'config.json').read_text(encoding='utf8'))
    if len(config.get('id2label',{}))!=8:raise ValueError('V25 must have 8 model outputs')
    processor=json.loads((folder/'preprocessor_config.json').read_text(encoding='utf8'))
    if processor.get('do_resize') is not False:raise ValueError('Window resizing must be disabled')
    results_dir().mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryFile(dir=data_dir()):pass
    for key in ('HOME','XDG_CACHE_HOME','HF_HOME','YOLO_CONFIG_DIR'):
        if os.environ.get(key):Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    return {'models':str(folder),'results':str(results_dir()),'checksums':'verified'}

def smoke_test():
    import numpy as np
    from PIL import Image
    from inference import ShoreModel,ExclusionDetector
    model=ShoreModel(device='cpu')
    prob=model.predict_probability(Image.new('RGB',(96,80),(180,165,135)))
    assert prob.shape==(80,96) and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
    detector=ExclusionDetector(device='cpu')
    assert isinstance(detector.detect(Image.new('RGB',(96,80))),list)
    print(json.dumps({'cpu_inference':'passed','segformer_shape':list(prob.shape),'yolo_load_and_inference':'passed'}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('STREAMLIT_SERVER_COOKIE_SECRET','').strip():os.environ.pop('STREAMLIT_SERVER_COOKIE_SECRET',None)
    os.environ.setdefault('CLEAN_SHORE_SERVER','1')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    try:
        print(json.dumps(check_assets()),flush=True)
        if args.smoke_test:smoke_test();return
        if args.check_only:return
        port=int(os.environ.get('PORT','8501'));upload=int(os.environ.get('MAX_UPLOAD_MB','100'))
        if not 1<=port<=65535 or upload<1:raise ValueError('Invalid port or upload size')
    except (OSError,ValueError,AssertionError) as exc:
        parser.exit(1,f'Clean Shore startup failed: {exc}\n')
    os.chdir(Path(__file__).resolve().parent)
    os.execv(sys.executable,[sys.executable,'-m','streamlit','run','app.py',
        '--server.address','0.0.0.0','--server.port',str(port),'--server.headless','true',
        '--server.maxUploadSize',str(upload),'--server.fileWatcherType','none','--browser.gatherUsageStats','false'])

if __name__=='__main__':main()
