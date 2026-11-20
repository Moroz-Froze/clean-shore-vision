"""Download only verified V25/YOLO weights from this project's GitHub release."""
from pathlib import Path
import argparse,hashlib,os,tarfile,tempfile,urllib.request
URL='https://github.com/Moroz-Froze/clean-shore-vision/releases/download/v1.0.0/clean-shore-v25-linux-cpu.tar.gz'
HASHES={'model.pt':'28f88aaecb1771b61b98180b3b1aaa104ece522000f294fc66d0842856ffa899',
        'yolo11n.pt':'0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'}

def install(archive,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for name,expected in HASHES.items():
            target=destination/name
            if target.exists():
                with target.open('rb') as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()==expected:continue
            member=tar.getmember('clean-shore-linux/models/'+name)
            if not member.isfile():raise ValueError('Unexpected model archive entry')
            temp=None
            try:
                digest=hashlib.sha256()
                with tar.extractfile(member) as source,tempfile.NamedTemporaryFile(dir=destination,delete=False,suffix='.part') as output:
                    temp=Path(output.name)
                    while block:=source.read(8*1024*1024):digest.update(block);output.write(block)
                if digest.hexdigest()!=expected:raise ValueError('Model checksum mismatch: '+name)
                temp.replace(target);target.chmod(0o644)
            finally:
                if temp:temp.unlink(missing_ok=True)
    print('V25 and YOLO11n weights verified and installed.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,help='Use an already downloaded release archive')
    args=parser.parse_args();destination=Path(__file__).resolve().parent/'models'
    if args.archive:install(args.archive,destination);return
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'release.tar.gz'
        print('Downloading model bundle (approximately 102 MiB)…',flush=True)
        with urllib.request.urlopen(URL,timeout=120) as source,archive.open('wb') as output:
            while block:=source.read(8*1024*1024):output.write(block)
        install(archive,destination)

if __name__=='__main__':main()
