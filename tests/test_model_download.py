from pathlib import Path
import hashlib,io,os,sys,tarfile,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import download_models

class ModelDownloadTests(unittest.TestCase):
    def test_existing_verified_weights_need_no_network(self):
        with tempfile.TemporaryDirectory() as temp:
            data=b'verified test model';Path(temp,'model.pt').write_bytes(data)
            with patch.dict(os.environ,{'CLEAN_SHORE_MODELS_DIR':temp}),patch.object(download_models,'HASHES',{'model.pt':hashlib.sha256(data).hexdigest()}),patch.object(sys,'argv',['download_models.py']),patch('download_models.urllib.request.urlopen',side_effect=AssertionError('unexpected network')):
                download_models.main()

    def test_corrupted_download_does_not_replace_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);destination=root/'models';destination.mkdir();(destination/'model.pt').write_bytes(b'original')
            archive=root/'bad.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                member=tarfile.TarInfo('clean-shore-linux/models/model.pt');member.size=3;tar.addfile(member,io.BytesIO(b'bad'))
            with patch.object(download_models,'HASHES',{'model.pt':hashlib.sha256(b'valid').hexdigest()}),self.assertRaises(ValueError):download_models.install(archive,destination)
            self.assertEqual((destination/'model.pt').read_bytes(),b'original')
            self.assertEqual(list(destination.glob('*.part')),[])

if __name__=='__main__':unittest.main()
