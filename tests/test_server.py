"""Server storage must survive relocation without exposing another browser's job."""
from pathlib import Path
import os,sys,tempfile,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class ServerStorageTests(unittest.TestCase):
    def test_configurable_paths_and_unique_jobs(self):
        from runtime_paths import model_dir,results_dir,new_job_dir
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'CLEAN_SHORE_DATA_DIR':temp,'CLEAN_SHORE_MODELS_DIR':temp+'/weights'}):
            self.assertEqual(model_dir(),(Path(temp)/'weights').resolve())
            a,b=new_job_dir(),new_job_dir()
            self.assertNotEqual(a,b)
            self.assertTrue(a.is_dir() and b.is_dir())
            self.assertEqual(a.parent,results_dir())

    def test_server_never_restores_shared_preferences_or_photos(self):
        import local_state
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'CLEAN_SHORE_DATA_DIR':temp,'CLEAN_SHORE_SERVER':'1','CLEAN_SHORE_TEST':'0'}),patch.object(local_state,'ROOT',Path(temp)):
            (Path(temp)/'.preferences.json').write_text('{"min_area":999}')
            (Path(temp)/'.current_run.json').write_text('{"folder":"someone_else"}')
            self.assertIsNone(local_state.read_local('.preferences.json'))
            self.assertIsNone(local_state.restore_job())
            local_state.save_local('.preferences.json',{'min_area':120})
            self.assertEqual(json.loads((Path(temp)/'.preferences.json').read_text())['min_area'],999)

    def test_desktop_restore_uses_configured_data_dir(self):
        import local_state
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'CLEAN_SHORE_DATA_DIR':temp,'CLEAN_SHORE_SERVER':'0','CLEAN_SHORE_TEST':'0'}),patch.object(local_state,'ROOT',Path(temp)):
            local_state.save_local('.preferences.json',{'min_area':120})
            self.assertEqual(local_state.read_local('.preferences.json'),{'min_area':120})
            folder=Path(temp)/'results'/'abc';folder.mkdir(parents=True)
            settings=dict(tta=False,min_area=120,scale={'mode':'pixels'},project=False,allow_invalid_gps=False)
            (folder/'run_manifest.json').write_text(json.dumps({'inference':settings,'frames':[]}))
            local_state.remember_job({'folder':str(folder)})
            self.assertEqual(Path(local_state.restore_job()['folder']),folder.resolve())

if __name__=='__main__':unittest.main()
