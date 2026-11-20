from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from photo_editor import selected_exclusions
class PhotoEditorTests(unittest.TestCase):
    def test_only_current_revision_valid_ids_are_accepted(self):
        frame={'review_revision':3,'objects':[{'object_id':2},{'object_id':8}],'excluded_object_ids':[1]}
        event={'revision':3,'ids':[8],'token':'a'}
        self.assertEqual(selected_exclusions(frame,event),[1,8])
        self.assertIsNone(selected_exclusions(frame,{**event,'revision':2}))
        self.assertIsNone(selected_exclusions(frame,{**event,'ids':[999]}))
        frame['last_photo_action']='a';self.assertIsNone(selected_exclusions(frame,event))
    def test_default_filter_is120(self):
        from settings_ui import DEFAULTS
        self.assertEqual(DEFAULTS['min_area'],120)
if __name__=='__main__':unittest.main()
