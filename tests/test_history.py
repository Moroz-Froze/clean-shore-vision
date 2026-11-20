from pathlib import Path
import sys,unittest,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_history import history_state,history_edit,history_move

class HistoryTests(unittest.TestCase):
    def test_undo_redo_branch_and_persistence(self):
        frame={'excluded_object_ids':[]}
        frame.update(history_edit(frame,[2]));frame['excluded_object_ids']=[2]
        frame.update(history_edit(frame,[2,3]));frame['excluded_object_ids']=[2,3]
        restored=json.loads(json.dumps(frame))
        state,ids=history_move(restored,-1);self.assertEqual(ids,[2]);restored.update(state);restored['excluded_object_ids']=ids
        state,ids=history_move(restored,1);self.assertEqual(ids,[2,3])
        restored.update(history_edit(restored,[2,4]));restored['excluded_object_ids']=[2,4]
        self.assertEqual(history_state(restored),([[ ],[2],[2,4]],2))
        self.assertIsNone(history_move(restored,1))

    def test_old_exclusions_can_be_restored(self):
        frame={'excluded_object_ids':[4,6]}
        self.assertEqual(history_move(frame,-1)[1],[])
        self.assertIsNone(history_move({'excluded_object_ids':[]},-1))

if __name__=='__main__':unittest.main()
