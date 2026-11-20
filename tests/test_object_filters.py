from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shore_core import hysteresis,merge_exclusion_detections

class ObjectFiltersTests(unittest.TestCase):
    def test_tile_detections_deduplicate_only_same_class(self):
        boxes=[dict(xyxy=[0,0,50,50],confidence=.9,class_id=0),dict(xyxy=[1,1,51,51],confidence=.7,class_id=0),dict(xyxy=[0,0,50,50],confidence=.8,class_id=2)]
        kept=merge_exclusion_detections(boxes)
        self.assertEqual(len(kept),2);self.assertEqual({o['class_id'] for o in kept},{0,2})
    def test_native_area_range_is_inclusive(self):
        p=np.zeros((100,100),np.float32)
        p[1:11,1:13]=.9  # 120
        p[20:50,20:60]=.9  # 1200
        p[70:80,70:81]=.9  # 110
        m=hysteresis(p,.75,.51,120,12,max_area=1200)
        self.assertEqual(m.sum(),1320)
        self.assertEqual(hysteresis(p,.75,.51,120,12,max_area=1199).sum(),120)

    def test_detector_rectangles_clip_and_do_not_change_probability_cache(self):
        p=np.full((40,40),.9,np.float32);original=p.copy()
        # Exclude the left half; remaining 800 pixels are within range.
        m=hysteresis(p,.75,.51,120,12,max_area=1000,exclude_boxes=[[-10,-10,20,90]])
        self.assertEqual(m.sum(),800);self.assertEqual(m[:,:20].sum(),0)
        np.testing.assert_array_equal(p,original)
        self.assertEqual(hysteresis(p,.75,.51,120,12,max_area=1000).sum(),0)
        with self.assertRaises(ValueError):hysteresis(p,min_area=120,max_area=119)

if __name__=='__main__':unittest.main()
