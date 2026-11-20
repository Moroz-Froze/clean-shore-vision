from pathlib import Path
import sys,unittest
import numpy as np
from PIL import Image
import torch
import threading
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shore_core import hysteresis,blend_tensor

class RuntimeTests(unittest.TestCase):
    def test_cuda_oom_retries_smaller_batch(self):
        from inference import ShoreModel
        model=ShoreModel.__new__(ShoreModel);model.device='cuda';model.use_amp=True;model.batch_default=4;model.lock=threading.Lock()
        expected=np.zeros((2,2),np.float32)
        with patch('inference.blend_tensor',side_effect=[torch.cuda.OutOfMemoryError('test'),expected]) as blend,patch('torch.cuda.empty_cache'):
            result=model.predict_probability(Image.new('RGB',(2,2)))
        self.assertIs(result,expected);self.assertEqual(model.last_runtime['batch_size_effective'],2)
        self.assertEqual([c.args[3] for c in blend.call_args_list],[4,2])

    def test_seed_count_is_per_component_and_inclusive(self):
        p=np.zeros((40,50),np.float32)
        p[2:17,2:17]=.6;p[3,3:14]=.8  # 11 strong pixels
        p[22:37,30:45]=.6;p[23,31:43]=.8  # 12 strong pixels
        m=hysteresis(p,high=.75,low=.58,min_area=100,min_seed_pixels=12)
        self.assertEqual(m[:20].sum(),0);self.assertEqual(m[20:].sum(),225)
        self.assertEqual(hysteresis(p,.75,.58,226,12).sum(),0)
        with self.assertRaises(ValueError):hysteresis(p,min_seed_pixels=0)

    def test_batched_blend_alignment_edges_tta_and_tail(self):
        rgb=np.random.default_rng(2).integers(0,256,(800,1400,3),dtype=np.uint8)
        image=Image.fromarray(rgb);calls=[]
        def predict(images):
            calls.append(len(images))
            return torch.stack([torch.from_numpy(np.array(im)[:,:,0].copy()).float()/255 for im in images])
        for tta in (False,True):
            calls.clear();result=blend_tensor(image,predict,'cpu',batch_size=4,tta=tta)
            np.testing.assert_allclose(result,rgb[:,:,0]/255,atol=2e-7)
            self.assertIn(4,calls);self.assertIn(2,calls);self.assertLessEqual(max(calls),4)

if __name__=='__main__':unittest.main()
