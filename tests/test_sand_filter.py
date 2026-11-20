import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference import filter_sand_candidates,rgb_to_lab

class SandFilterTests(unittest.TestCase):
    def scene(self,color):
        rgb=np.full((80,80,3),(185,158,116),np.uint8)
        mask=np.zeros((80,80),np.uint8);mask[30:50,30:50]=1
        rgb[mask==1]=color
        p=mask.astype(np.float32)*.6;p[32:42,32:42]=.9
        return rgb,mask,p

    def test_shadow_rejected_but_colored_and_white_kept(self):
        for color,keep in [((140,120,88),False),((20,90,220),True),((250,250,250),True)]:
            rgb,m,p=self.scene(color)
            lab=rgb_to_lab(rgb)
            if not keep:
                # A shadow changes L only; this explicitly tests equal sand chroma.
                lab[m==1,1:]=lab[0,0,1:]
            out=filter_sand_candidates(m,p,lab)
            self.assertEqual(bool(out.any()),keep)

    def test_core_requires_both_count_and_ratio(self):
        rgb,m,p=self.scene((20,90,220));p[:]=m*.6;p[30,30:50]=.9
        self.assertFalse(filter_sand_candidates(m,p,rgb_to_lab(rgb)).any())
        p[30:33,30:50]=.9 # exactly 15 percent
        self.assertTrue(filter_sand_candidates(m,p,rgb_to_lab(rgb)).any())

    def test_disable_color_does_not_disable_core(self):
        rgb,m,p=self.scene((185,158,116))
        self.assertTrue(filter_sand_candidates(m,p,None,color_sensitivity=0).any())
        p[:]=m*.6;p[30,30:44]=.9
        self.assertFalse(filter_sand_candidates(m,p,None,color_sensitivity=0,min_core_ratio=0).any())

    def test_no_ring_keeps_color_candidate_and_inputs_unchanged(self):
        m=np.ones((20,20),np.uint8);p=np.full(m.shape,.9,np.float32)
        rgb=np.full((*m.shape,3),120,np.uint8)
        out=filter_sand_candidates(m,p,rgb_to_lab(rgb))
        np.testing.assert_array_equal(out,m);self.assertTrue((p==.9).all())

    def test_other_candidates_excluded_from_ring(self):
        rgb,m,p=self.scene((185,158,116))
        # A blue neighbouring candidate must not make a sand-coloured patch appear coloured.
        m[30:50,52:60]=1;rgb[30:50,52:60]=(0,0,255);p[30:50,52:60]=.9
        out=filter_sand_candidates(m,p,rgb_to_lab(rgb))
        self.assertFalse(out[30:50,30:50].any());self.assertTrue(out[30:50,52:60].any())

    def test_lab_shape_validation(self):
        _,m,p=self.scene((0,0,255))
        with self.assertRaises(ValueError):filter_sand_candidates(m,p,np.zeros((2,2,3),np.uint8))

if __name__=='__main__':unittest.main()
