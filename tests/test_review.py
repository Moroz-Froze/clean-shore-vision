from pathlib import Path
import sys,unittest,tempfile,json,zipfile
import numpy as np
from PIL import Image
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from processing import apply_review,write_exports

class ReviewTests(unittest.TestCase):
    def test_reject_restore_masks_totals_exports_and_original_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);image=Image.new('RGB',(100,80),(120,130,140));image.save(folder/'source.png')
            mask=np.zeros((80,100),np.uint8);mask[10:20,10:20]=1;mask[40:50,60:80]=1
            Image.fromarray(mask).save(folder/'raw.png');original=(folder/'raw.png').read_bytes()
            objects=[dict(object_id=1,area_px=100,area_m2=1.,bbox_x=10,bbox_y=10,bbox_width=10,bbox_height=10,filename='a.jpg',latitude=55,longitude=37),
                dict(object_id=2,area_px=200,area_m2=2.,bbox_x=60,bbox_y=40,bbox_width=20,bbox_height=10,filename='a.jpg',latitude=55,longitude=37)]
            frame=dict(frame_id='frame',filename='a.jpg',source_file='source.png',model_mask='raw.png',mask='review.png',overlay='boxes.jpg',
                all_objects=objects,objects=objects.copy(),gsd_m_per_px=.1,metadata={'width':100,'height':80},review_revision=0)
            apply_review(frame,folder,[2])
            self.assertEqual(frame['object_count'],1);self.assertEqual(frame['area_px'],100);self.assertAlmostEqual(frame['area_m2'],1.)
            self.assertEqual(int(np.array(Image.open(folder/'review.png')).sum()),100)
            self.assertEqual(frame['excluded_object_ids'],[2]);self.assertEqual((folder/'raw.png').read_bytes(),original)
            manifest=write_exports(folder,[frame],[],{},build_zip=True)
            geo=json.loads((folder/'detections.geojson').read_bytes());self.assertEqual(len(geo['features']),1)
            self.assertEqual(geo['features'][0]['properties']['object_id'],1)
            self.assertEqual(manifest['frames'][0]['excluded_object_ids'],[2])
            with zipfile.ZipFile(folder/'processed_photos.zip') as z:
                self.assertIsNone(z.testzip());self.assertIn('review_changes.json',z.namelist())
                self.assertNotIn('raw.png',z.namelist());self.assertNotIn('source.png',z.namelist())
            apply_review(frame,folder,[])
            self.assertEqual(frame['object_count'],2);self.assertAlmostEqual(frame['area_m2'],3.)
            np.testing.assert_array_equal(np.array(Image.open(folder/'review.png')),mask)
            apply_review(frame,folder,[1,2])
            self.assertEqual(frame['object_count'],0);self.assertEqual(frame['area_px'],0)

    def test_reject_unknown_object_is_not_a_silent_edit(self):
        with self.assertRaises(ValueError):apply_review({'all_objects':[{'object_id':1}]},Path('.'),[999])

if __name__=='__main__':unittest.main()
