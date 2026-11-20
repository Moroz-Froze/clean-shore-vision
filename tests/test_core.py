import io,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shore_core import hysteresis,tta_probability,scale_for,object_location,extract_objects,csv_bytes,geojson_bytes,parse_xmp

class CoreTests(unittest.TestCase):
    def test_hysteresis_seed_growth_and_native_area(self):
        p=np.zeros((30,30),np.float32);p[2:10,2:10]=.5;p[4,4]=.7;p[15:25,15:25]=.5
        p[0,28:30]=.99
        m=hysteresis(p,.6,.45,45,1)
        self.assertEqual(int(m.sum()),64)
        self.assertEqual(int(hysteresis(p,.6,.45,65,1).sum()),0)

    def test_tta_undoes_every_flip_before_averaging(self):
        from PIL import Image
        rgb=np.random.default_rng(9).integers(0,256,(17,23,3),dtype=np.uint8)
        result=tta_probability(Image.fromarray(rgb),lambda im:np.asarray(im)[:,:,0]/255)
        np.testing.assert_allclose(result,rgb[:,:,0]/255,atol=1e-7)

    def test_unknown_scale_and_relative_altitude_not_assumed(self):
        meta=dict(relative_altitude=10,focal35=24,width=4000,height=3000,pitch=-90)
        self.assertIsNone(scale_for(meta,dict(mode='pixels'))[0])
        self.assertIsNone(scale_for(meta,dict(mode='altitude',confirm_flat=False))[0])
        gsd,_=scale_for(meta,dict(mode='altitude',confirm_flat=True))
        self.assertAlmostEqual(gsd,10*(36**2+24**2)**.5/(24*5000))
        meta['pitch']=-45
        self.assertIsNone(scale_for(meta,dict(mode='altitude',confirm_flat=True))[0])

    def test_points_are_frame_gps_unless_projection_explicit(self):
        meta=dict(latitude=55,longitude=37,gps_valid=True,pitch=-90,yaw=0,orientation=1,width=100,height=100)
        a=object_location(meta,60,50,.1,False,False)
        self.assertEqual(a,(55,37,'frame_gps'))
        lat,lon,method=object_location(meta,60,50,.1,True,False)
        self.assertEqual(lat,55);self.assertGreater(lon,37);self.assertEqual(method,'approx_nadir_projection')
        meta['gps_valid']=False
        self.assertEqual(object_location(meta,60,50,.1,True,False),(None,None,'gps_invalid'))

    def test_objects_area_and_unknown_geojson_are_honest(self):
        mask=np.zeros((20,20),np.uint8);mask[1:6,1:11]=1
        objects=extract_objects(mask,np.full(mask.shape,.8),dict(width=20,height=20),.02,False,False)
        self.assertEqual(len(objects),1);self.assertAlmostEqual(objects[0]['area_m2'],.02)
        objects[0]['filename']='=test.jpg'
        data=json.loads(geojson_bytes(objects));self.assertIsNone(data['features'][0]['geometry'])
        self.assertEqual(data['features'][0]['properties']['area_px'],50)
        self.assertIn("'=test.jpg",csv_bytes(objects).decode('utf-8-sig'))

    def test_dji_xmp_attributes_and_elements(self):
        data=b'<drone-dji:RelativeAltitude>+12.5</drone-dji:RelativeAltitude> drone-dji:GimbalPitchDegree="-90" drone-dji:GpsStatus="Invalid"'
        x=parse_xmp(data);self.assertEqual(x['RelativeAltitude'],'+12.5');self.assertEqual(x['GpsStatus'],'Invalid')

if __name__=='__main__':unittest.main()
