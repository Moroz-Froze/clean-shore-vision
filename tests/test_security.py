import io, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image,UnidentifiedImageError
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class SecurityTests(unittest.TestCase):
    def test_csv_all_untrusted_strings(self):
        from shore_core import csv_bytes
        import csv
        row = next(csv.DictReader(io.StringIO(csv_bytes([{'filename': ' \t=1+1', 'gps_status': '@SUM(1)', 'area_px': -1}]).decode('utf-8-sig'))))
        self.assertTrue(row['filename'].startswith("'"))
        self.assertTrue(row['gps_status'].startswith("'"))
        self.assertEqual(row['area_px'], '-1')

    def test_pixel_limit_before_decode(self):
        from shore_core import read_image
        data=io.BytesIO(); Image.new('RGB',(20,20)).save(data,format='PNG')
        with patch.dict(os.environ, {'MAX_IMAGE_PIXELS':'300'}):
            with self.assertRaises(ValueError):read_image(data.getvalue())

    def test_single_frame_and_real_format(self):
        from shore_core import read_image
        data=io.BytesIO(); Image.new('RGB',(10,10)).save(data,format='GIF')
        with self.assertRaises((ValueError,UnidentifiedImageError)):read_image(data.getvalue())
        data=io.BytesIO(); Image.new('RGB',(10,10)).save(data,format='TIFF',save_all=True,append_images=[Image.new('RGB',(10,10))])
        with self.assertRaises(ValueError):read_image(data.getvalue())

    def test_batch_limits(self):
        from security import validate_uploads
        class Upload:
            size=11
        with patch.dict(os.environ, {'MAX_UPLOAD_MB':'1','MAX_BATCH_MB':'1','MAX_BATCH_FILES':'1'}):
            with self.assertRaises(ValueError):validate_uploads([Upload(),Upload()])
            Upload.size=2*1024*1024
            with self.assertRaises(ValueError):validate_uploads([Upload()])

    def test_admission_rejects_busy_and_releases(self):
        from security import analysis_slot
        with analysis_slot():
            with self.assertRaises(ValueError):
                with analysis_slot():pass
        with analysis_slot():pass

    def test_storage_quota(self):
        from security import check_storage
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{'CLEAN_SHORE_DATA_DIR':temp,'MAX_STORED_MB':'1','MIN_FREE_MB':'1'}):
            root=Path(temp)/'results';root.mkdir();(root/'data').write_bytes(b'x'*(1024*1024))
            with self.assertRaises(ValueError):check_storage()

    def test_exception_message_does_not_leak(self):
        from security import public_error
        value=public_error(RuntimeError('/data/private/photo.jpg password=secret'))
        self.assertNotIn('/data',value);self.assertNotIn('secret',value)
