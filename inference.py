"""Self-contained offline V25 loader. Never fetch weights or change classifier shape."""
from pathlib import Path
import hashlib,json,threading
from contextlib import nullcontext
import numpy as np
import cv2
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import SegformerConfig,SegformerForSemanticSegmentation,SegformerImageProcessor
from shore_core import blend_tensor,hysteresis,HIGH,LOW,MIN_AREA,MAX_AREA,MIN_SEED_PIXELS
from runtime_paths import model_dir,data_dir
ROOT=Path(__file__).resolve().parent
EXPECTED='28f88aaecb1771b61b98180b3b1aaa104ece522000f294fc66d0842856ffa899'

COLOR_SENSITIVITY=6.0
MIN_CORE_RATIO=.15
MIN_CORE_PIXELS=15

def rgb_to_lab(image):
    """Cache compact OpenCV uint8 LAB, at original resolution, not a thumbnail."""
    return cv2.cvtColor(np.asarray(image,dtype=np.uint8),cv2.COLOR_RGB2LAB)

def filter_sand_candidates(mask,probability,lab=None,high_thresh=HIGH,
        color_sensitivity=COLOR_SENSITIVITY,min_core_ratio=MIN_CORE_RATIO,min_core_pixels=MIN_CORE_PIXELS):
    """Heuristic only: retain dense probability cores and chromatically distinct objects.

    LAB threshold is Euclidean delta_ab in CIELAB units. L is converted to 0..100.
    Zero sensitivity disables colour rejection. Both core conditions are required.
    All candidate components, including rejected ones, are excluded from each halo.
    """
    if mask.shape!=probability.shape or mask.ndim!=2:raise ValueError('Размеры маски и вероятности не совпадают')
    if not 0<=min_core_ratio<=1 or min_core_pixels<1 or not 0<=color_sensitivity<=20:raise ValueError('Некорректные параметры отсева следов')
    if color_sensitivity and (lab is None or lab.shape!=(*mask.shape,3) or lab.dtype!=np.uint8):raise ValueError('Нужен LAB исходного изображения в uint8')
    n,labels,stats,_=cv2.connectedComponentsWithStats(np.ascontiguousarray(mask,dtype=np.uint8),connectivity=8)
    keep=np.zeros(n,bool)
    strong=np.bincount(labels[probability>=high_thresh],minlength=n)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)) # 10 px external halo
    h,w=mask.shape
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if strong[i]<min_core_pixels or strong[i]/area<min_core_ratio:continue
        keep[i]=True
        if not color_sensitivity:continue
        x0=max(0,x-10);y0=max(0,y-10);x1=min(w,x+bw+10);y1=min(h,y+bh+10)
        region=labels[y0:y1,x0:x1];own=region==i
        ring=(cv2.dilate(own.astype(np.uint8),kernel)>0)&(region==0)
        # Without enough surroundings there is no reliable background comparison.
        if np.count_nonzero(ring)<30:continue
        colors=lab[y0:y1,x0:x1]
        obj=colors[own].astype(np.float32);bg=colors[ring].astype(np.float32)
        median_obj=np.median(obj,axis=0);median_bg=np.median(bg,axis=0)
        delta_ab=float(np.linalg.norm(median_obj[1:]-median_bg[1:]))
        # Protect bright white surfaces/highlights, even if their chroma matches sand.
        light90=float(np.percentile(obj[:,0],90))*100/255
        background_l=float(median_bg[0])*100/255
        bright_white=light90>=85 and light90-background_l>=12
        if delta_ab<color_sensitivity and not bright_white:keep[i]=False
    return keep[labels].astype(np.uint8)

def postprocess(probability,lab=None,high_thresh=HIGH,low_thresh=LOW,min_area=MIN_AREA,
        min_seed_pixels=MIN_SEED_PIXELS,max_area=MAX_AREA,exclude_boxes=None,
        color_sensitivity=COLOR_SENSITIVITY,min_core_ratio=MIN_CORE_RATIO,min_core_pixels=MIN_CORE_PIXELS):
    mask=hysteresis(probability,high_thresh,low_thresh,min_area,min_seed_pixels,max_area,exclude_boxes)
    return filter_sand_candidates(mask,probability,lab,high_thresh,color_sensitivity,min_core_ratio,min_core_pixels)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

class ShoreModel:
    def __init__(self,device='auto',precision='auto',batch_size='auto'):
        self.lock=threading.Lock();self.device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        folder=model_dir();path=folder/'model.pt'
        if digest(path)!=EXPECTED:raise ValueError('Контрольная сумма весов V25 не совпала')
        config=SegformerConfig.from_pretrained(folder,local_files_only=True)
        if config.num_labels!=8:raise ValueError('Ожидалось 8 выходов исходной V25')
        self.model=SegformerForSemanticSegmentation(config)
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True);self.model.to(self.device).eval();torch.set_num_threads(2)
        self.processor=SegformerImageProcessor.from_pretrained(folder,local_files_only=True)
        if self.processor.do_resize:raise ValueError('Изменение размера окна отключено')
        self.last_runtime={}
        if precision not in ('auto','fp32','fp16'):raise ValueError('Неизвестная точность вычислений')
        # GP106/GP107 Pascal have very low FP16 throughput (unlike Tensor Core GPUs).
        self.use_amp=self.device=='cuda' and precision!='fp32' and (precision=='fp16' or torch.cuda.get_device_capability()!=(6,1))
        if batch_size not in ('auto',1,2,4):raise ValueError('Неизвестный размер батча')
        self.batch_default=(1 if self.device=='cpu' or torch.cuda.get_device_capability()==(6,1) else 4) if batch_size=='auto' else batch_size

    @torch.inference_mode()
    def predict_batch(self,images,amp=None):
        amp=self.use_amp if amp is None else amp
        pixels=self.processor(images=images,return_tensors='pt')['pixel_values'].to(self.device)
        if tuple(pixels.shape[-2:])!=(768,768):raise ValueError('Неверный размер окна')
        with torch.amp.autocast('cuda',dtype=torch.float16) if self.device=='cuda' and amp else nullcontext():
            logits=self.model(pixel_values=pixels).logits
        z=F.interpolate(logits.float(),size=(768,768),mode='bilinear',align_corners=False)
        # Preserve V25's calibrated binary-max probability definition.
        binary=torch.stack((z[:,:7].amax(1),z[:,7]),dim=1)
        return binary.softmax(dim=1)[:,1]

    @torch.inference_mode()
    def tile(self,image):
        """Single-window compatibility helper; the application uses batches."""
        return self.predict_batch([image])[0].cpu().numpy()

    @torch.inference_mode()
    def predict_probability(self,image,tta=False,batch_size=None,progress=None,amp=None):
        amp=self.use_amp if amp is None else amp
        batch_size=self.batch_default if batch_size is None else batch_size
        with self.lock:
            batch=batch_size
            while True:
                try:
                    probability=blend_tensor(image,lambda images:self.predict_batch(images,amp),self.device,batch,tta,progress)
                    self.last_runtime=dict(device=self.device,batch_size_requested=batch_size,batch_size_effective=batch,
                        precision='fp16_autocast' if self.device=='cuda' and amp else 'fp32',probability_method='binary_max')
                    return probability
                except torch.cuda.OutOfMemoryError:
                    if self.device!='cuda' or batch==1:raise
                    batch=max(1,batch//2);torch.cuda.empty_cache()

    def predict(self,image,tta=False,min_area=MIN_AREA,progress=None,high_thresh=HIGH,low_thresh=LOW,min_seed_pixels=MIN_SEED_PIXELS,batch_size=None,max_area=MAX_AREA,exclude_boxes=None,
            color_sensitivity=COLOR_SENSITIVITY,min_core_ratio=MIN_CORE_RATIO,min_core_pixels=MIN_CORE_PIXELS):
        probability=self.predict_probability(image,tta,batch_size,progress)
        return postprocess(probability,rgb_to_lab(image) if color_sensitivity else None,high_thresh,low_thresh,min_area,
            min_seed_pixels,max_area,exclude_boxes,color_sensitivity,min_core_ratio,min_core_pixels),probability


class ExclusionDetector:
    """Local YOLO11n veto only; these rectangles are never segmentation labels."""
    CLASSES=[0,2,3,5,7]  # person, car, motorcycle, bus, truck
    def __init__(self,device='auto'):
        import os
        config_dir=data_dir()/'.yolo_config';config_dir.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR',str(config_dir));os.environ.setdefault('YOLO_AUTOINSTALL','false')
        from ultralytics import YOLO,settings
        settings.update({'sync':False})
        path=model_dir()/'yolo11n.pt'
        if not path.exists():raise FileNotFoundError('Нет models/yolo11n.pt. Обновите архив приложения.')
        self.sha256=digest(path)
        if self.sha256!='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1':raise ValueError('Контрольная сумма YOLO11n не совпала')
        self.device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        self.model=YOLO(str(path),task='detect');self.lock=threading.Lock()

    def detect(self,image):
        from shore_core import merge_exclusion_detections
        # Whole-frame pass for large vehicles plus tiles for small aerial people.
        w,h=image.size;regions=[(0,0,image)]
        if max(w,h)>1280:
            regions.extend((x,y,image.crop((x,y,min(w,x+1280),min(h,y+1280))))
                for y in range(0,max(1,h-192),1088) for x in range(0,max(1,w-192),1088))
        detections=[]
        with self.lock,torch.inference_mode():
            for x,y,crop in regions:
                result=self.model.predict(source=crop,imgsz=960,conf=.30,iou=.7,classes=self.CLASSES,
                    device=self.device,half=False,verbose=False,save=False)[0]
                if result.boxes is None:continue
                for row in result.boxes.data.cpu().numpy():
                    detections.append(dict(xyxy=[float(row[0])+x,float(row[1])+y,float(row[2])+x,float(row[3])+y],
                        confidence=float(row[4]),class_id=int(row[5]),class_name=str(result.names[int(row[5])])))
        return merge_exclusion_detections(detections)

    def provenance(self):
        return dict(model='yolo11n.pt',sha256=self.sha256,classes=self.CLASSES,conf=.30,iou=.7,imgsz=960,window=1280,overlap=192,nms_iou=.6,
            method='rectangular_probability_veto',license='AGPL-3.0',source='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt')
