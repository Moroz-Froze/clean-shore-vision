"""Native-pixel segmentation, DJI metadata and explicit approximate ground scale."""
import csv,io,json,math,re
import numpy as np
import cv2
from PIL import Image,ImageOps
from security import limit,InputRejected
cv2.setNumThreads(2)
HIGH=.75;LOW=.51;MIN_AREA=120;MAX_AREA=10000;MIN_SEED_PIXELS=12;WINDOW=768;OVERLAP=192

def hysteresis(probability,high=HIGH,low=LOW,min_area=MIN_AREA,min_seed_pixels=MIN_SEED_PIXELS,max_area=None,exclude_boxes=None):
    p=np.asarray(probability)
    if p.ndim!=2 or not p.size or not np.isfinite(p).all() or not 0<=low<=high<=1 or min_area<1 or type(min_seed_pixels) is not int or min_seed_pixels<1 or p.min()<0 or p.max()>1:
        raise ValueError('Некорректные вероятности или пороги')
    if max_area is not None and max_area<min_area:raise ValueError('Максимальная площадь меньше минимальной')
    if exclude_boxes:
        p=p.copy();h,w=p.shape
        for box in exclude_boxes:
            if len(box)!=4 or not np.isfinite(box).all():raise ValueError('Некорректная рамка исключения')
            x1,y1,x2,y2=box
            x1,y1=max(0,min(w,int(math.floor(x1)))),max(0,min(h,int(math.floor(y1))))
            x2,y2=max(0,min(w,int(math.ceil(x2)))),max(0,min(h,int(math.ceil(y2))))
            if x2>x1 and y2>y1:p[y1:y2,x1:x2]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(np.ascontiguousarray(p>=low,dtype=np.uint8),connectivity=8)
    seeds=np.bincount(labels[p>=high],minlength=n)
    keep=seeds>=min_seed_pixels;keep[0]=False
    keep &= stats[:,cv2.CC_STAT_AREA]>=min_area
    if max_area is not None:keep &= stats[:,cv2.CC_STAT_AREA]<=max_area
    return keep[labels].astype(np.uint8)

def merge_exclusion_detections(detections,iou_threshold=.6):
    """Class-aware NMS in original coordinates; never merge people with cars."""
    kept=[]
    for obj in sorted(detections,key=lambda o:o['confidence'],reverse=True):
        a=np.asarray(obj['xyxy'],dtype=float);duplicate=False
        for prior in kept:
            if obj['class_id']!=prior['class_id']:continue
            b=np.asarray(prior['xyxy'],dtype=float)
            inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
            if union>0 and inter/union>iou_threshold:duplicate=True;break
        if not duplicate:kept.append(obj)
    return kept

def tta_probability(image,predict):
    rgb=np.asarray(image);total=np.zeros(rgb.shape[:2],np.float32)
    for axes in ((),(1,),(0,),(0,1)):
        transformed=np.flip(rgb,axis=axes) if axes else rgb
        prob=np.asarray(predict(Image.fromarray(np.ascontiguousarray(transformed))),np.float32)
        total+=np.flip(prob,axis=axes) if axes else prob
    return total/4

def starts(length):return range(0,max(1,length-OVERLAP),WINDOW-OVERLAP)

def window_crop(image,x,y):
    w,h=image.size;rw,rh=min(WINDOW,w-x),min(WINDOW,h-y)
    crop=image.crop((x,y,x+rw,y+rh))
    if (rw,rh)!=(WINDOW,WINDOW):
        a=np.asarray(crop);a=np.pad(a,((0,WINDOW-rh),(0,WINDOW-rw),(0,0)),mode='reflect');crop=Image.fromarray(a)
    return crop,rw,rh

def blend_tensor(image,predict_batch,device,batch_size=4,tta=False,progress=None):
    """Blend float32 tensors on the inference device; one final CPU transfer."""
    import torch
    if type(batch_size) is not int or batch_size<1:raise ValueError('Некорректный размер батча')
    w,h=image.size;total=torch.zeros((h,w),dtype=torch.float32,device=device);count=torch.zeros_like(total)
    boxes=[(x,y) for y in starts(h) for x in starts(w)]
    flips=((),(2,),(1,),(1,2)) if tta else ((),)
    for begin in range(0,len(boxes),batch_size):
        group=boxes[begin:begin+batch_size];crops=[window_crop(image,x,y) for x,y in group]
        probability=None
        for axes in flips:
            images=[Image.fromarray(np.ascontiguousarray(np.flip(np.asarray(im),axis=tuple(a-1 for a in axes)))) if axes else im for im,_,_ in crops]
            p=predict_batch(images).float()
            if axes:p=torch.flip(p,dims=axes)
            probability=p if probability is None else probability+p
        probability/=len(flips)
        for j,((x,y),(_,rw,rh)) in enumerate(zip(group,crops)):
            total[y:y+rh,x:x+rw]+=probability[j,:rh,:rw];count[y:y+rh,x:x+rw]+=1
        if progress:progress(min(begin+len(group),len(boxes)),len(boxes))
    return (total/count).cpu().numpy()

def blend(image,predict,tta=False,progress=None):
    w,h=image.size;total=np.zeros((h,w),np.float32);count=np.zeros((h,w),np.uint16)
    boxes=[(x,y) for y in starts(h) for x in starts(w)]
    for i,(x,y) in enumerate(boxes,1):
        rw,rh=min(WINDOW,w-x),min(WINDOW,h-y)
        crop=image.crop((x,y,x+rw,y+rh))
        if (rw,rh)!=(WINDOW,WINDOW):
            a=np.asarray(crop);a=np.pad(a,((0,WINDOW-rh),(0,WINDOW-rw),(0,0)),mode='reflect');crop=Image.fromarray(a)
        p=tta_probability(crop,predict) if tta else predict(crop)
        total[y:y+rh,x:x+rw]+=p[:rh,:rw];count[y:y+rh,x:x+rw]+=1
        if progress:progress(i,len(boxes))
    if not (count>0).all():raise ValueError('Остались необработанные пиксели')
    return total/count

def parse_xmp(data):
    values={}
    for key,value in re.findall(rb'drone-dji:([A-Za-z0-9]+)\s*=\s*[\x22\x27]([^\x22\x27]*)[\x22\x27]',data):
        values[key.decode('ascii')]=value.decode('utf-8',errors='replace')
    for key,value in re.findall(rb'<drone-dji:([A-Za-z0-9]+)>([^<]*)</drone-dji:[A-Za-z0-9]+>',data):
        values[key.decode('ascii')]=value.decode('utf-8',errors='replace')
    return values

def number(value):
    try:
        n=float(value);return n if math.isfinite(n) else None
    except (TypeError,ValueError,ZeroDivisionError):return None

def dms(value,ref):
    try:
        result=float(value[0])+float(value[1])/60+float(value[2])/3600
        return -result if str(ref).strip().upper() in ('S','W') else result
    except (TypeError,IndexError,ValueError,ZeroDivisionError):return None

def read_image(data):
    if not data or len(data)>limit("MAX_UPLOAD_MB",100)*1024**2:raise InputRejected("Недопустимый размер файла.")
    xmp=parse_xmp(data[:4*1024*1024])
    with Image.open(io.BytesIO(data),formats=("JPEG","PNG","TIFF")) as raw:
        if raw.format not in ("JPEG","PNG","TIFF"):raise InputRejected("Разрешены только JPEG, PNG и TIFF.")
        if raw.width*raw.height>limit("MAX_IMAGE_PIXELS",24000000):raise InputRejected("Снимок превышает лимит пикселей сервера.")
        if getattr(raw,"n_frames",1)!=1:raise InputRejected("Многостраничные изображения не поддерживаются.")
        exif=raw.getexif()
        try:gps=exif.get_ifd(34853)
        except (KeyError,ValueError,TypeError):gps={}
        try:lens=exif.get_ifd(34665)
        except (KeyError,ValueError,TypeError):lens={}
        lat=number(xmp.get('GpsLatitude',xmp.get('GPSLatitude')));lon=number(xmp.get('GpsLongitude',xmp.get('GPSLongitude')))
        if lat is None:lat=dms(gps.get(2),gps.get(1))
        if lon is None:lon=dms(gps.get(4),gps.get(3))
        status=str(xmp.get('GpsStatus',gps.get(9,''))).strip('\x00 ').lower()
        valid=status not in ('invalid','v','void','0')
        if lat is None or lon is None or not(-90<=lat<=90 and -180<=lon<=180):lat=lon=None;valid=False
        orientation=int(exif.get(274,1))
        image=ImageOps.exif_transpose(raw).convert('RGB')
        meta=dict(width=image.width,height=image.height,latitude=lat,longitude=lon,gps_valid=valid,gps_status=status or 'unspecified',
            relative_altitude=number(xmp.get('RelativeAltitude')),pitch=number(xmp.get('GimbalPitchDegree')),
            yaw=number(xmp.get('GimbalYawDegree')),roll=number(xmp.get('GimbalRollDegree')),
            focal35=number(lens.get(41989,exif.get(41989))),orientation=orientation,
            camera=str(exif.get(272,'')),captured_at=str(lens.get(36867,exif.get(306,''))))
    return image,meta

def scale_for(meta,settings):
    mode=settings.get('mode','pixels')
    if mode=='gsd':
        cm=number(settings.get('gsd_cm'));return (cm/100,'user_gsd') if cm and cm>0 else (None,'missing_gsd')
    if mode!='altitude':return None,'pixels_only'
    if not settings.get('confirm_flat'):return None,'relative_altitude_not_confirmed_AGL'
    pitch=meta.get('pitch');h=meta.get('relative_altitude');f=meta.get('focal35')
    if pitch is None or abs(pitch+90)>2:return None,'camera_not_nadir'
    if not h or h<=0 or not f or f<=0:return None,'missing_height_or_focal35'
    gsd=h*math.hypot(36,24)/(f*math.hypot(meta['width'],meta['height']))
    return gsd,'approx_relative_altitude_flat_ground'

def object_location(meta,x,y,gsd,project,allow_invalid):
    lat,lon=meta.get('latitude'),meta.get('longitude')
    if lat is None or lon is None:return None,None,'gps_missing'
    if not meta.get('gps_valid',False) and not allow_invalid:return None,None,'gps_invalid'
    method='frame_gps' if meta.get('gps_valid') else 'frame_gps_invalid_allowed'
    pitch=meta.get('pitch');yaw=meta.get('yaw');roll=meta.get('roll',0)
    if not (project and gsd and pitch is not None and abs(pitch+90)<=2 and yaw is not None and meta.get('orientation',1)==1 and (roll is None or abs(roll)<=2)):
        return lat,lon,method
    if abs(lat)>=85:return lat,lon,method
    east=(x-meta['width']/2)*gsd;north=(meta['height']/2-y)*gsd;t=math.radians(yaw)
    e=east*math.cos(t)+north*math.sin(t);n=-east*math.sin(t)+north*math.cos(t)
    return lat+math.degrees(n/6378137),lon+math.degrees(e/(6378137*math.cos(math.radians(lat)))),'approx_nadir_projection' if meta.get('gps_valid') else 'approx_projection_invalid_gps_allowed'

def extract_objects(mask,prob,meta,gsd,project,allow_invalid):
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(np.ascontiguousarray(mask,dtype=np.uint8),connectivity=8)
    result=[]
    for i in range(1,n):
        x,y=map(float,centroids[i]);area=int(stats[i,cv2.CC_STAT_AREA])
        bx,by,bw,bh=map(int,stats[i,:4]);values=prob[by:by+bh,bx:bx+bw][labels[by:by+bh,bx:bx+bw]==i]
        lat,lon,method=object_location(meta,x,y,gsd,project,allow_invalid)
        result.append(dict(object_id=i,area_px=area,area_m2=area*gsd*gsd if gsd else None,
            centroid_x_px=x,centroid_y_px=y,bbox_x=int(stats[i,0]),bbox_y=int(stats[i,1]),bbox_width=int(stats[i,2]),bbox_height=int(stats[i,3]),
            probability_peak=float(values.max()),probability_mean=float(values.mean()),
            latitude=lat,longitude=lon,location_method=method,gps_status=meta.get('gps_status','unknown'),gsd_m_per_px=gsd))
    return result

FIELDS=['filename','frame_id','object_id','area_px','area_m2','centroid_x_px','centroid_y_px','bbox_x','bbox_y','bbox_width','bbox_height',
        'probability_peak','probability_mean','latitude','longitude','location_method','gps_status','gsd_m_per_px','scale_method','mode','min_area_px','model_sha256']

def csv_bytes(objects):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,FIELDS,extrasaction='ignore');writer.writeheader()
    for obj in objects:
        row=dict(obj)
        for key,value in row.items():
            if isinstance(value,str) and (value.startswith(("\t","\r","\n","\0")) or value.lstrip().startswith(("=","+","-","@","＝","＋","－","＠"))):
                row[key]="\'"+value
        writer.writerow(row)
    return stream.getvalue().encode('utf-8-sig')

def geojson_bytes(objects):
    features=[]
    for obj in objects:
        lat,lon=obj.get('latitude'),obj.get('longitude')
        geometry={'type':'Point','coordinates':[lon,lat]} if lat is not None and lon is not None else None
        features.append({'type':'Feature','geometry':geometry,'properties':obj})
    return json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
