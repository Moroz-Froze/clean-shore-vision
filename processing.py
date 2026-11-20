"""One image at a time to disk; result exports remain tied to inference settings."""
from pathlib import Path
import io,json,re,zipfile,hashlib,copy
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cv2
from shore_core import read_image,scale_for,extract_objects,csv_bytes,geojson_bytes,hysteresis,HIGH,LOW,MIN_AREA,MAX_AREA,MIN_SEED_PIXELS
from inference import postprocess,COLOR_SENSITIVITY,MIN_CORE_RATIO,MIN_CORE_PIXELS

def filter_params(settings):
    return dict(high_thresh=settings.get('high_thresh',HIGH),low_thresh=settings.get('low_thresh',LOW),
        min_area=settings.get('min_area',MIN_AREA),max_area=settings.get('max_area',MAX_AREA),
        min_seed_pixels=settings.get('min_seed_pixels',MIN_SEED_PIXELS),exclude_people_vehicles=settings.get('exclude_people_vehicles',False),
        color_sensitivity=settings.get('color_sensitivity',COLOR_SENSITIVITY),min_core_ratio=settings.get('min_core_ratio',MIN_CORE_RATIO),
        min_core_pixels=settings.get('min_core_pixels',MIN_CORE_PIXELS))

def safe_stem(name):
    name=Path(name.replace('\\','/')).stem
    return re.sub(r'[^\w .-]','_',name)[:90] or 'photo'

def process_photo(model,data,filename,index,folder,settings,progress=None,detector=None):
    image,meta=read_image(data);gsd,scale_method=scale_for(meta,settings['scale'])
    params=filter_params(settings)
    use_detector=params.pop('exclude_people_vehicles')
    detections=None;detector_meta=None
    if use_detector:
        if detector is None:raise ValueError('Фильтр людей и транспорта включён, но YOLO не загружен')
        detections=detector.detect(image);detector_meta=detector.provenance()
    mask,prob=model.predict(image,tta=settings['tta'],progress=progress,
        exclude_boxes=[o['xyxy'] for o in detections] if detections else None,**params)
    params['exclude_people_vehicles']=use_detector
    objects=extract_objects(mask,prob,meta,gsd,settings['project'],settings['allow_invalid_gps'])
    key=f'{index:04d}_{safe_stem(filename)}'
    probability_file=f'{key}_probability.npy';np.save(folder/probability_file,prob,allow_pickle=False)
    for obj in objects:obj.update(filename=filename,frame_id=key,scale_method=scale_method,mode='TTA4' if settings['tta'] else 'standard',min_area_px=settings['min_area'],model_sha256=model_hash())
    source_file=folder/f'{key}_source.bin';source_file.write_bytes(data)
    overlay=folder/f'{key}_boxes.jpg';render_boxes(image,objects).save(overlay,quality=93)
    maskpath=folder/f'{key}_mask.png';Image.fromarray(mask).save(maskpath)
    model_mask=folder/f'{key}_model_mask.png';Image.fromarray(mask).save(model_mask)
    original=folder/f'{key}_original_preview.jpg';image.thumbnail((1800,1800));image.save(original,quality=90)
    area_px=int(mask.sum())
    return dict(frame_id=key,filename=filename,source_sha256=hashlib.sha256(data).hexdigest(),metadata=meta,source_file=source_file.name,
        gsd_m_per_px=gsd,scale_method=scale_method,objects=objects,object_count=len(objects),area_px=area_px,
        area_m2=area_px*gsd*gsd if gsd else None,overlay=overlay.name,mask=maskpath.name,original_preview=original.name,
        model_mask=model_mask.name,all_objects=copy.deepcopy(objects),excluded_object_ids=[],review_revision=0,user_reviewed=False,
        probability_file=probability_file,filter_params=params,inference_runtime=copy.deepcopy(model.last_runtime),
        exclusion_detections=detections,exclusion_detector=detector_meta)

def filtered_preview(frame,probability,settings,lab=None):
    """Compute postprocessing only. No model, file writes or mutation of edits."""
    params=filter_params(settings)
    if params['exclude_people_vehicles'] and frame.get('exclusion_detections') is None:raise ValueError('Для исключения людей и транспорта сначала рассчитайте YOLO-рамки')
    boxes=[o['xyxy'] for o in frame.get('exclusion_detections',[]) or []] if params['exclude_people_vehicles'] else None
    mask=postprocess(probability,lab=lab,exclude_boxes=boxes,**{k:v for k,v in params.items() if k!='exclude_people_vehicles'})
    objects=extract_objects(mask,probability,frame['metadata'],frame['gsd_m_per_px'],settings['project'],settings['allow_invalid_gps'])
    for obj in objects:obj.update(filename=frame['filename'],frame_id=frame['frame_id'],scale_method=frame['scale_method'],
        mode=frame['all_objects'][0].get('mode','standard') if frame['all_objects'] else 'standard',min_area_px=params['min_area'],model_sha256=model_hash())
    preview=copy.deepcopy(frame);area=int(mask.sum())
    preview.update(objects=objects,all_objects=copy.deepcopy(objects),excluded_object_ids=[],filter_params=params,
        object_count=len(objects),area_px=area,area_m2=area*frame['gsd_m_per_px']**2 if frame['gsd_m_per_px'] else None)
    return preview,mask

def save_filtered(frame,preview,mask,folder):
    """Only called explicitly if replacing manual edits; archive their assets first."""
    if frame.get('user_reviewed'):
        import uuid,shutil
        archive=folder/'review_snapshots'/uuid.uuid4().hex;archive.mkdir(parents=True)
        (archive/'frame.json').write_text(json.dumps(frame,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ('mask','model_mask','overlay'):shutil.copy2(folder/frame[key],archive/Path(frame[key]).name)
    for key in ('mask','model_mask'):Image.fromarray(mask).save(folder/frame[key])
    preview.update(review_revision=frame.get('review_revision',0)+1,user_reviewed=False,overlay_dirty=True)
    for key in ('review_history','review_history_index','last_photo_action'):preview.pop(key,None)
    frame.clear();frame.update(preview)

def ensure_overlay(frame,folder):
    if frame.get('overlay_dirty'):
        image,_=read_image((folder/frame['source_file']).read_bytes())
        render_boxes(image,frame['objects']).save(folder/frame['overlay'],quality=93);frame['overlay_dirty']=False

def render_boxes(image,objects):
    image=image.copy();draw=ImageDraw.Draw(image);size=max(image.size)
    width=max(2,round(size/700));font=ImageFont.load_default(size=max(14,round(size/105)))
    for obj in objects:
        x,y,w,h=(obj[k] for k in ('bbox_x','bbox_y','bbox_width','bbox_height'))
        draw.rectangle((x,y,min(image.width-1,x+w-1),min(image.height-1,y+h-1)),outline='#2463eb',width=width)
        text=str(obj['object_id']);box=draw.textbbox((0,0),text,font=font);tw,th=box[2]-box[0],box[3]-box[1]
        px=min(x,max(0,image.width-tw-10));py=max(0,y-th-12)
        draw.rounded_rectangle((px,py,px+tw+10,py+th+8),radius=3,fill='#2463eb')
        draw.text((px+5,py+3-box[1]),text,font=font,fill='white')
    return image

def apply_review(frame,folder,excluded_ids):
    objects=frame['all_objects'];known={o['object_id'] for o in objects};excluded=set(map(int,excluded_ids))
    if not excluded<=known:raise ValueError('Неизвестный номер объекта')
    raw=np.array(Image.open(folder/frame['model_mask']))
    n,labels=cv2.connectedComponents(np.ascontiguousarray(raw,dtype=np.uint8),connectivity=8)
    if known!=set(range(1,n)):raise ValueError('Объекты не соответствуют исходной маске')
    keep=np.zeros(n,bool)
    for obj in objects:
        if obj['object_id'] not in excluded:keep[obj['object_id']]=True
    reviewed=keep[labels].astype(np.uint8)
    image,_=read_image((folder/frame['source_file']).read_bytes())
    active=[copy.deepcopy(o) for o in objects if o['object_id'] not in excluded]
    render_boxes(image,active).save(folder/frame['overlay'],quality=93)
    Image.fromarray(reviewed).save(folder/frame['mask'])
    frame.update(objects=active,excluded_object_ids=sorted(excluded),object_count=len(active),area_px=int(reviewed.sum()),overlay_dirty=False,
        area_m2=sum(o['area_m2'] for o in active) if frame.get('gsd_m_per_px') else None,
        user_reviewed=True,review_revision=frame.get('review_revision',0)+1)

def model_hash():return '28f88aaecb1771b61b98180b3b1aaa104ece522000f294fc66d0842856ffa899'

def write_exports(folder,frames,errors,settings,build_zip=True):
    objects=[o for frame in frames for o in frame['objects']]
    (folder/'detections.csv').write_bytes(csv_bytes(objects));(folder/'detections.geojson').write_bytes(geojson_bytes(objects))
    if build_zip:
        for frame in frames:ensure_overlay(frame,folder)
    manifest={'model':'V25','model_sha256':model_hash(),'inference':{'window':768,'overlap':192,'probability_method':'binary_max',**settings},
        'postprocessing_params_are_recorded_per_frame':True,
        'frames':frames,'errors':errors,'objects_are_connected_components_not_unique_physical_items':True,
        'frame_overlaps_not_deduplicated':True,'area_m2_is_approximate_and_requires_scale':True,
        'frame_gps_not_object_gps_unless_explicit_approx_projection':True}
    (folder/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    changes={'format':'shore-object-review-v1','model_sha256':model_hash(),'frames':[{'frame_id':f['frame_id'],'filename':f['filename'],
        'source_sha256':f.get('source_sha256'),'excluded_object_ids':f.get('excluded_object_ids',[]),'user_reviewed':f.get('user_reviewed',False)} for f in frames]}
    (folder/'review_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
    if build_zip:
        with zipfile.ZipFile(folder/'processed_photos.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for frame in frames:
                for key in ('overlay','mask'):archive.write(folder/frame[key],frame[key])
            for name in ('detections.csv','detections.geojson','run_manifest.json','review_changes.json'):archive.write(folder/name,name)
    return manifest
