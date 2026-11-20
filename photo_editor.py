"""Local, dependency-free photo selection; reject stale component events."""
from pathlib import Path
import base64
import streamlit.components.v1 as components
from PIL import Image

_editor=components.declare_component('shore_photo_editor',path=str(Path(__file__).parent/'photo_component'))

def selected_exclusions(frame,event):
    if not isinstance(event,dict) or event.get('revision')!=frame.get('review_revision',0):return None
    if not event.get('token') or event['token']==frame.get('last_photo_action'):return None
    ids=event.get('ids')
    if not isinstance(ids,list) or not ids or any(type(v) is not int for v in ids):return None
    if not set(ids)<={o['object_id'] for o in frame['objects']}:return None
    return sorted(set(frame.get('excluded_object_ids',[]))|set(ids))

def photo_editor(frame,folder,read_only=False,max_height=None):
    source=folder/frame['original_preview']
    width,height=frame['metadata']['width'],frame['metadata']['height']
    return _editor(image='data:image/jpeg;base64,'+base64.b64encode(source.read_bytes()).decode('ascii'),
        image_width=width,image_height=height,objects=frame['objects'],revision=frame.get('review_revision',0),read_only=read_only,max_height=max_height,
        key=f'photo_{folder.name}_{frame["frame_id"]}',default=None)
