from pathlib import Path
import uuid
import numpy as np
import time
import streamlit as st
from PIL import Image
from inference import ShoreModel,ExclusionDetector,rgb_to_lab
from shore_core import read_image
from processing import process_photo,write_exports,apply_review,filter_params,filtered_preview,save_filtered,ensure_overlay
from settings_ui import applied,seed_draft,settings_dialog
from local_state import remember_job,restore_job,save_local
from photo_editor import photo_editor,selected_exclusions
from review_history import history_state,history_edit,history_move
from runtime_paths import new_job_dir,server_mode
from security import validate_uploads,analysis_slot,check_storage,public_error
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Чистый берег',page_icon=str(ROOT/'icon.svg'),layout='wide',initial_sidebar_state='expanded')

@st.cache_resource(show_spinner='Загрузка модели…')
def _load_model(device,precision='auto',batch_size='auto'):return ShoreModel(device,precision,batch_size)

def load_model(device,precision='auto',batch_size='auto'):
    return _load_model('cpu','fp32',1) if server_mode() else _load_model(device,precision,batch_size)

@st.cache_resource(show_spinner='Загрузка фильтра людей и транспорта…')
def _load_exclusion_detector(device='auto'):return ExclusionDetector(device)

def load_exclusion_detector(device='auto'):
    return _load_exclusion_detector('cpu' if server_mode() else device)

def navigate(view):st.session_state['view']=view

def rail(job):
    with st.sidebar:
        st.markdown('<div class="rail-brand">'+(ROOT/'icon.svg').read_text(encoding='utf-8')+'</div>',unsafe_allow_html=True)
        with st.container(key='rail_nav'):
            for view,label,icon in [('upload','Добавить снимки','add'),('gallery','Галерея','crop_free'),('export','Экспорт','download')]:
                st.button(label,icon=f':material/{icon}:',key='nav_'+view,help=label,width='stretch',
                    type='primary' if st.session_state['view']==view else 'tertiary',disabled=not job and view!='upload',on_click=navigate,args=(view,))
        with st.container(key='rail_bottom'):
            st.session_state.setdefault('live_color',float(applied()['color_sensitivity']))
            with st.popover('Отсев следов',icon=':material/texture:',help='Отсев следов на песке',width='stretch'):
                st.slider('Чувствительность к цвету (отсев следов на песке)',0.,20.,step=.5,key='live_color',on_change=update_color_filter,
                    help='Порог delta_ab в LAB. Больше — строже. 0 отключает цветовой фильтр. Кольцо 10 px; яркие белые участки защищены. Прозрачный пластик цвета песка тоже может быть отброшен.')
                st.caption('Уверенное ядро: не меньше 15% объекта и 15 пикселей выше верхнего порога.')
            if st.button('Настройки',icon=':material/settings:',key='settings',help='Настройки',type='tertiary',width='stretch'):
                seed_draft();st.session_state['settings_open']=True

def update_color_filter():
    cfg=applied();cfg['color_sensitivity']=st.session_state['live_color'];save_local('.preferences.json',cfg)

def update_filters():
    cfg=applied();high=st.session_state.get('live_high',cfg['high_thresh'])
    low=min(st.session_state.get('live_low',cfg['low_thresh']),high)
    st.session_state['live_low']=low
    cfg.update(high_thresh=high,low_thresh=low,min_area=st.session_state.get('live_area',cfg['min_area']),
        max_area=st.session_state.get('live_max_area',cfg['max_area']),
        exclude_people_vehicles=st.session_state.get('live_exclude',cfg['exclude_people_vehicles']),min_seed_pixels=12)
    save_local('.preferences.json',cfg)


def filter_controls():
    cfg=applied()
    for key,value in (('live_high',float(cfg['high_thresh'])),('live_low',float(cfg['low_thresh'])),('live_area',int(cfg['min_area'])),
        ('live_max_area',int(cfg['max_area'])),('live_exclude',bool(cfg['exclude_people_vehicles']))):
        st.session_state.setdefault(key,value)
    with st.container(key='photo_filters'):
        a,b,c,d=st.columns(4)
        a.slider('Верхний порог',.05,.99,step=.01,key='live_high',on_change=update_filters)
        b.slider('Нижний порог',.0,float(cfg['high_thresh']),step=.01,key='live_low',on_change=update_filters)
        c.slider('Минимальная площадь, px',1,1000,key='live_area',on_change=update_filters)
        d.slider('Максимальная площадь, px',2000,50000,step=100,key='live_max_area',on_change=update_filters)
        st.checkbox('Исключать людей и машины',key='live_exclude',on_change=update_filters,
            help='YOLO11n: люди, легковые машины, мотоциклы, автобусы и грузовики. Рамки исключаются целиком; мусор внутри них тоже может исчезнуть. Детекции кэшируются, ползунки не запускают YOLO повторно.')


def summary(frames):
    with st.container(key='summary'):
        cols=st.columns(3);scaled=[f for f in frames if f['area_m2'] is not None]
        cols[0].metric('Объекты',sum(f['object_count'] for f in frames))
        cols[1].metric('Площадь, м²',f'{sum(f["area_m2"] for f in scaled):.3f}' if scaled else '—',help='Оценка по масштабу, сумма по кадрам без вычитания перекрытий.')
        cols[2].metric('Снимки',len(frames))

def update_review(job,frame,excluded,history=None):
    history=history if history is not None else history_edit(frame,excluded)
    folder=Path(job['folder']);apply_review(frame,folder,excluded);frame.update(history)
    write_exports(folder,job['frames'],job['errors'],job['settings'],build_zip=False)
    remember_job(job)
    job['exports_dirty']=True;st.rerun()

def live_filter(job,frame):
    settings={**job['settings'],**filter_params(applied())}
    if frame.get('filter_params')==filter_params(settings):return None
    folder=Path(job['folder']);name=frame.get('probability_file')
    if not name or not (folder/name).exists():
        st.caption('Для быстрых ползунков повторите обработку этого снимка один раз: прежний запуск не сохранил карту вероятностей.');return None
    if settings['exclude_people_vehicles'] and frame.get('exclusion_detections') is None:
        try:
            with st.spinner('Поиск людей и транспорта…'):
                detector=load_exclusion_detector(applied()['device'])
                image,_=read_image((folder/frame['source_file']).read_bytes())
                detections=detector.detect(image)
            frame['exclusion_detections']=detections;frame['exclusion_detector']=detector.provenance()
            st.session_state['exclusion_detections']={'source_sha256':frame.get('source_sha256'),'detections':detections,'detector':frame['exclusion_detector']}
            write_exports(folder,job['frames'],job['errors'],job['settings'],build_zip=False);remember_job(job)
        except Exception as exc:
            st.error(public_error(exc));return None
    key=(str(folder),name)
    cache=st.session_state.get('probability')
    if not cache or cache['key']!=key:
        cache={'key':key,'array':np.load(folder/name,allow_pickle=False)};st.session_state['probability']=cache
    lab_cache=st.session_state.get('image_lab')
    if settings['color_sensitivity'] and (not lab_cache or lab_cache['key']!=key):
        image,_=read_image((folder/frame['source_file']).read_bytes())
        lab_cache={'key':key,'array':rgb_to_lab(image)};st.session_state['image_lab']=lab_cache
    signature=(key,tuple(filter_params(settings).values()),frame.get('review_revision',0))
    previous=st.session_state.get('filter_preview')
    if previous and previous['signature']==signature:preview,mask=previous['frame'],previous['mask']
    else:
        started=time.perf_counter();preview,mask=filtered_preview(frame,cache['array'],settings,lab_cache['array'] if settings['color_sensitivity'] else None)
        st.session_state['postprocess_seconds']=time.perf_counter()-started
        st.session_state['filter_preview']=dict(signature=signature,frame=preview,mask=mask)
    if frame.get('user_reviewed'):
        st.info('Предпросмотр новых порогов. Ручные правки пока сохранены в исходном результате.')
        if not st.button('Применить фильтр к снимку',key='apply_live_filter',help='Создаст новую разметку; предыдущие ручные правки сохранятся в review_snapshots.'):return preview
    save_filtered(frame,preview,mask,folder);job['exports_dirty']=True
    write_exports(folder,job['frames'],job['errors'],job['settings'],build_zip=False);remember_job(job)
    return None

def choose_frame(index):st.session_state['frame_picker']=index


@st.cache_data(show_spinner=False,max_entries=256)
def thumbnail(path):
    import io
    with Image.open(path) as im:
        im.thumbnail((140,100));out=io.BytesIO();im.convert('RGB').save(out,format='JPEG',quality=80)
    return out.getvalue()


def frame_inspector(job,index,shown):
    frames=job['frames'];folder=Path(job['folder']);meta=shown['metadata']
    with st.container(key='frame_inspector'):
        st.markdown('**Сведения о кадре**')
        a,b=st.columns(2)
        a.metric('Объекты',shown['object_count'])
        b.metric('Площадь, м²',f'{shown["area_m2"]:.3f}' if shown['area_m2'] is not None else '—')
        st.caption(f'{meta["width"]} × {meta["height"]} px')
        if meta.get('camera'):st.caption(meta['camera'])
        if meta.get('captured_at'):st.caption(meta['captured_at'])
    with st.container(key='filmstrip'):
        st.markdown(f'**Снимки · {len(frames)}**')
        with st.container(height=410,border=False,key='filmstrip_scroll'):
            for i,item in enumerate(frames):
                with st.container(key=f'film_item_{i}'):
                    im,text=st.columns([1,2.2],gap='small',vertical_alignment='center')
                    path=folder/item.get('original_preview',item['overlay'])
                    im.image(thumbnail(str(path)),width='stretch')
                    text.button(item['filename'],key=f'thumb_{i}',type='primary' if i==index else 'tertiary',
                        width='stretch',on_click=choose_frame,args=(i,))
                    text.caption(f'{item["object_count"]} объектов')


def gallery(job):
    frames=job['frames'];folder=Path(job['folder'])
    if not frames:st.info('Нет обработанных снимков.');return
    with st.container(key='workspace_header'):
        name,prev,next_,back,forward=st.columns([6,1,1,1,1],gap='small',vertical_alignment='center')
        current=min(st.session_state.get('frame_picker',0),len(frames)-1)
        prev.button('Предыдущий снимок',icon=':material/chevron_left:',key='previous_frame',disabled=current==0,
            on_click=choose_frame,args=(current-1,),help='Предыдущий снимок',width='stretch')
        next_.button('Следующий снимок',icon=':material/chevron_right:',key='next_frame',disabled=current==len(frames)-1,
            on_click=choose_frame,args=(current+1,),help='Следующий снимок',width='stretch')
        index=name.selectbox('Снимок',range(len(frames)),format_func=lambda i:frames[i]['filename'],label_visibility='collapsed',key='frame_picker')
        frame=frames[index];editable='all_objects' in frame and 'source_file' in frame
        states,cursor=history_state(frame)
        if back.button('Назад',icon=':material/undo:',key='undo_review',disabled=not editable or cursor==0,
            help='Отменить изменение',width='stretch'):
            state,ids=history_move(frame,-1);update_review(job,frame,ids,history=state)
        if forward.button('Вперёд',icon=':material/redo:',key='redo_review',disabled=not editable or cursor==len(states)-1,
            help='Повторить изменение',width='stretch'):
            state,ids=history_move(frame,1);update_review(job,frame,ids,history=state)
    with st.container(key='workspace_status'):status=st.empty()
    canvas,inspector=st.columns([4.2,1.3],gap='medium')
    with canvas:
        preview=live_filter(job,frame) if editable else None
        shown=preview or frame
        with st.container(key='photo_panel'):
            if preview:photo_editor(preview,folder,read_only=True,max_height=560)
            elif editable:
                event=photo_editor(frame,folder,max_height=560)
                exclusions=selected_exclusions(frame,event)
                if exclusions is not None:
                    frame['last_photo_action']=event['token'];update_review(job,frame,exclusions)
            else:st.image(str(folder/frame['overlay']),width='stretch')
        filter_controls()
        if frame['objects'] and not preview:
            zoom=st.selectbox('Приблизить объект',[None]+[o['object_id'] for o in frame['objects']],
                format_func=lambda v:'Весь снимок' if v is None else f'Объект {v}',key=f'zoom_{frame["frame_id"]}_{frame.get("review_revision",0)}')
            if zoom is not None:
                ensure_overlay(frame,folder)
                obj=next(o for o in frame['objects'] if o['object_id']==zoom)
                im=Image.open(folder/frame['overlay']);pad=max(80,obj['bbox_width'],obj['bbox_height'])
                box=(max(0,obj['bbox_x']-pad),max(0,obj['bbox_y']-pad),min(im.width,obj['bbox_x']+obj['bbox_width']+pad),min(im.height,obj['bbox_y']+obj['bbox_height']+pad))
                st.image(im.crop(box),caption=f'Объект {zoom}',width=min(650,box[2]-box[0]))
    with inspector:frame_inspector(job,index,shown)
    status.caption(f'Снимок {index+1} из {len(frames)} · Всего объектов: {sum(f["object_count"] for f in frames)}')


def export_view(job):
    folder=Path(job['folder'])
    if job.get('exports_dirty') or not (folder/'processed_photos.zip').exists():
        with st.spinner('Подготовка файлов…'):write_exports(folder,job['frames'],job['errors'],job['settings'])
        job['exports_dirty']=False
    cols=st.columns(3)
    for col,(name,label,mime) in zip(cols,[('detections.csv','Таблица CSV','text/csv'),('detections.geojson','Карта GeoJSON','application/geo+json'),('processed_photos.zip','Фото и маски ZIP','application/zip')]):
        with (folder/name).open('rb') as f:col.download_button(label,f,file_name=name,mime=mime,icon=':material/download:',width='stretch',key=name)
    st.caption('Только оставленные объекты. Правки сохранены в review_changes.json внутри ZIP.')

def run_analysis(uploads,settings,device):
    try:
        validate_uploads(uploads)
        with analysis_slot():
            check_storage()
            _run_analysis(uploads,settings,device)
    except ValueError as exc:st.error(public_error(exc))


def _run_analysis(uploads,settings,device):
    try:
        model=load_model(device,applied()['precision'],applied()['batch_size'])
        detector=load_exclusion_detector(device) if settings.get('exclude_people_vehicles') else None
    except Exception as exc:st.error(public_error(exc));return
    folder=new_job_dir();frames=[];errors=[]
    bar=st.progress(0.,text='Обработка…')
    for i,uploaded in enumerate(uploads):
        try:
            check_storage()
            def progress(done,total):bar.progress((i+done/total)/len(uploads),text=f'{i+1} / {len(uploads)} · {uploaded.name}')
            frames.append(process_photo(model,uploaded.getvalue(),uploaded.name,i+1,folder,settings,progress,detector=detector))
        except Exception as exc:errors.append({'filename':uploaded.name,'error':public_error(exc)})
        bar.progress((i+1)/len(uploads),text=f'{i+1} / {len(uploads)}')
    write_exports(folder,frames,errors,settings,build_zip=False)
    st.session_state.pop('probability',None);st.session_state.pop('filter_preview',None)
    st.session_state['job']=dict(folder=str(folder),frames=frames,errors=errors,settings=settings,exports_dirty=True)
    remember_job(st.session_state['job'])
    st.session_state['view']='gallery';st.session_state.pop('frame_picker',None);st.rerun()

def upload_view(settings,device):
    with st.container(key='welcome'):
        st.title('Какие снимки проверим?')
        with st.container(key='upload_bar'):
            a,b=st.columns([10,1],vertical_alignment='center')
            with a:uploads=st.file_uploader('Добавить снимки',type=['jpg','jpeg','png','tif','tiff'],accept_multiple_files=True,label_visibility='collapsed',key='uploads')
            with b:
                with st.container(key='analyze'):submitted=st.button('Обработать',icon=':material/arrow_upward:',key='analyze_button',type='primary',disabled=not uploads,help='Обработать снимки')
        st.markdown('<div class="welcome-note">JPG, PNG или TIFF</div>',unsafe_allow_html=True)
    if submitted:run_analysis(uploads,settings,device)

def main():
    st.markdown('<style>'+ (ROOT/'style.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)
    cfg=applied()
    if 'job' not in st.session_state:
        recovered=restore_job()
        if recovered:st.session_state['job']=recovered;st.session_state['view']='gallery'
    st.session_state.setdefault('view','upload')
    if st.session_state['view'] not in ('upload','gallery','export'):st.session_state['view']='gallery'
    job=st.session_state.get('job');rail(job)
    settings={k:v for k,v in cfg.items() if k not in ('device','precision','batch_size')}
    if st.session_state.get('settings_open'):settings_dialog()
    st.markdown('<div class="app-brand">Чистый берег</div>',unsafe_allow_html=True)
    view=st.session_state['view']
    with st.container(key='page_'+view):
        if view=='upload' or not job:upload_view(settings,cfg['device']);return
        if view=='export':st.title('Экспорт')
        if view!='gallery':summary(job['frames'])
        if any(job['settings'].get(k)!=settings[k] for k in ('tta','scale','project','allow_invalid_gps')):st.info('Новые настройки модели и географии применятся при следующей обработке.')
        if job['errors']:
            with st.expander(f'Не обработано: {len(job["errors"])}'):st.json(job['errors'])
        if view=='gallery':gallery(job)
        elif view=='export':export_view(job)

if __name__=='__main__':main()
