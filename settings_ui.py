"""Draft settings live in a modal; Cancel never changes applied parameters."""
import copy
import streamlit as st
from local_state import read_local,save_local
DEFAULTS=dict(tta=False,min_area=120,max_area=10000,high_thresh=.75,low_thresh=.51,min_seed_pixels=12,exclude_people_vehicles=False,color_sensitivity=6.,min_core_ratio=.15,min_core_pixels=15,scale={'mode':'pixels'},project=False,allow_invalid_gps=False,device='auto',precision='auto',batch_size='auto')

def applied():
    if 'app_settings' not in st.session_state:
        saved=read_local('.preferences.json')
        st.session_state['app_settings']=copy.deepcopy(DEFAULTS)
        if isinstance(saved,dict):st.session_state['app_settings'].update({k:v for k,v in saved.items() if k in DEFAULTS})
    return st.session_state['app_settings']

def seed_draft():
    cfg=applied();s=cfg['scale']
    values={'mode':'С TTA' if cfg['tta'] else 'Стандартный','area':cfg['min_area'],
        'scale':{'pixels':'Не задан','gsd':'GSD','altitude':'Высота DJI'}[s['mode']],
        'gsd':float(s.get('gsd_cm',1.)),'flat':s.get('confirm_flat',False),'project':cfg['project'],
        'invalid':cfg['allow_invalid_gps'],'device':cfg['device'],'precision':cfg['precision'],'batch_size':cfg['batch_size']}
    for k,v in values.items():st.session_state['draft_'+k]=v

def close_settings():st.session_state['settings_open']=False

@st.dialog('Настройки',width='medium',icon=':material/settings:',on_dismiss=close_settings)
def settings_dialog():
    processing,geo,about=st.tabs(['Обработка','Масштаб и GPS','О программе'])
    with processing:
        st.radio('Режим',['Стандартный','С TTA'],key='draft_mode',horizontal=True,help='Четыре отражённых варианта. Медленнее; прирост точности не гарантирован.')
        st.caption('Пороги и площадь настраиваются ползунками под фотографией; карта вероятностей не пересчитывается нейросетью.')
        st.selectbox('Вычисления',['auto','cpu','cuda'],key='draft_device',format_func=lambda x:{'auto':'Автоматически','cpu':'Процессор','cuda':'NVIDIA CUDA'}[x])
        st.selectbox('Точность вычислений',['auto','fp32','fp16'],key='draft_precision',help='Авто: GTX 1060 использует FP32, так как FP16 на GP106 заметно медленнее. На подходящих CUDA GPU включается FP16.')
        st.selectbox('Окон в батче',['auto',1,2,4],key='draft_batch_size',help='Авто: на GTX 1060 замер показал минимум времени при одном окне. Батч 4 доступен принудительно. При нехватке VRAM размер снижается до 2 или 1.')
    with geo:
        scale=st.selectbox('Масштаб',['Не задан','GSD','Высота DJI'],key='draft_scale')
        if scale=='GSD':st.number_input('GSD, см/px',min_value=.01,max_value=1000.,step=.1,key='draft_gsd',help='Один масштаб для всех снимков запуска.')
        if scale=='Высота DJI':st.checkbox('Старт на уровне берега, поверхность плоская',key='draft_flat',help='Для оценки по RelativeAltitude нужны эти условия, надир ±2° и фокус EXIF.')
        st.checkbox('Приблизительные координаты объектов',key='draft_project',disabled=scale=='Не задан',help='Без масштаба, надира и yaw используется GPS кадра.')
        st.checkbox('Разрешить GPS с отметкой Invalid',key='draft_invalid',help='Только для отдельно проверенных координат; статус остаётся в экспорте.')
    with about:
        st.write('Чистый берег · V25')
        st.caption('После гистерезиса: уверенное ядро ≥15% площади и ≥15 пикселей. Цветовой отсев сравнивает LAB с внешним кольцом 10 px, исключая другие кандидаты. Порог delta_ab по умолчанию 6; настройка на боковой панели, 0 — отключение. Яркие белые участки защищены. Это эвристика, возможны пропуски прозрачного пластика.')
        st.caption('SegFormer MiT-B2 · binary-max. По умолчанию: 0,75 / 0,51, площадь 120–10000 px, минимум 12 сильных пикселей. Окно 768 px, перекрытие 192 px, батч до 4. CUDA FP16 доступен; на GTX 1060 автоматически FP32.')
        st.caption('Прежние метрики 86,02% / 70,30% / 63,09% относятся к 0,60 / 0,45 / 45 px и одному сильному пикселю. Новый режим требует отдельной оценки.')
        st.caption('М² — оценка по масштабу, сумма по кадрам без вычитания перекрытий. Точки без проекции — GPS кадра. Правки не переобучают модель.')
    left,right=st.columns(2)
    if left.button('Сохранить',type='primary',width='stretch',key='save_settings'):
        s={'mode':'pixels'}
        if scale=='GSD':s={'mode':'gsd','gsd_cm':st.session_state['draft_gsd']}
        elif scale=='Высота DJI':s={'mode':'altitude','confirm_flat':st.session_state['draft_flat']}
        st.session_state['app_settings']={**applied(),**dict(tta=st.session_state['draft_mode']=='С TTA',
            scale=s,project=bool(st.session_state['draft_project'] and s['mode']!='pixels'),
            allow_invalid_gps=st.session_state['draft_invalid'],device=st.session_state['draft_device'],precision=st.session_state['draft_precision'],batch_size=st.session_state['draft_batch_size'])}
        save_local('.preferences.json',st.session_state['app_settings']);close_settings();st.rerun()
    if right.button('Отмена',width='stretch',key='cancel_settings'):close_settings();st.rerun()
