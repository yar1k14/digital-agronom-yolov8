# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import threading
import queue
from model import TomatoClassifier
from auth import register_user, login_user
from datetime import datetime

st.set_page_config(
    page_title="Сортировка помидоров",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
#  СТИЛИ
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;900&family=Roboto:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }

  /* ── AUTH страница ── */
  .auth-wrapper {
    max-width: 480px;
    margin: 0 auto;
    padding: 0 8px;
  }
  .auth-logo {
    text-align: center;
    font-family: 'Montserrat', sans-serif;
    font-weight: 900;
    font-size: 2.8rem;
    color: #c0392b;
    margin-bottom: 4px;
    letter-spacing: -1px;
  }
  .auth-sub {
    text-align: center;
    color: #7f8c8d;
    font-size: 0.95rem;
    margin-bottom: 32px;
  }
  .auth-card {
    background: white;
    border-radius: 20px;
    padding: 36px 40px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.10);
  }
  .auth-tab-active {
    background: #c0392b !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
  }
  .auth-divider {
    text-align: center;
    color: #bdc3c7;
    margin: 16px 0;
    font-size: 0.85rem;
  }

  /* ── Основное приложение ── */
  .main-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 900;
    font-size: 2.2rem;
    color: #c0392b;
    letter-spacing: -1px;
    margin: 0;
  }
  .sub-title { font-weight:300; color:#7f8c8d; font-size:.9rem; margin:0 0 6px 0; }

  .user-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #fdf2f2;
    border: 1px solid #f5c6c6;
    border-radius: 24px;
    padding: 6px 14px;
    font-size: .85rem;
    color: #922b21;
    font-weight: 600;
  }

  .ripeness-card { border-radius:16px; padding:16px 18px; text-align:center;
                   box-shadow:0 4px 20px rgba(0,0,0,0.12); margin:6px 0; }
  .ripe    { background:linear-gradient(135deg,#e74c3c,#c0392b); color:white; }
  .unripe  { background:linear-gradient(135deg,#27ae60,#1e8449); color:white; }
  .partial { background:linear-gradient(135deg,#f39c12,#d68910); color:white; }
  .ripeness-card h2 { font-size:2.4rem; font-weight:700; margin:4px 0; }
  .ripeness-card p  { font-size:.88rem; margin:2px 0 0 0; opacity:.9; }

  .progress-bar-wrap { background:#ecf0f1; border-radius:10px; height:18px;
                       overflow:hidden; margin:4px 0; }
  .progress-bar-fill { height:100%; border-radius:10px; transition:width .4s ease; }

  .cam-status { font-size:.78rem; padding:4px 10px; border-radius:20px;
                display:inline-block; font-weight:600; }
  .cam-on  { background:#d5f5e3; color:#1e8449; }
  .cam-off { background:#fadbd8; color:#922b21; }

  .det-badge { display:inline-block; border-radius:8px; padding:2px 8px;
               font-size:.76rem; font-weight:700; color:white; margin:1px; }
  .det-ripe    { background:#e74c3c; }
  .det-unripe  { background:#27ae60; }
  .det-partial { background:#f39c12; }

  .log-entry { font-size:.78rem; padding:3px 8px; border-left:3px solid #e74c3c;
               margin:2px 0; background:#fdfefe; border-radius:0 6px 6px 0; }

  /* Скрыть стандартные элементы */
  #MainMenu { visibility:hidden; }
  footer    { visibility:hidden; }

  /* Поля ввода */
  .stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e0e0e0 !important;
    padding: 10px 14px !important;
    font-size: .95rem !important;
    transition: border .2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: #c0392b !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,.12) !important;
  }
  .stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all .2s !important;
  }
  hr { border: 1px solid #f0f0f0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        # Auth
        "authenticated": False,
        "current_user": None,
        "auth_tab": "login",       # "login" | "register"
        # App
        "running": False,
        "frame_queue": queue.Queue(maxsize=2),
        "stats": {"зрелый": 0, "незрелый": 0, "частично зрелый": 0},
        "log": [],
        "last_result": None,
        "source_type": "webcam",
        "ip_url": "http://192.168.1.100:8080/video",
        "confidence_threshold": 0.40,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════
#  СТРАНИЦА АВТОРИЗАЦИИ
# ═══════════════════════════════════════════════════════════
def show_auth_page():
    # Центрируем содержимое
    _, center, _ = st.columns([1, 2.4, 1])
    with center:
        st.markdown('<div class="auth-logo">🍅 Цифровой агроном</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Система анализа зрелости</div>',
                    unsafe_allow_html=True)

        # Табы: Вход / Регистрация
        tab_login, tab_reg = st.tabs(["🔑  Войти", "📝  Регистрация"])

        # ── Вкладка ВХОД ──────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)

            username = st.text_input("Логин", placeholder="Введите логин",
                                     key="li_user")
            password = st.text_input("Пароль", placeholder="Введите пароль",
                                     type="password", key="li_pass")

            col_btn, col_hint = st.columns([1.4, 1])
            with col_btn:
                login_btn = st.button("Войти", type="primary",
                                      use_container_width=True, key="btn_login")
            with col_hint:
                st.markdown("<br>", unsafe_allow_html=True)

            if login_btn:
                if not username or not password:
                    st.error("Заполните все поля")
                else:
                    ok, msg, user_info = login_user(username, password)
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.current_user  = user_info
                        st.success(f"Добро пожаловать, {user_info['full_name']}!")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(f"❌  {msg}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center;color:#aaa;font-size:.82rem'>"
                "Нет аккаунта? Перейдите на вкладку «Регистрация»</div>",
                unsafe_allow_html=True)

        # ── Вкладка РЕГИСТРАЦИЯ ────────────────────────────────
        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)

            full_name = st.text_input("Имя (отображается в профиле)",
                                      placeholder="Иван Иванов", key="rg_name")
            r_user    = st.text_input("Логин",
                                      placeholder="Только латиница, цифры, _",
                                      key="rg_user")
            r_pass    = st.text_input("Пароль (минимум 6 символов)",
                                      placeholder="Придумайте надёжный пароль",
                                      type="password", key="rg_pass")
            r_confirm = st.text_input("Подтвердите пароль",
                                      placeholder="Повторите пароль",
                                      type="password", key="rg_confirm")

            st.markdown("<br>", unsafe_allow_html=True)
            reg_btn = st.button("Зарегистрироваться", type="primary",
                                use_container_width=True, key="btn_reg")

            if reg_btn:
                ok, msg = register_user(r_user, r_pass, r_confirm, full_name)
                if ok:
                    st.success(f"✅  {msg} Теперь войдите в систему.")
                else:
                    st.error(f"❌  {msg}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center;color:#aaa;font-size:.82rem'>"
                "Уже есть аккаунт? Перейдите на вкладку «Войти»</div>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    c = TomatoClassifier()
    c.load_or_build()
    return c


CLASS_CSS = {"зрелый": "ripe", "незрелый": "unripe", "частично зрелый": "partial"}
DET_CSS   = {"зрелый": "det-ripe", "незрелый": "det-unripe", "частично зрелый": "det-partial"}
EMOJI_MAP = {"зрелый": "🍅", "незрелый": "🟢", "частично зрелый": "🟡"}
BAR_COLOR = {"зрелый": "#e74c3c", "незрелый": "#27ae60", "частично зрелый": "#f39c12"}


def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def stats_html(stats):
    total = sum(stats.values()) or 1
    out = ""
    for cls in ["зрелый", "незрелый", "частично зрелый"]:
        count = stats[cls]; pct = count / total * 100
        out += f"""
        <div style="margin-bottom:10px">
          <b>{EMOJI_MAP[cls]} {cls.capitalize()}</b>
          <span style="float:right;color:#7f8c8d">{count} ({pct:.0f}%)</span>
          <div class="progress-bar-wrap" style="clear:both">
            <div class="progress-bar-fill"
                 style="width:{pct:.1f}%;background:{BAR_COLOR[cls]}"></div>
          </div>
        </div>"""
    return out

def result_card_html(label, conf):
    css = CLASS_CSS.get(label, "partial"); em = EMOJI_MAP.get(label, "")
    return f"""<div class="ripeness-card {css}">
      <h2>{em}</h2><h2>{conf:.1%}</h2><p><b>{label.upper()}</b></p>
    </div>"""

def probs_html(class_probs):
    out = "<b style='font-size:.9rem'>Вероятности по классам</b>"
    for cls, prob in sorted(class_probs.items(), key=lambda x: -x[1]):
        out += f"<div style='font-size:.82rem;margin-top:6px'><b>{cls}</b> — {prob:.1%}</div>"
        out += f"""<div class="progress-bar-wrap">
          <div class="progress-bar-fill"
               style="width:{prob*100:.1f}%;background:{BAR_COLOR[cls]}"></div></div>"""
    return out

def detections_html(detections):
    if not detections:
        return "<small style='color:#999'>Помидоры не обнаружены</small>"
    out = f"<b>Обнаружено: {len(detections)} плодов</b><br style='margin:4px'>"
    for lb, cf, _, _ in detections:
        css = DET_CSS.get(lb, "det-partial")
        out += f'<span class="det-badge {css}">{lb} {cf:.0%}</span>'
    return out


def capture_loop(source, fq, stop_event):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.release(); time.sleep(1)
            cap = cv2.VideoCapture(source); continue
        if fq.full():
            try: fq.get_nowait()
            except: pass
        fq.put(frame)
        time.sleep(0.04)
    cap.release()


def show_main_app():
    classifier = load_model()
    user = st.session_state.current_user or {}

    # ── Боковая панель ──────────────────────────────────────
    with st.sidebar:
        # Профиль пользователя
        st.markdown(
            f'<div class="user-chip">👤 {user.get("full_name", "Пользователь")}</div>',
            unsafe_allow_html=True)
        st.markdown(f"<small style='color:#aaa'>@{user.get('username','')}</small>",
                    unsafe_allow_html=True)

        if user.get("last_login"):
            st.markdown(
                f"<small style='color:#bbb'>Последний вход: {user['last_login']}</small>",
                unsafe_allow_html=True)

        if st.button("🚪 Выйти", use_container_width=True):
            # Останавливаем камеру если работает
            if st.session_state.running and hasattr(st.session_state, "_stop_event"):
                st.session_state._stop_event.set()
            st.session_state.authenticated = False
            st.session_state.current_user  = None
            st.session_state.running       = False
            st.session_state.log           = []
            st.session_state.stats         = {"зрелый": 0, "незрелый": 0, "частично зрелый": 0}
            st.rerun()

        st.markdown("---")
        st.markdown("## ⚙️ Настройки")

        source_opt = st.radio("Источник видео",
                              ["Веб-камера", "IP-камера / RTSP", "Загрузить изображение"])
        st.session_state.source_type = {
            "Веб-камера": "webcam",
            "IP-камера / RTSP": "ip",
            "Загрузить изображение": "image"
        }[source_opt]

        cam_index = 0
        if st.session_state.source_type == "webcam":
            cam_index = st.number_input("Индекс камеры", min_value=0, max_value=10, value=0)
        elif st.session_state.source_type == "ip":
            st.session_state.ip_url = st.text_input(
                "URL камеры", value=st.session_state.ip_url,
                help="http://192.168.x.x:8080/video  или  rtsp://user:pass@ip:554/stream")

        st.markdown("---")
        st.session_state.confidence_threshold = st.slider(
            "Порог уверенности", 0.20, 0.95,
            st.session_state.confidence_threshold, 0.05)
        show_overlay = st.checkbox("Аннотации на кадре", value=True)
        show_log     = st.checkbox("Лог детекций", value=True)

        st.markdown("---")
        if st.button("🗑️ Сбросить статистику"):
            st.session_state.stats = {"зрелый": 0, "незрелый": 0, "частично зрелый": 0}
            st.session_state.log   = []
            st.rerun()

    # ── Шапка ──────────────────────────────────────────────
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown('<p class="main-title">🍅 Система сортировки помидоров</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Детекция и определение степени зрелости в реальном времени</p>',
                    unsafe_allow_html=True)
    with hcol2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="user-chip" style="float:right">👤 {user.get("full_name","")}</div>',
            unsafe_allow_html=True)
    st.markdown("---")

    # ── Режим: изображение ──────────────────────────────────
    if st.session_state.source_type == "image":
        uploaded = st.file_uploader(
            "Загрузите изображение помидора",
            type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            img   = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            ann, dets, lb, cf, probs = classifier.predict_with_detection(frame)

            c1, c2 = st.columns([2.2, 1])
            with c1:
                st.image(frame_to_pil(ann if show_overlay else frame),
                         caption=f"Обнаружено помидоров: {len(dets)}",
                         use_container_width=True)
            with c2:
                st.markdown(result_card_html(lb, cf), unsafe_allow_html=True)
                st.markdown(detections_html(dets), unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(probs_html(probs), unsafe_allow_html=True)
        return  # stop here for image mode

    # ── Режим: видео ────────────────────────────────────────
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    with cc1:
        start_btn = st.button("▶ Запустить", type="primary",
                              use_container_width=True,
                              disabled=st.session_state.running)
    with cc2:
        stop_btn = st.button("⏹ Остановить", use_container_width=True,
                             disabled=not st.session_state.running)
    with cc3:
        if st.session_state.running:
            st.markdown('<span class="cam-status cam-on">● КАМЕРА АКТИВНА</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="cam-status cam-off">● КАМЕРА ОСТАНОВЛЕНА</span>',
                        unsafe_allow_html=True)

    if start_btn and not st.session_state.running:
        src = cam_index if st.session_state.source_type == "webcam" \
              else st.session_state.ip_url
        st.session_state.frame_queue = queue.Queue(maxsize=2)
        st.session_state._stop_event = threading.Event()
        threading.Thread(target=capture_loop,
                         args=(src, st.session_state.frame_queue,
                               st.session_state._stop_event),
                         daemon=True).start()
        st.session_state.running = True
        st.rerun()

    if stop_btn and st.session_state.running:
        if hasattr(st.session_state, "_stop_event"):
            st.session_state._stop_event.set()
        st.session_state.running = False
        st.rerun()

    main_col, stats_col = st.columns([3, 1])
    with main_col:
        frame_ph = st.empty()
        if not st.session_state.running:
            frame_ph.info("📷 Нажмите «Запустить» для начала анализа")
    with stats_col:
        st.markdown("### 📊 Статистика")
        stat_ph   = st.empty()
        result_ph = st.empty()
        det_ph    = st.empty()
    log_ph = st.empty()

    if st.session_state.running:
        last_refresh = 0.0
        while st.session_state.running:
            try:
                frame = st.session_state.frame_queue.get(timeout=2.0)
            except queue.Empty:
                frame_ph.warning("⚠️ Нет сигнала. Ожидание...")
                time.sleep(0.5); continue

            ann, dets, lb, cf, probs = classifier.predict_with_detection(frame)

            if cf >= st.session_state.confidence_threshold:
                if lb in st.session_state.stats:
                    st.session_state.stats[lb] += 1
                ts = datetime.now().strftime("%H:%M:%S")
                entry = f"{ts}  {EMOJI_MAP.get(lb,'')} {lb}  ({cf:.1%})  [{len(dets)} плодов]"
                st.session_state.log.insert(0, entry)
                if len(st.session_state.log) > 60:
                    st.session_state.log.pop()
                st.session_state.last_result = (lb, cf, probs, dets)

            now = time.time()
            if now - last_refresh >= 0.12:
                frame_ph.image(frame_to_pil(ann if show_overlay else frame),
                               caption="Прямой эфир",
                               use_container_width=True, channels="RGB")
                stat_ph.markdown(stats_html(st.session_state.stats),
                                 unsafe_allow_html=True)
                if st.session_state.last_result:
                    _lb, _cf, _pr, _dets = st.session_state.last_result
                    result_ph.markdown(result_card_html(_lb, _cf), unsafe_allow_html=True)
                    det_ph.markdown(detections_html(_dets), unsafe_allow_html=True)
                if show_log and st.session_state.log:
                    entries = "".join(
                        f'<div class="log-entry">{e}</div>'
                        for e in st.session_state.log[:15])
                    log_ph.markdown(
                        f"<details open><summary><b>📋 Лог детекций</b></summary>{entries}</details>",
                        unsafe_allow_html=True)
                last_refresh = now


# ═══════════════════════════════════════════════════════════
#  РОУТЕР
# ═══════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    show_auth_page()
else:
    show_main_app()
