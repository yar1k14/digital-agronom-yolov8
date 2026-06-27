# -*- coding: utf-8 -*-
"""
model.py — Классификатор + детектор зрелости помидоров.

Архитектура:
  - RandomForest на HSV-признаках (обучен на реальных фото + синтетике)
  - Детекция отдельных плодов через HSV-маску + контурный анализ
  - Резервный HSV-анализ если pkl-модель не найдена

Классы:  0 — зрелый  |  1 — незрелый  |  2 — частично зрелый
"""

import os, ssl, pickle, logging
import numpy as np
import cv2

logging.getLogger("tensorflow").setLevel(logging.ERROR)
ssl._create_default_https_context = ssl._create_unverified_context

CLASSES    = ["зрелый", "незрелый", "частично зрелый"]
MODEL_PATH = "tomato_clf.pkl"   # sklearn-модель
TF_PATH    = "tomato_model.h5"  # (опционально, если есть TF)


# ── Извлечение признаков (должно совпадать с train-скриптом) ─────────────────

def _extract_features(patch_bgr: np.ndarray) -> np.ndarray:
    patch = cv2.resize(patch_bgr, (32, 32))
    hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0].astype(float)
    s = hsv[:,:,1].astype(float)
    v = hsv[:,:,2].astype(float)
    red    = (((h < 12) | (h > 155)) & (s > 80) & (v > 80)).sum()
    green  = ((h > 38)  & (h < 88)  & (s > 50)  & (v > 60)).sum()
    orange = ((h > 11)  & (h < 32)  & (s > 80)  & (v > 80)).sum()
    total  = red + green + orange + 1e-6
    hists = []
    for c in range(3):
        hh = cv2.calcHist([hsv], [c], None, [8], [0, 256]).flatten()
        hists.append(hh / (hh.sum() + 1e-6))
    return np.concatenate([
        [red/total, green/total, orange/total,
         h.mean()/179, s.mean()/255, v.mean()/255,
         h.std()/179,  s.std()/255]
    ] + hists)


# ── HSV-fallback без модели ──────────────────────────────────────────────────

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def _color_probs(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0].astype(float)
    s = hsv[:,:,1].astype(float)
    v = hsv[:,:,2].astype(float)
    red    = (((h < 12) | (h > 155)) & (s > 70) & (v > 70)).sum()
    green  = ((h > 38) & (h < 88)   & (s > 40)  & (v > 50)).sum()
    orange = ((h > 11) & (h < 32)   & (s > 70)  & (v > 70)).sum()
    total  = red + green + orange + 1e-6
    return _softmax(np.array([red, green, orange]) / total * 5)


# ── Детекция отдельных помидоров (возвращает список bbox) ───────────────────

def detect_tomatoes(img_bgr: np.ndarray):
    """
    Находит области с красными/оранжевыми/зелёными помидорами.
    Возвращает список (x, y, w, h).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Маска: красный (зрелый)
    m_red1 = cv2.inRange(hsv, np.array([0,  70, 60]), np.array([12,  255, 255]))
    m_red2 = cv2.inRange(hsv, np.array([155,70, 60]), np.array([180, 255, 255]))
    # Маска: оранжевый (частично)
    m_orng = cv2.inRange(hsv, np.array([10, 80, 60]), np.array([30,  255, 255]))
    # Маска: зелёный помидор (насыщенный, не листья — листья менее насыщены)
    m_grn  = cv2.inRange(hsv, np.array([38, 70, 60]), np.array([85,  255, 255]))

    tomato_mask = cv2.bitwise_or(cv2.bitwise_or(m_red1, m_red2),
                                  cv2.bitwise_or(m_orng, m_grn))

    # Морфология для объединения соседних пикселей
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, kernel)
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(tomato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.003:   # слишком маленький
            continue
        if area > img_area * 0.60:    # слишком большой (весь кадр)
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Фильтр по соотношению сторон (помидор ~круглый)
        aspect = w / (h + 1e-6)
        if aspect < 0.35 or aspect > 2.8:
            continue
        boxes.append((x, y, w, h))

    # NMS — убираем сильно перекрывающиеся боксы
    boxes = _nms(boxes, iou_thresh=0.45)
    return boxes


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    iw = min(ax+aw, bx+bw) - ix
    ih = min(ay+ah, by+bh) - iy
    if iw <= 0 or ih <= 0: return 0.0
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return inter / (union + 1e-6)

def _nms(boxes, iou_thresh=0.45):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)  # по убыванию площади
    kept = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]: continue
        kept.append(b)
        for j in range(i+1, len(boxes)):
            if not used[j] and _iou(b, boxes[j]) > iou_thresh:
                used[j] = True
    return kept


# ── Основной класс ───────────────────────────────────────────────────────────

class TomatoClassifier:
    """
    Классификатор зрелости помидоров.
    Основан на RandomForest (sklearn) + HSV-детектор помидоров.
    Не требует TensorFlow и интернета.
    """

    def __init__(self):
        self._clf = None
        self._load_sklearn()

    def _load_sklearn(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self._clf = pickle.load(f)
                return
            except Exception:
                pass

    def load_or_build(self):
        """Совместимость с app.py — модель уже загружена в __init__."""
        pass  # sklearn-модель загружается сразу

    # ── Предсказание для одного патча / bbox ─────────────────────────────────

    def predict_patch(self, patch_bgr: np.ndarray):
        """Возвращает (label, confidence, class_probs_dict) для патча."""
        if self._clf is not None:
            feats = _extract_features(patch_bgr).reshape(1, -1)
            proba = self._clf.predict_proba(feats)[0]
            # Убедимся что порядок классов совпадает [зрелый, незрелый, частично]
            classes_order = list(self._clf.classes_)
            probs = np.zeros(3)
            for i, c in enumerate(classes_order):
                probs[c] = proba[i]
        else:
            probs = _color_probs(patch_bgr)

        idx  = int(np.argmax(probs))
        label = CLASSES[idx]
        conf  = float(probs[idx])
        class_probs = {CLASSES[i]: float(probs[i]) for i in range(3)}
        return label, conf, class_probs

    # ── Предсказание для всего кадра (без детекции) ──────────────────────────

    def predict(self, img_bgr: np.ndarray):
        """Классификация всего изображения (без bbox)."""
        return self.predict_patch(img_bgr)

    # ── Предсказание с детекцией отдельных помидоров ─────────────────────────

    def predict_with_detection(self, img_bgr: np.ndarray):
        """
        Находит помидоры, классифицирует каждый.
        Возвращает:
          - annotated_frame (np.ndarray)  — кадр с нарисованными боксами
          - detections: list[(label, conf, class_probs, (x,y,w,h))]
          - summary_label, summary_conf, summary_probs — итог по кадру
        """
        boxes = detect_tomatoes(img_bgr)
        detections = []
        annotated  = img_bgr.copy()

        COLORS = {
            "зрелый":          (0,  50, 210),   # красный (BGR)
            "незрелый":        (30, 160,  30),   # зелёный
            "частично зрелый": (0, 140, 230),    # оранжевый
        }

        for (x, y, w, h) in boxes:
            patch = img_bgr[y:y+h, x:x+w]
            if patch.size == 0: continue
            label, conf, probs = self.predict_patch(patch)
            detections.append((label, conf, probs, (x, y, w, h)))

            color = COLORS.get(label, (200, 200, 200))

            # Рамка
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Фон для текста
            text = f"{label}  {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.45, min(0.75, w / 220))
            (tw, th), bl = cv2.getTextSize(text, font, scale, 2)
            ty = max(y - 6, th + 4)
            cv2.rectangle(annotated,
                          (x, ty - th - bl - 4),
                          (x + tw + 6, ty + 2),
                          color, -1)
            cv2.putText(annotated, text,
                        (x + 3, ty - bl - 2),
                        font, scale, (255, 255, 255), 2, cv2.LINE_AA)

            # Полоса зрелости под боксом
            bar_y = y + h + 4
            if bar_y + 8 < annotated.shape[0]:
                cv2.rectangle(annotated, (x, bar_y), (x+w, bar_y+7),
                              (220, 220, 220), -1)
                fill = int(x + w * conf)
                cv2.rectangle(annotated, (x, bar_y), (fill, bar_y+7), color, -1)

        # Итоговый класс — взвешенное голосование по всем детекциям
        if detections:
            votes = np.zeros(3)
            for lb, cf, pr, _ in detections:
                votes[CLASSES.index(lb)] += cf
            sum_label = CLASSES[int(np.argmax(votes))]
            sum_conf  = float(np.max(votes) / (votes.sum() + 1e-6))
            sum_probs = {CLASSES[i]: float(votes[i] / (votes.sum() + 1e-6)) for i in range(3)}
        else:
            # Fallback — классификация всего кадра
            sum_label, sum_conf, sum_probs = self.predict(img_bgr)

        return annotated, detections, sum_label, sum_conf, sum_probs
