# fast_batch_ocr.py
import os, sys, glob, time, math, cv2, numpy as np
from dataclasses import dataclass
from multiprocessing import Pool, get_start_method, set_start_method
from functools import lru_cache
from typing import List, Tuple, Dict

# ===== Preproceso (tus parámetros) =====
THRESH_VALUE = int(os.environ.get("THRESH_VALUE", 250))  # valor legado (si se fuerza)
MIN_AREA     = int(os.environ.get("MIN_AREA", 12))
KERNEL       = int(os.environ.get("KERNEL", 3))
ERODE_ITER   = int(os.environ.get("ERODE_ITER", 5))
DILATE_ITER  = int(os.environ.get("DILATE_ITER", 5))

# Nuevos parámetros dinámicos
TRY_DYNAMIC      = os.environ.get("DYNAMIC", "1") == "1"   # intenta varios métodos de binarización
MAX_BIN_VARIANTS = int(os.environ.get("BIN_VARIANTS", 6))    # límite de variantes evaluadas
USE_TEMPLATES    = os.environ.get("USE_TEMPLATES", "1") == "1"  # fallback a matching por plantillas
TEMPLATE_DIR     = os.environ.get("TEMPLATE_DIR", "templates")
TEMPLATE_NORM    = int(os.environ.get("TEMPLATE_NORM", 40))
FALLBACK_MIN_LEN = int(os.environ.get("FALLBACK_MIN_LEN", 4))  # si OCR < este largo, probar plantillas
EARLY_STOP_CONF  = float(os.environ.get("EARLY_STOP_CONF", 0.92))  # si ya tenemos buena conf, paramos
RESCALE_HEIGHT   = int(os.environ.get("RESCALE_HEIGHT", 120))       # reescala cada crop a esta altura antes de OCR (mantiene aspecto)
UNION_VARIANTS   = os.environ.get("UNION_VARIANTS", "1") == "1"   # agrega un crop usando el bounding box union de variantes
TARGET_LEN       = int(os.environ.get("TARGET_LEN", 5))             # longitud esperada del captcha (usa 5 por defecto)
AUTO_MORPH       = os.environ.get("AUTO_MORPH", "1") == "1"       # explorar iteraciones morfológicas ligeras por variante
MORPH_MAX        = int(os.environ.get("MORPH_MAX", 2))              # máximo extra de iteraciones a explorar (0..MORPH_MAX)
KEEP_VARIANTS    = int(os.environ.get("KEEP_VARIANTS", 8))          # límite de variantes finales tras expandir morfología
SPLIT_WIDE       = os.environ.get("SPLIT_WIDE", "1") == "1"       # intentar dividir componentes anchos (caracteres pegados)
WIDE_WIDTH_FACTOR= float(os.environ.get("WIDE_WIDTH_FACTOR", 1.8))  # umbral (ancho > factor * alto) para considerar corte
VALLEY_REL_THRESH= float(os.environ.get("VALLEY_REL_THRESH", 0.35)) # fracción de pico para considerar valle de corte
MIN_SEG_WIDTH    = int(os.environ.get("MIN_SEG_WIDTH", 3))          # ancho mínimo segmento tras corte
MAX_SPLITS_PER_COMP = int(os.environ.get("MAX_SPLITS_PER_COMP", 4)) # máximo de sub-segmentos creados de un componente
PRE_MEDIAN       = os.environ.get("PRE_MEDIAN", "1") == "1"       # aplicar blur mediano antes de binarizar
MEDIAN_K         = int(os.environ.get("MEDIAN_K", 3))
PRE_BILATERAL    = os.environ.get("PRE_BILATERAL", "0") == "1"    # filtro bilateral (más lento)
BILATERAL_D      = int(os.environ.get("BILATERAL_D", 5))
BILATERAL_SC     = int(os.environ.get("BILATERAL_SC", 75))
BILATERAL_SS     = int(os.environ.get("BILATERAL_SS", 75))
PRE_CONTRAST     = os.environ.get("PRE_CONTRAST", "1") == "1"     # aplicar CLAHE previo a variantes
USE_SAUVOLA      = os.environ.get("USE_SAUVOLA", "1") == "1"      # añadir variante Sauvola
SAUVOLA_W        = int(os.environ.get("SAUVOLA_W", 25))             # ventana Sauvola (impar)
SAUVOLA_K        = float(os.environ.get("SAUVOLA_K", 0.2))          # k de Sauvola
SAUVOLA_R        = float(os.environ.get("SAUVOLA_R", 128.0))        # R de Sauvola (normalmente 128/255)
NOISE_OPEN_ITER  = int(os.environ.get("NOISE_OPEN_ITER", 1))        # apertura ligera para quitar puntos
MIN_DENSITY      = float(os.environ.get("MIN_DENSITY", 0.05))       # descartar variantes demasiado vacías
MAX_DENSITY      = float(os.environ.get("MAX_DENSITY", 0.65))       # descartar variantes muy llenas
MAX_COMPONENTS   = int(os.environ.get("MAX_COMPONENTS", 40))        # descartar si excede número de componentes

HEX = "0123456789abcdef"

# ===== Config =====
ENGINE      = os.environ.get("ENGINE", "easy").lower()  # "easy" | "paddle"
# Ángulos base; se expanden dinámicamente si la confianza es baja
ANGLE_SWEEP = [0]
EXTRA_ANGLES = [-8,-5,-3,3,5,8]
WORKERS     = int(os.environ.get("WORKERS", max(1, os.cpu_count()-1)))
WRITE_DEBUG = os.environ.get("WRITE_DEBUG", "0") == "1"   # True = guarda pasos
DEBUG_DIR   = os.environ.get("DEBUG_DIR", "debug")
LOG_LEVEL   = os.environ.get("LOG_LEVEL", "info").lower()
CHARSET_ENV = os.environ.get("CHARSET", HEX)

# Evita oversubscription al paralelizar (OpenCV usa hilos internos)
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

@dataclass
class OcrResult:
    name: str
    text: str
    conf: float
    source: str
    angle: int
    ms: float

# === OCR Engines ===
_EASY = None
_PADDLE = None
_HAVE_EASY = False
_HAVE_PADDLE = False

_TEMPLATES: Dict[str, List[np.ndarray]] = {}

def _load_templates():
    """Carga plantillas (una sola vez por worker) en memoria normalizada a TEMPLATE_NORM."""
    if not USE_TEMPLATES:
        return
    if not os.path.isdir(TEMPLATE_DIR):
        return
    for ch in HEX:
        dir_c = os.path.join(TEMPLATE_DIR, ch)
        imgs = []
        if os.path.isdir(dir_c):
            for fn in os.listdir(dir_c):
                if not fn.lower().endswith('.png'): continue
                p = os.path.join(dir_c, fn)
                try:
                    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if im is None: continue
                    if im.shape[0] != TEMPLATE_NORM or im.shape[1] != TEMPLATE_NORM:
                        im = cv2.resize(im, (TEMPLATE_NORM, TEMPLATE_NORM), interpolation=cv2.INTER_AREA)
                    imgs.append(im)
                except:  # noqa
                    pass
        if imgs:
            _TEMPLATES[ch] = imgs

def init_worker(engine: str):
    global _EASY, _PADDLE, _HAVE_EASY, _HAVE_PADDLE, ENGINE
    ENGINE = engine
    if ENGINE == "paddle":
        try:
            from paddleocr import PaddleOCR
            _PADDLE = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                show_log=False,
                enable_mkldnn=True,
                cpu_threads=max(1, os.cpu_count()//2)
            )
            _HAVE_PADDLE = True
        except Exception:
            ENGINE = "easy"
    if ENGINE == "easy":
        try:
            import easyocr
            _EASY = easyocr.Reader(['en'], gpu=False, verbose=False)
            _HAVE_EASY = True
        except Exception as e:
            raise RuntimeError("No se pudo inicializar EasyOCR") from e
    _load_templates()

# === Utilidades ===
def _postfilter_hex(text):
    """Filtra alfabeto permitido y recorta solo si excede 2x TARGET_LEN (protege sobre-segmentación)."""
    allowed = CHARSET_ENV.lower()
    t = "".join([c for c in text.lower() if c in allowed])
    limit = max(TARGET_LEN*2, TARGET_LEN)
    if len(t) > limit:
        return t[:limit]
    return t

def _order_by_x(items):
    # items: [(box, text, conf)] ; box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    return sorted(items, key=lambda it: np.mean([p[0] for p in it[0]]))

def _score(items):
    vals = [float(conf) for (_, txt, conf) in items if _postfilter_hex(txt)]
    return float(np.mean(vals)) if vals else 0.0

def _extract_largest_mask(bin_img, max_components=8):
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    mask = np.zeros_like(bin_img)
    if nb > 1:
        areas = sorted(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb)],
            key=lambda x: x[1], reverse=True)
        for i,_ in areas[:max_components]:
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue
            mask[labels==i] = 255
    return mask

def _auto_variants(gray: np.ndarray) -> List[np.ndarray]:
    variants: List[np.ndarray] = []
    h, w = gray.shape
    base = gray.copy()
    if PRE_MEDIAN:
        try:
            base = cv2.medianBlur(base, MEDIAN_K if MEDIAN_K%2==1 else MEDIAN_K+1)
        except:
            pass
    if PRE_BILATERAL:
        base = cv2.bilateralFilter(base, d=BILATERAL_D, sigmaColor=BILATERAL_SC, sigmaSpace=BILATERAL_SS)
    if PRE_CONTRAST:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        base = clahe.apply(base)

    # 1. OTSU inverso
    _, b1 = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    variants.append(b1)
    # 2. OTSU normal invertido
    _, b2 = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variants.append(255-b2)
    # 3. Adaptativo mean
    b3 = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    variants.append(b3)
    # 4. Adaptativo gaussian
    b4 = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)
    variants.append(b4)
    # 5. Legacy threshold sobre original
    _, b5 = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY_INV)
    variants.append(b5)
    # 6. CLAHE + Otsu (si no se aplicó antes)
    if not PRE_CONTRAST:
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        g2 = clahe2.apply(gray)
        _, b6 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        variants.append(b6)
    # 7. Sauvola opcional
    if USE_SAUVOLA:
        variants.append(_sauvola(base))

    uniq = []
    seen_hash = set()
    for v in variants:
        hsh = int(v.mean())*1000 + v.std()
        if hsh in seen_hash: continue
        seen_hash.add(hsh)
        uniq.append(v)
        if len(uniq) >= MAX_BIN_VARIANTS:
            break
    return uniq

def _sauvola(gray: np.ndarray) -> np.ndarray:
    win = SAUVOLA_W if SAUVOLA_W % 2 == 1 else SAUVOLA_W + 1
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(win,win), normalize=True)
    mean_sq = cv2.boxFilter(g*g, ddepth=-1, ksize=(win,win), normalize=True)
    std = cv2.sqrt(cv2.max(mean_sq - mean*mean, 0))
    k = SAUVOLA_K; R = SAUVOLA_R
    thresh = mean * (1 + k * ((std / R) - 1))
    bin_inv = (g < thresh).astype(np.uint8) * 255
    return bin_inv

def _log(msg, level="info"):
    lv_rank = {"debug":0, "info":1, "warn":2, "error":3}
    if lv_rank.get(level,1) >= lv_rank.get(LOG_LEVEL,1):
        print(f"[{level.upper()}] {msg}")

def _refine(bin_img: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL, KERNEL))
    er = cv2.erode(bin_img, k, iterations=min(ERODE_ITER,3))
    di = cv2.dilate(er, k, iterations=min(DILATE_ITER,3))
    if NOISE_OPEN_ITER>0:
        # apertura ligera extra para quitar puntos sueltos (sobre versión invertida para robustez)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        open_ = cv2.morphologyEx(di, cv2.MORPH_OPEN, k2, iterations=NOISE_OPEN_ITER)
        return open_
    return di

def _refine_custom(bin_img: np.ndarray, e_it: int, d_it: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL, KERNEL))
    er = cv2.erode(bin_img, k, iterations=e_it)
    di = cv2.dilate(er, k, iterations=d_it)
    return di

def _component_stats(bin_img: np.ndarray):
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    areas = []
    widths = []
    heights = []
    for i in range(1, nb):
        x,y,w,h,a = stats[i]
        if a < MIN_AREA: continue
        areas.append(a)
        widths.append(w)
        heights.append(h)
    return len(areas), areas, widths, heights

def _variant_priority(img_bin: np.ndarray) -> tuple:
    comp_count, areas, widths, heights = _component_stats(img_bin)
    # heurística: buscar conteo cerca de TARGET_LEN (o TARGET_LEN +/-1)
    diff = abs(comp_count - TARGET_LEN)
    # densidad de tinta (media) preferida ~0.30-0.40
    density = img_bin.mean()/255.0
    dens_penalty = abs(density - 0.35)
    # varianza de ancho: demasiada varianza sugiere ruido
    var_w = np.var(widths) if widths else 1e6
    # prioridad menor es mejor
    return (diff, dens_penalty, var_w, -img_bin.shape[1])

def _crop_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask>0)
    if len(xs)==0:
        return mask
    x0, x1 = max(xs.min()-4,0), min(xs.max()+4, mask.shape[1]-1)
    y0, y1 = max(ys.min()-4,0), min(ys.max()+4, mask.shape[0]-1)
    return mask[y0:y1+1, x0:x1+1]

def clean_and_crop(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Devuelve varias variantes de recorte binario para intentar OCR.
    Antes retornaba solo una. Ahora una lista ordenada (mejores primero)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if not TRY_DYNAMIC:
        _, bin_inv = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY_INV)
        mask = _extract_largest_mask(_refine(bin_inv))
        if mask.sum()==0:
            mask = bin_inv
        return [_crop_mask(mask)]

    variants = _auto_variants(gray)
    expanded = []
    # Heurística: primero valores base, luego combinaciones
    for v in variants:
        expanded.append(_refine(v))
        if AUTO_MORPH:
            for delta in range(1, MORPH_MAX+1):
                expanded.append(_refine_custom(v, max(1,ERODE_ITER-delta), max(1,DILATE_ITER+delta)))
                expanded.append(_refine_custom(v, max(1,ERODE_ITER+delta), max(1,DILATE_ITER-delta)))
    crops = []
    boxes = []
    for ref in expanded[:100]:  # safety cap
        mask = _extract_largest_mask(ref)
        if mask.sum()==0:
            mask = ref
        crop = _crop_mask(mask)
        ys, xs = np.where(mask>0)
        if len(xs)>0:
            boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
        # filtrar por densidad y componentes
        dens = crop.mean()/255.0
        if dens < MIN_DENSITY or dens > MAX_DENSITY:
            continue
        comp_count, _, _, _ = _component_stats(crop)
        if comp_count > MAX_COMPONENTS:
            continue
        crops.append(crop)
    # Unión de bounding boxes para tomar el "mayor rango" si se solicita
    if UNION_VARIANTS and boxes:
        x0 = min(b[0] for b in boxes)
        x1 = max(b[1] for b in boxes)
        y0 = min(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        # usar la primera variante refinada como base
        base_full = variants[0]
        union_mask = np.zeros_like(base_full)
        union_mask[y0:y1+1, x0:x1+1] = 255
        union_crop = _crop_mask(union_mask)
        # priorizarlo al inicio (más probable capturar todos los chars)
        crops.insert(0, union_crop)
    # Puntuar y quedarnos con las mejores
    scored = [( _variant_priority(c), idx, c) for idx,c in enumerate(crops)]
    scored.sort(key=lambda x: x[0])
    crops = [c for _,_,c in scored[:KEEP_VARIANTS]]
    return crops

def rotate_keep(img_bin, ang):
    if ang == 0: 
        return img_bin
    M = cv2.getRotationMatrix2D((img_bin.shape[1]/2, img_bin.shape[0]/2), ang, 1.0)
    rot = cv2.warpAffine(img_bin, M, (img_bin.shape[1], img_bin.shape[0]),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    return cv2.threshold(rot, 127, 255, cv2.THRESH_BINARY)[1]

def paddle_read(img_bin):
    res = _PADDLE.ocr(img_bin, cls=True)
    items=[]
    for line in res:
        for (box, (txt, conf)) in line:
            items.append((box, txt, float(conf)))
    return items

def easyocr_read(img_bin):
    rgb = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    items = _EASY.readtext(rgb, detail=1, paragraph=False, allowlist=HEX)
    return [(box, txt, float(conf)) for (box, txt, conf) in items]

def _char_match_single(glyph: np.ndarray) -> Tuple[str, float]:
    if not _TEMPLATES:
        return '', 0.0
    # normalizar a TEMPLATE_NORM
    g = glyph
    if g.shape[0] != TEMPLATE_NORM or g.shape[1] != TEMPLATE_NORM:
        g = cv2.resize(g, (TEMPLATE_NORM, TEMPLATE_NORM), interpolation=cv2.INTER_AREA)
    g_f = g.astype(np.float32)
    best_ch, best_score = '', 1e9
    for ch, arrs in _TEMPLATES.items():
        for tmpl in arrs:
            diff = cv2.absdiff(g_f, tmpl.astype(np.float32))
            score = diff.mean()
            if score < best_score:
                best_score = score
                best_ch = ch
    # convertir score a pseudo-conf (inversa normalizada)
    conf = max(0.0, min(1.0, 1.0 - (best_score/255.0)))
    return best_ch, conf

def _template_fallback(bin_img: np.ndarray) -> Tuple[str, float]:
    # segmentar por componentes conectados
    nb, labels, stats, cent = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    comps = []
    for i in range(1, nb):
        x,y,w,h,a = stats[i]
        if a < MIN_AREA: continue
        roi = bin_img[y:y+h, x:x+w]
        # Intentar dividir componentes anchos (posibles 2+ caracteres pegados)
        if SPLIT_WIDE and w > WIDE_WIDTH_FACTOR * h:
            sub_rois = _split_wide_component(roi)
            # Ajustar x incremental según orden
            offset_local = 0
            for sub in sub_rois:
                comps.append((x + offset_local, sub))
                offset_local += sub.shape[1]
        else:
            comps.append((x, roi))
    comps.sort(key=lambda t: t[0])
    text = ''
    confs = []
    for _, roi in comps:
        ch, c = _char_match_single(roi)
        if ch:
            text += ch
            confs.append(c)
    if not text:
        return '', 0.0
    return _postfilter_hex(text), float(np.mean(confs) if confs else 0.0)

def _split_wide_component(roi: np.ndarray) -> List[np.ndarray]:
    """Divide un componente ancho en sub-imágenes usando proyección vertical.
    Retorna lista (>=1)."""
    inv = 255 - roi  # tinta ~ blanco
    proj = inv.sum(axis=0).astype(np.float32)
    if proj.max() <= 0:
        return [roi]
    # suavizar
    k = max(3, min(11, roi.shape[1]//15*2+1))  # kernel impar adaptativo
    kernel = np.ones(k, np.float32)/k
    smooth = np.convolve(proj, kernel, mode='same')
    thresh_valley = VALLEY_REL_THRESH * smooth.max()
    valleys = []
    for i in range(1, len(smooth)-1):
        if smooth[i] < thresh_valley and smooth[i] <= smooth[i-1] and smooth[i] <= smooth[i+1]:
            valleys.append(i)
    # Evitar cortes muy juntos o cerca de bordes
    filtered = []
    last = -999
    min_sep = max(3, roi.shape[1]//TARGET_LEN//2)
    for v in valleys:
        if v < MIN_SEG_WIDTH or v > roi.shape[1]-MIN_SEG_WIDTH:
            continue
        if v - last < min_sep:
            continue
        filtered.append(v)
        last = v
    # Construir segmentos
    cuts = [0] + filtered + [roi.shape[1]]
    segs = []
    for i in range(len(cuts)-1):
        c0, c1 = cuts[i], cuts[i+1]
        if c1 - c0 < MIN_SEG_WIDTH:
            continue
        seg = roi[:, c0:c1]
        # eliminar columnas vacías extremo
        col_sum = (255 - seg).sum(axis=0)
        nz = np.where(col_sum > 0)[0]
        if len(nz)==0:
            continue
        seg = seg[:, nz.min():nz.max()+1]
        segs.append(seg)
        if len(segs) >= MAX_SPLITS_PER_COMP:
            break
    return segs if segs else [roi]

def _auto_target_len(crops: List[np.ndarray]) -> int:
    if TARGET_LEN > 0:
        return TARGET_LEN
    # estimar longitud por mediana de componentes de primeras variantes
    counts=[]
    for c in crops[:3]:
        cnt,_,_,_ = _component_stats(c)
        if cnt>0: counts.append(cnt)
    return int(np.median(counts)) if counts else 5

def _save_debug_variants(base_name: str, original_bgr: np.ndarray, crops: List[np.ndarray]):
    if not WRITE_DEBUG: return
    try:
        out_dir = os.path.join(DEBUG_DIR, os.path.splitext(base_name)[0])
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "orig.png"), original_bgr)
        for i,c in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, f"crop_{i}.png"), c)
    except Exception as e:
        _log(f"debug save error: {e}", "warn")

def run_one(path):
    t0 = time.time()
    name = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return OcrResult(name, "", 0.0, "none", 0, 0.0)

    crops = clean_and_crop(img)
    dyn_target = _auto_target_len(crops)
    best_text, best_conf, best_src, best_ang = "", 0.0, ENGINE, 0
    _save_debug_variants(name, img, crops)

    tried_angles = list(ANGLE_SWEEP)
    for crop_i, crop in enumerate(crops):
        for ang in ANGLE_SWEEP:
            cand = rotate_keep(crop, ang)
            # reescalar a altura estándar (solo para OCR, no para fallback de plantillas que usa binario original)
            if RESCALE_HEIGHT > 0 and cand.shape[0] != RESCALE_HEIGHT:
                new_w = int(round(cand.shape[1] * (RESCALE_HEIGHT / cand.shape[0])))
                cand_rs = cv2.resize(cand, (new_w, RESCALE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            else:
                cand_rs = cand
            items = paddle_read(cand_rs) if (ENGINE=="paddle" and _HAVE_PADDLE) else easyocr_read(cand_rs)
            items = _order_by_x(items)
            text  = _postfilter_hex("".join(txt for (_,txt,_) in items))
            conf  = _score(items)
            prefer = (len(text)==dyn_target, conf)
            best   = (len(best_text)==dyn_target, best_conf)
            if prefer > best:
                best_text, best_conf, best_ang = text, conf, ang
            if len(best_text)==dyn_target and best_conf >= EARLY_STOP_CONF:
                break
        # Si tras primera pasada no logramos buena confianza, expandimos ángulos extra una sola vez
        if crop_i == 0 and len(best_text) < dyn_target and best_conf < EARLY_STOP_CONF:
            for ang in EXTRA_ANGLES:
                if ang in tried_angles: continue
                cand = rotate_keep(crop, ang)
                if RESCALE_HEIGHT > 0 and cand.shape[0] != RESCALE_HEIGHT:
                    new_w = int(round(cand.shape[1] * (RESCALE_HEIGHT / cand.shape[0])))
                    cand_rs = cv2.resize(cand, (new_w, RESCALE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                else:
                    cand_rs = cand
                items = paddle_read(cand_rs) if (ENGINE=="paddle" and _HAVE_PADDLE) else easyocr_read(cand_rs)
                items = _order_by_x(items)
                text  = _postfilter_hex("".join(txt for (_,txt,_) in items))
                conf  = _score(items)
                prefer = (len(text)==dyn_target, conf)
                best   = (len(best_text)==dyn_target, best_conf)
                if prefer > best:
                    best_text, best_conf, best_ang = text, conf, ang
                tried_angles.append(ang)
                if len(best_text)==dyn_target and best_conf >= EARLY_STOP_CONF:
                    break
        if len(best_text)==dyn_target and best_conf >= EARLY_STOP_CONF:
            break

    # Fallback a plantillas si texto muy corto / vacío
    if USE_TEMPLATES and (len(best_text) < FALLBACK_MIN_LEN):
        # usar el primer crop (más probable) para fallback
        fb_text, fb_conf = _template_fallback(crops[0]) if crops else ('',0.0)
        # aceptar fallback solo si mejora longitud o conf
        if len(fb_text) > len(best_text) or (len(fb_text)==len(best_text) and fb_conf>best_conf):
            best_text, best_conf, best_src = fb_text, fb_conf, 'template'

    ms = (time.time()-t0)*1000.0
    return OcrResult(name, best_text, float(best_conf), best_src, int(best_ang), ms)

def main(folder="imgs", pattern="*.png", out_file="resultados.txt"):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        print("No se encontraron imágenes.")
        return

    # En Windows, usar 'spawn'
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    print(f"[INFO] ENGINE={ENGINE} | WORKERS={WORKERS} | ANGLES={ANGLE_SWEEP} | DYNAMIC={TRY_DYNAMIC} | TEMPLATES={USE_TEMPLATES} | TARGET_LEN={TARGET_LEN} | AUTO_MORPH={AUTO_MORPH} | SPLIT_WIDE={SPLIT_WIDE} | CHARSET={CHARSET_ENV}")
    t0 = time.time()
    with Pool(processes=WORKERS, initializer=init_worker, initargs=(ENGINE,)) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, paths, chunksize=32), 1):
            print(f"[{i}/{len(paths)}] {res.name} -> {res.text}  (conf={res.conf:.3f}, ang={res.angle}, {res.ms:.0f} ms)")

            # Guardado incremental (append) — evita perder el progreso
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(f"{res.name}\t{res.text}\t{res.conf:.3f}\t{res.source}\t{res.angle}\t{res.ms:.0f}\n")

    print(f"\n[OK] {len(paths)} imágenes en {time.time()-t0:.1f}s. Resultados -> {out_file}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv)>1 else "imgs"
    main(folder=folder, pattern="*.png", out_file="resultados.txt")
