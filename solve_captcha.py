# fast_batch_ocr.py
import os, sys, glob, time, math, cv2, numpy as np
from dataclasses import dataclass
from multiprocessing import Pool, get_start_method, set_start_method

# ===== Preproceso (tus parámetros) =====
THRESH_VALUE = 250
MIN_AREA     = 12
KERNEL       = 3
ERODE_ITER   = 5
DILATE_ITER  = 5

HEX = "0123456789abcdef"

# ===== Config =====
ENGINE      = os.environ.get("ENGINE", "easy").lower()  # "easy" | "paddle"
ANGLE_SWEEP = [0]   # 0 = sin barrido (más rápido) — si falla mucho, prueba [-5,0,5]
WORKERS     = int(os.environ.get("WORKERS", max(1, os.cpu_count()-1)))
WRITE_DEBUG = False   # True = MUY lento

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

def init_worker(engine: str):
    global _EASY, _PADDLE, _HAVE_EASY, _HAVE_PADDLE, ENGINE
    ENGINE = engine
    if ENGINE == "paddle":
        try:
            from paddleocr import PaddleOCR
            # Optimiza CPU (usa MKLDNN si está disponible)
            _PADDLE = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                show_log=False,
                enable_mkldnn=True,  # si MKL-DNN está presente
                cpu_threads=max(1, os.cpu_count()//2)
            )
            _HAVE_PADDLE = True
        except Exception as e:
            # Fallback a EasyOCR
            ENGINE = "easy"
    if ENGINE == "easy":
        try:
            import easyocr
            _EASY = easyocr.Reader(['en'], gpu=False, verbose=False)
            _HAVE_EASY = True
        except Exception as e:
            raise RuntimeError("No se pudo inicializar EasyOCR") from e

# === Utilidades ===
def _postfilter_hex(text):
    t = "".join([c for c in text.lower() if c in HEX])
    return t[:5]

def _order_by_x(items):
    # items: [(box, text, conf)] ; box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    return sorted(items, key=lambda it: np.mean([p[0] for p in it[0]]))

def _score(items):
    vals = [float(conf) for (_, txt, conf) in items if _postfilter_hex(txt)]
    return float(np.mean(vals)) if vals else 0.0

def clean_and_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL, KERNEL))
    eroded  = cv2.erode(bin_inv, k, iterations=ERODE_ITER)
    dilated = cv2.dilate(eroded, k, iterations=DILATE_ITER)

    nb, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    mask = np.zeros_like(dilated)
    if nb > 1:
        areas = sorted([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb)], key=lambda x: x[1], reverse=True)
        for i,_ in areas[:8]:
            mask[labels==i] = 255
    else:
        mask = dilated

    ys, xs = np.where(mask>0)
    if len(xs)>0:
        x0, x1 = max(xs.min()-4,0), min(xs.max()+4, mask.shape[1]-1)
        y0, y1 = max(ys.min()-4,0), min(ys.max()+4, mask.shape[0]-1)
        crop = mask[y0:y1+1, x0:x1+1]
    else:
        crop = mask
    return crop

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

def run_one(path):
    t0 = time.time()
    name = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return OcrResult(name, "", 0.0, "none", 0, 0.0)

    crop = clean_and_crop(img)

    best_text, best_conf, best_src, best_ang = "", 0.0, ENGINE, 0
    for ang in ANGLE_SWEEP:
        cand = rotate_keep(crop, ang)
        items = paddle_read(cand) if (ENGINE=="paddle" and _HAVE_PADDLE) else easyocr_read(cand)
        items = _order_by_x(items)
        text  = _postfilter_hex("".join(txt for (_,txt,_) in items))
        conf  = _score(items)
        # preferir largo 5; desempatar por conf
        prefer = (len(text)==5, conf)
        best   = (len(best_text)==5, best_conf)
        if prefer > best:
            best_text, best_conf, best_ang = text, conf, ang

    ms = (time.time()-t0)*1000.0
    return OcrResult(name, best_text, float(best_conf), ENGINE, int(best_ang), ms)

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

    print(f"[INFO] ENGINE={ENGINE} | WORKERS={WORKERS} | ANGLES={ANGLE_SWEEP}")
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
