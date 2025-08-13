import os, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2

# === ajustes ===
CHARSET     = "0123456789abcdef"                   # minúsculas
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",               # bold
]
OUT_DIR     = "templates"
SIZES       = [38, 42, 46, 50, 54, 58]
ANGLES      = [-10, -6, -3, 0, 3, 6, 10]
THICK_ITERS = [0, 1, 2, 3]                         # variación de grosor
NORM        = 40                                   # tamaño final por plantilla

os.makedirs(OUT_DIR, exist_ok=True)
for c in CHARSET:
    os.makedirs(os.path.join(OUT_DIR, c), exist_ok=True)

def render_char(ch, font_path, size, angle):
    W=160; H=160
    img = Image.new("L", (W,H), 255)
    drw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=size)
    bbox = drw.textbbox((0,0), ch, font=font)
    tw = drw.textlength(ch, font=font); th = bbox[3]-bbox[1]
    x = (W - int(tw))//2; y = (H - th)//2 - bbox[1]
    drw.text((x,y), ch, 0, font=font)
    if angle != 0:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=255)
    # binario texto negro
    img = img.point(lambda p: 0 if p < 128 else 255, '1').convert('L')
    box = ImageOps.invert(img).getbbox()
    if box: img = img.crop(box)
    # cuadrar y normalizar
    pad = max(img.size)
    canvas = Image.new("L", (pad,pad), 255)
    canvas.paste(img, ((pad-img.width)//2, (pad-img.height)//2))
    canvas = canvas.resize((NORM,NORM), Image.BILINEAR)
    return canvas

def thicken(pil_img, iters):
    if iters<=0: return pil_img
    arr = np.array(pil_img)        # 0=negro trazo, 255=fondo
    k = np.ones((3,3), np.uint8)
    arr = 255-arr
    arr = cv2.dilate(arr, k, iterations=iters)
    arr = 255-arr
    return Image.fromarray(arr)

count=0
fonts = [f for f in FONT_CANDIDATES if os.path.exists(f)]
if not fonts:
    raise FileNotFoundError("No encontré Arial. Ajusta FONT_CANDIDATES a la ruta correcta.")

for ch in CHARSET:
    for font_path in fonts:
        base_name = os.path.splitext(os.path.basename(font_path))[0]
        for s in SIZES:
            for a in ANGLES:
                base = render_char(ch, font_path, s, a)
                for t in THICK_ITERS:
                    img = thicken(base, t)
                    img.save(os.path.join(OUT_DIR, ch, f"{ch}_{base_name}_s{s}_a{a}_t{t}.png"))
                    count+=1
print(f"[OK] Plantillas generadas: {count} en '{OUT_DIR}'")
