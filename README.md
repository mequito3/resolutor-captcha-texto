## Resolutor de CAPTCHAs Hexadecimales

Proyecto sencillo para (1) generar plantillas sintéticas de caracteres hexadecimales (0-9 a-f) usando fuentes del sistema (Arial) y (2) realizar OCR por lote sobre captchas que contienen 5 caracteres hexadecimales, aplicando preprocesamiento morfológico y motores OCR (EasyOCR por defecto, PaddleOCR opcional).

> Uso educativo / experimental. Asegúrate de tener permiso para automatizar el reconocimiento de imágenes en cualquier sistema externo.

---

### Contenido del repositorio

```
gen_templates_arial.py   # Genera plantillas (templates) normalizadas para cada caracter
solve_captcha.py         # OCR por lotes sobre imágenes de captchas en carpeta imgs/
imgs/                    # Coloca aquí los captchas a procesar (*.png)
templates/               # Se generan (o ya existen) subcarpetas por caracter con PNGs sintéticos
resultados.txt           # (Se crea al ejecutar solve_captcha.py) Resultados acumulados
```

### 1. Requisitos

Python 3.10+ recomendado.

Dependencias mínimas:

- numpy
- opencv-python
- pillow
- easyocr (motor por defecto)

Dependencias opcionales:

- paddleocr (si se quiere usar ENGINE=paddle)

Instalación rápida (CMD en Windows):

```
pip install numpy opencv-python pillow easyocr
pip install paddleocr  # opcional
```

### 2. Generar plantillas sintéticas (opcional)

El script `gen_templates_arial.py` crea un conjunto amplio de variantes (tamaño, ángulo, grosor) para cada caracter en `templates/<caracter>/`.

Revisa / ajusta en el script:

- `FONT_CANDIDATES`: rutas a fuentes. Por defecto rutas estándar de Windows para Arial.
- `CHARSET`: conjunto 0123456789abcdef.
- Parámetros de tamaños, ángulos y grosor.

Ejecutar:

```
python gen_templates_arial.py
```

Salida esperada:

```
[OK] Plantillas generadas: XXXX en 'templates'
```

Si obtienes `FileNotFoundError` ajusta las rutas de las fuentes.

### 3. Preparar imágenes de captcha

Coloca todos los archivos PNG en la carpeta `imgs/` (puedes cambiar la carpeta al ejecutar el script de OCR). Cada imagen debe contener un código de 5 caracteres hexadecimales sobre fondo claro.

### 4. Ejecución del OCR por lote

Script principal: `solve_captcha.py`.

Parámetros internos importantes (puedes editarlos en el archivo):

- `THRESH_VALUE`: umbral binarización (default 250)
- `ERODE_ITER`, `DILATE_ITER`: iteraciones morfológicas para limpiar ruido
- `ANGLE_SWEEP`: lista de ángulos a probar (por defecto [0])
- `WORKERS`: procesos paralelos (por defecto CPU_COUNT-1 o variable de entorno WORKERS)


Ejecutar sobre carpeta `imgs` (por defecto):

```
python solve_captcha.py
```

O especificar carpeta distinta:

```
python solve_captcha.py ruta\a\carpeta_captchas
```

Mientras corre mostrará líneas tipo:

```
[12/200] captcha_b0_20250727_050337_142_000003.png -> a3f9d  (conf=0.842, ang=0, 37 ms)
```

También (append) se va creando / ampliando `resultados.txt` con formato TSV:

```
nombre_imagen	texto	confianza	motor	angulo	ms
captcha_...png	abc12	0.845	easy	0	37
```

Al finalizar imprime resumen:

```
[OK] N imágenes en X.Xs. Resultados -> resultados.txt
```

### 5. Cómo funciona (resumen técnico)

1. Carga y binariza cada imagen (inversión para obtener texto blanco sobre fondo negro temporalmente).
2. Operaciones morfológicas (erode/dilate) para consolidar trazos.
3. Selección de componentes conectados principales para eliminar ruido y recorte (crop) ajustado.
4. (Opcional) barrido de ángulos (`ANGLE_SWEEP`).
5. OCR con EasyOCR o PaddleOCR limitando `allowlist` a caracteres hexadecimales.
6. Filtrado / normalización: se toman solo caracteres en `0123456789abcdef`, se trunca a 5 y se calcula un score promedio de confianza.
7. Selección del mejor resultado según (longitud==5, confianza).

Si falla la inicialización, el script hace fallback automático a EasyOCR.


### 6. Aviso ético

Eludir medidas de seguridad ajenas puede violar términos de servicio o leyes locales. Usa este código sólo en tus propios sistemas o con autorización explícita.

---

### 7. Ejemplo rápido (todo en uno)

```
REM Instalar dependencias
pip install numpy opencv-python pillow easyocr

REM instalar paddle
pip install paddleocr

REM Generar plantillas 
python gen_templates_arial.py

REM Ejecutar OCR (carpeta por defecto imgs/)
set ENGINE=easy
python solve_captcha.py
```

---

