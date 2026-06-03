# Estado del TFM — Verificación Documental

## Repositorio
- GitHub: https://github.com/RaulHernandoViadero/validacion_prestamos
- Rama activa: `dev`
- Python: Anaconda `C:\Users\raulh\anaconda3\python.exe`
- Ejecutar tests: `C:\Users\raulh\anaconda3\python.exe -m pytest tests/ -v`

## ✅ PROYECTO COMPLETADO

### FASE 1 — Datos sintéticos ✅
### FASE 2 — Modelos ✅ (entrenados)
### FASE 3 — Pipeline inferencia ✅
### FASE 4 — API REST FastAPI ✅
### FASE 5 — Frontend Streamlit ✅
### FASE 6 — Docker + Tests + README ✅

## Modelos entrenados (en weights/)
- `yolo_dni.pt`      — mAP50=99.5%, mAP50-95=98.9%, precision=99.9%, recall=100%
- `yolo_loan.pt`     — mAP50=99.5%, mAP50-95=95.5%, precision=99.7%, recall=100%
- `authenticity_classifier.pth` — AUC-ROC=93.6%, accuracy=82%, precision=100%

## Tests: 77/77 pasando

## Sistema funcionando
- API: http://localhost:8000/docs
- Frontend: http://localhost:8501
- docker compose up --build

## Pipeline probado con imagenes reales (Docker)
- YOLO: 9/9 ROIs detectados (conf >93%)
- OCR: 7/9 campos extraidos
- ResNet-18: clasifica LEGITIMO correctamente
- Tiempo primera llamada: ~22s (descarga modelos EasyOCR)
- Tiempo llamadas siguientes: ~6-8s

## Ultimo commit
`feat: modelos entrenados + fix JSON ResNet-18` (rama dev)

## Pendiente menor
- fix text_postprocessor: "334.631mes" -> limpiar "mes" sin / prefijo (hecho, pendiente commit)
