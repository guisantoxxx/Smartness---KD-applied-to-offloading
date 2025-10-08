import os
import json
from mmdeploy_runtime import Detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np

# ----------------------------
# Caminhos
# ----------------------------
onnx_model = '/proj/aurora/Smartness/checkpoints-GFLV1/onnx_temp'
model_cfg = '/proj/aurora/Smartness/checkpoints-GFLV1/crosskd_r50_gflv1_r101-2x-ms_fpn_1x_coco.py'
deploy_cfg = "mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"

coco_img_dir = '/proj/aurora/Smartness/data/coco/val2017'
coco_ann_file = '/proj/aurora/Smartness/data/coco/annotations/instances_val2017.json'
out_json = '/proj/aurora/Smartness/custom_test/GFLV1-12_epochs-results.json'

# ----------------------------
# Inicializar modelo ONNX via MMDeploy Runtime
# ----------------------------
# Forçar CPU
model = Detector(model_path=onnx_model, device_name='cuda')

# ----------------------------
# Carregar COCO
# ----------------------------
coco = COCO(coco_ann_file)
cat_ids_sorted = sorted(coco.getCatIds())

results = []

# ----------------------------
# Inferência
# ----------------------------
img_ids = coco.getImgIds()[:100]  # apenas 100 imagens para teste
for img_info in coco.loadImgs(img_ids):
    img_path = os.path.join(coco_img_dir, img_info['file_name'])

    # Carregar imagem como RGB
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Executar inferência
    dets = model(img_rgb)
    bboxes, labels, _ = dets  # ignora masks

    # Iterar sobre cada detecção
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = bbox
        if score < 0.05:
            continue
        results.append({
            "image_id": img_info['id'],
            "category_id": cat_ids_sorted[label],
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score)
        })

# ----------------------------
# Salvar resultados
# ----------------------------
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(results, f)

# ----------------------------
# Avaliação COCO
# ----------------------------
coco_dt = coco.loadRes(out_json)
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
