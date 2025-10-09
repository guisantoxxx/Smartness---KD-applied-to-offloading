import os
import json
import time
import random
import cv2
import numpy as np
from mmdeploy_runtime import Detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ----------------------------
# Funções auxiliares
# ----------------------------
def entropy_from_probs(probs):
    """Entropia normalizada (0-1) para distribuições multi-classe."""
    probs = np.clip(probs, 1e-9, 1.0)
    ent = -np.sum(probs * np.log2(probs), axis=1)
    max_ent = np.log2(probs.shape[1])
    return ent / max_ent

def binary_entropy_from_score(scores):
    """Entropia binária a partir dos scores de confiança (0-1)."""
    scores = np.clip(scores, 1e-9, 1 - 1e-9)
    return -scores * np.log2(scores) - (1 - scores) * np.log2(1 - scores)

def calcular_entropia_media(bboxes, extras=None):
    """Calcula a entropia média normalizada com base nas probabilidades ou scores."""
    if bboxes is None or len(bboxes) == 0:
        return 1.0  # entropia alta -> baixa confiança

    probs_available = None

    if extras is not None:
        if isinstance(extras, dict):
            if 'probs' in extras:
                probs_available = np.asarray(extras['probs'], dtype=np.float32)
            elif 'scores' in extras:
                arr = np.asarray(extras['scores'], dtype=np.float32)
                if arr.ndim == 2:
                    probs_available = arr
        elif isinstance(extras, (list, np.ndarray)):
            arr = np.asarray(extras, dtype=np.float32)
            if arr.ndim == 2:
                probs_available = arr

    detection_scores = np.asarray([b[4] for b in bboxes], dtype=np.float32)

    if probs_available is not None and probs_available.size > 0:
        norm_ent_per_det = entropy_from_probs(probs_available)
        avg_norm_entropy = float(np.mean(norm_ent_per_det))
    elif detection_scores.size > 0:
        norm_ent_per_det = binary_entropy_from_score(detection_scores)
        avg_norm_entropy = float(np.mean(norm_ent_per_det))
    else:
        avg_norm_entropy = 1.0

    return avg_norm_entropy


# ----------------------------
# Caminhos
# ----------------------------
student_onnx_model = '/proj/aurora/Smartness/checkpoints-GFLV1/student_onnx'
teacher_onnx_model = '/proj/aurora/Smartness/checkpoints-GFLV1/teacher_onnx'
model_cfg = '/proj/aurora/Smartness/checkpoints-GFLV1/crosskd_r50_gflv1_r101-2x-ms_fpn_1x_coco.py'
deploy_cfg = "mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"

coco_img_dir = '/proj/aurora/Smartness/data/coco/val2017'
coco_ann_file = '/proj/aurora/Smartness/data/coco/annotations/instances_val2017.json'
out_json = '/proj/aurora/Smartness/custom_test/GFLV1-12_epochs-results.json'

# ----------------------------
# Inicializar modelos ONNX
# ----------------------------
student_model = Detector(model_path=student_onnx_model, device_name='cpu')
teacher_model = Detector(model_path=teacher_onnx_model, device_name='cpu')

# ----------------------------
# Carregar COCO
# ----------------------------
coco = COCO(coco_ann_file)
cat_ids_sorted = sorted(coco.getCatIds())
results = []

# ----------------------------
# Thresholds de decisão
# ----------------------------
CQI_THRESHOLD = 10          # exemplo: valores até 10 são aceitáveis
ENTROPY_THRESHOLD = 0.6     # entropia média até 0.6 = boa confiança

# ----------------------------
# Loop de inferência
# ----------------------------
img_ids = coco.getImgIds()
start_time = time.time()

for idx, img_info in enumerate(coco.loadImgs(img_ids), 1):
    img_path = os.path.join(coco_img_dir, img_info['file_name'])
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Predição do estudante
    dets_student = student_model(img_rgb)
    bboxes_s, labels_s, extras_s = dets_student

    # Calcular entropia média
    avg_ent = calcular_entropia_media(bboxes_s, extras_s)
    print("Avg entropy: ", avg_ent)

    # Gerar CQI aleatório entre 1 e 15
    CQI = random.randint(1, 15)

    # Decisão de qual modelo usar
    usar_student = (CQI >= CQI_THRESHOLD) and (avg_ent <= ENTROPY_THRESHOLD)
    if usar_student:
        bboxes, labels = bboxes_s, labels_s
        origem = "STUDENT"
    else:
        dets_teacher = teacher_model(img_rgb)
        bboxes, labels, _ = dets_teacher
        origem = "TEACHER"

    # Registrar predições
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = bbox
        if score < 0.05:
            continue
        results.append({
            "image_id": img_info['id'],
            "category_id": cat_ids_sorted[label],
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
            "source": origem,
            "cqi": CQI,
            "entropy": avg_ent
        })

    # Progresso
    if idx % 10 == 0:
        print(f"Processadas {idx}/{len(img_ids)} imagens... "
              f"CQI={CQI}  Entropia={avg_ent:.3f}  -> {origem}")

# ----------------------------
# Avaliação COCO
# ----------------------------
elapsed_time = time.time() - start_time
time_per_image = elapsed_time / len(img_ids)

os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(results, f)

coco_dt = coco.loadRes(out_json)
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# ----------------------------
# Print estilo MMEngine
# ----------------------------
stats = coco_eval.stats
print(
    f"coco/bbox_mAP: {stats[0]:.4f}  "
    f"coco/bbox_mAP_50: {stats[1]:.4f}  "
    f"coco/bbox_mAP_75: {stats[2]:.4f}  "
    f"coco/bbox_mAP_s: {stats[3]:.4f}  "
    f"coco/bbox_mAP_m: {stats[4]:.4f}  "
    f"coco/bbox_mAP_l: {stats[5]:.4f}  "
    f"data_time: {time_per_image:.4f}  "
    f"time: {time_per_image:.4f}"
)
