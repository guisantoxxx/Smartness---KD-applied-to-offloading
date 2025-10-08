import os
import cv2
import json
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ----------------------------
# Utilitários
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def compute_entropy(probs, axis=-1):
    """Calcula entropia de Shannon para distribuição de probabilidades."""
    probs = np.clip(probs, 1e-10, 1.0)  # evita log(0)
    return -np.sum(probs * np.log(probs), axis=axis)

def nms_numpy(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# ----------------------------
# Pré-processamento
# ----------------------------
def preprocess_image(img_path, input_shape=(640, 640)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = min(input_shape[0] / h, input_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    new_img = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
    new_img[:nh, :nw, :] = resized

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    new_img = new_img.astype(np.float32)
    new_img = (new_img - mean) / std

    tensor = new_img.transpose(2, 0, 1)[None, :, :, :]
    return tensor, scale, (w, h)

# ----------------------------
# Decodificadores heurísticos
# ----------------------------
def decode_from_final_detections_array(arr, scale, coco_cat_map, img_id, score_thr=0.05):
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    results = []
    for det in arr:
        if det.size < 6:
            continue
        x1, y1, x2, y2, score = det[:5].tolist()
        cls_id = int(det[5])
        if score < score_thr:
            continue
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
        cat_id = coco_cat_map[cls_id] if cls_id < len(coco_cat_map) else coco_cat_map[-1]
        results.append({
            "image_id": img_id,
            "category_id": int(cat_id),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score)
        })
    return results

def decode_grid_bbox_and_cls(bbox_map, cls_map, input_shape, scale, coco_cat_map,
                             img_id, score_thr=0.05, iou_thr=0.5, return_entropy=False):
    _, H, W, _ = bbox_map.shape
    stride = input_shape[0] // H
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)

    tx = bbox_map[0, :, :, 0]
    ty = bbox_map[0, :, :, 1]
    tw = bbox_map[0, :, :, 2]
    th = bbox_map[0, :, :, 3]

    cx = (gx + sigmoid(tx)) * stride
    cy = (gy + sigmoid(ty)) * stride
    w = np.exp(tw) * stride
    h = np.exp(th) * stride

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    cls_probs = softmax(cls_map, axis=-1)[0]
    
    # Calcula entropia média se solicitado
    mean_entropy = None
    if return_entropy:
        entropies = compute_entropy(cls_probs, axis=-1)
        mean_entropy = np.mean(entropies)

    x1_f = x1.reshape(-1)
    y1_f = y1.reshape(-1)
    x2_f = x2.reshape(-1)
    y2_f = y2.reshape(-1)
    cls_probs_f = cls_probs.reshape(-1, cls_probs.shape[-1])

    best_cls = np.argmax(cls_probs_f, axis=1)
    best_scores = cls_probs_f[np.arange(cls_probs_f.shape[0]), best_cls]

    keep_mask = best_scores >= score_thr
    if not np.any(keep_mask):
        return ([], mean_entropy) if return_entropy else []

    x1_f = x1_f[keep_mask]
    y1_f = y1_f[keep_mask]
    x2_f = x2_f[keep_mask]
    y2_f = y2_f[keep_mask]
    best_cls = best_cls[keep_mask]
    best_scores = best_scores[keep_mask]

    boxes = np.stack([x1_f, y1_f, x2_f, y2_f], axis=1)
    final_results = []
    for cls in np.unique(best_cls):
        idxs = np.where(best_cls == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = best_scores[idxs]
        keep = nms_numpy(cls_boxes, cls_scores, iou_thr)
        for k in keep:
            i = idxs[k]
            x1s = float(boxes[k,0] / scale)
            y1s = float(boxes[k,1] / scale)
            x2s = float(boxes[k,2] / scale)
            y2s = float(boxes[k,3] / scale)
            cat_id = coco_cat_map[int(cls)] if int(cls) < len(coco_cat_map) else coco_cat_map[-1]
            final_results.append({
                "image_id": img_id,
                "category_id": int(cat_id),
                "bbox": [x1s, y1s, x2s - x1s, y2s - y1s],
                "score": float(cls_scores[k])
            })
    
    return (final_results, mean_entropy) if return_entropy else final_results

def process_model_outputs(outputs_all, scale, coco_cat_map, img_id, input_shape, 
                         score_thr=0.05, nms_iou=0.5, return_entropy=False):
    """Processa os outputs de um modelo e retorna detecções (e entropia se solicitado)."""
    image_results = []
    mean_entropy = None
    handled = False

    # Strategy A: detecções finais
    for out in outputs_all:
        arr = np.asarray(out)
        if arr.ndim == 2 and arr.shape[1] >= 6:
            image_results.extend(decode_from_final_detections_array(arr, scale, coco_cat_map, img_id, score_thr))
            handled = True
            break
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] >= 6:
            image_results.extend(decode_from_final_detections_array(arr, scale, coco_cat_map, img_id, score_thr))
            handled = True
            break

    # Strategy B: bbox_map + cls_map
    if not handled:
        found_pair = False
        for i, out_i in enumerate(outputs_all):
            for j, out_j in enumerate(outputs_all):
                if i == j:
                    continue
                a = np.asarray(out_i)
                b = np.asarray(out_j)
                if a.ndim == 4 and a.shape[0] == 1 and a.shape[3] == 4 and b.ndim == 4 and b.shape[0] == 1:
                    if a.shape[1] == b.shape[1] and a.shape[2] == b.shape[2]:
                        try:
                            result = decode_grid_bbox_and_cls(a, b, input_shape, scale, coco_cat_map, 
                                                             img_id, score_thr, nms_iou, return_entropy)
                            if return_entropy:
                                image_results.extend(result[0])
                                mean_entropy = result[1]
                            else:
                                image_results.extend(result)
                            found_pair = True
                            handled = True
                            break
                        except Exception:
                            pass
            if found_pair:
                break

    # Strategy C: YOLO-like
    if not handled:
        for out in outputs_all:
            a = np.asarray(out)
            if a.ndim == 4 and a.shape[0] == 1:
                H, W, C = a.shape[1], a.shape[2], a.shape[3]
                if C >= 5 + 1:
                    num_cls = C - 5
                    try:
                        arr = a
                        bbox_map = arr[0,:,:,:4]
                        obj_map = arr[0,:,:,4]
                        cls_map = arr[0,:,:,5:]
                        cls_probs = softmax(cls_map, axis=-1)
                        
                        if return_entropy:
                            entropies = compute_entropy(cls_probs, axis=-1)
                            mean_entropy = np.mean(entropies)
                        
                        best_cls = np.argmax(cls_probs, axis=-1)
                        best_scores = cls_probs[np.arange(H)[:,None], np.arange(W)[None,:], best_cls] * sigmoid(obj_map)
                        
                        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
                        stride = input_shape[0] // H
                        cx = (gx + sigmoid(bbox_map[:,:,0])) * stride
                        cy = (gy + sigmoid(bbox_map[:,:,1])) * stride
                        w = np.exp(bbox_map[:,:,2]) * stride
                        h = np.exp(bbox_map[:,:,3]) * stride
                        x1 = (cx - w/2).reshape(-1)
                        y1 = (cy - h/2).reshape(-1)
                        x2 = (cx + w/2).reshape(-1)
                        y2 = (cy + h/2).reshape(-1)
                        best_cls_f = best_cls.reshape(-1)
                        best_scores_f = best_scores.reshape(-1)
                        keep_mask = best_scores_f >= score_thr
                        if np.any(keep_mask):
                            boxes = np.stack([x1, y1, x2, y2], axis=1)[keep_mask]
                            cls_idxs = best_cls_f[keep_mask]
                            scores = best_scores_f[keep_mask]
                            for cls in np.unique(cls_idxs):
                                cls_inds = np.where(cls_idxs == cls)[0]
                                cls_boxes = boxes[cls_inds]
                                cls_scores = scores[cls_inds]
                                keep = nms_numpy(cls_boxes, cls_scores, nms_iou)
                                for k in keep:
                                    b = cls_boxes[k]
                                    s = float(cls_scores[k])
                                    cat_id = coco_cat_map[int(cls)] if int(cls) < len(coco_cat_map) else coco_cat_map[-1]
                                    image_results.append({
                                        "image_id": img_id,
                                        "category_id": int(cat_id),
                                        "bbox": [float(b[0] / scale), float(b[1] / scale), 
                                                float((b[2]-b[0]) / scale), float((b[3]-b[1]) / scale)],
                                        "score": s
                                    })
                            handled = True
                            break
                    except Exception:
                        pass

    return (image_results, mean_entropy) if return_entropy else image_results

# ----------------------------
# Rodar inferência com CQI
# ----------------------------
def run_inference(student_onnx_path, teacher_onnx_path, coco_img_dir, coco_ann_file, 
                  input_shape=(640, 640), out_json="results.json", score_thr=0.05, 
                  nms_iou=0.5, use_CQI=False, cqi_threshold=5.0):
    """
    Args:
        student_onnx_path: caminho do modelo estudante
        teacher_onnx_path: caminho do modelo teacher (opcional se use_CQI=False)
        use_CQI: se True, usa entropia + CQI para selecionar entre student/teacher
        cqi_threshold: threshold para decisão (entropia + CQI randint)
    """
    coco = COCO(coco_ann_file)
    cat_ids_sorted = sorted(coco.getCatIds())
    coco_cat_map = cat_ids_sorted

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    # Carregar student
    student_session = ort.InferenceSession(student_onnx_path, providers=providers)
    student_input_name = student_session.get_inputs()[0].name

    # Carregar teacher se necessário
    teacher_session = None
    teacher_input_name = None
    if use_CQI:
        if not teacher_onnx_path or not os.path.exists(teacher_onnx_path):
            raise ValueError("use_CQI=True requer um teacher_onnx_path válido")
        teacher_session = ort.InferenceSession(teacher_onnx_path, providers=providers)
        teacher_input_name = teacher_session.get_inputs()[0].name

    results = []
    img_ids = coco.getImgIds()
    total_imgs = len(img_ids)

    for idx, img_id in enumerate(tqdm(img_ids, desc="Inferência"), 1):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(coco_img_dir, img_info['file_name'])

        tensor, scale, (orig_w, orig_h) = preprocess_image(img_path, input_shape)
        onnx_inputs = {student_input_name: tensor.astype(np.float32)}

        # Predição do student
        student_outputs = student_session.run(None, onnx_inputs)
        
        if use_CQI:
            # Processa student com cálculo de entropia
            student_results, student_entropy = process_model_outputs(
                student_outputs, scale, coco_cat_map, img_id, input_shape, 
                score_thr, nms_iou, return_entropy=True
            )
            
            # Gera CQI aleatório (1-15)
            cqi_value = np.random.randint(1, 16)
            
            # Decisão: entropia + CQI > threshold?
            decision_metric = (student_entropy if student_entropy is not None else 0.0) + cqi_value
            
            if decision_metric > cqi_threshold:
                # Usa predição do student
                image_results = student_results
            else:
                # Usa predição do teacher
                teacher_inputs = {teacher_input_name: tensor.astype(np.float32)}
                teacher_outputs = teacher_session.run(None, teacher_inputs)
                image_results = process_model_outputs(
                    teacher_outputs, scale, coco_cat_map, img_id, input_shape, 
                    score_thr, nms_iou, return_entropy=False
                )
        else:
            # Apenas student
            image_results = process_model_outputs(
                student_outputs, scale, coco_cat_map, img_id, input_shape, 
                score_thr, nms_iou, return_entropy=False
            )

        results.extend(image_results)

        if idx % 100 == 0 or idx == total_imgs:
            print(f"Processadas {idx}/{total_imgs} imagens")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"Resultados salvos em {out_json} (total detections: {len(results)})")

    if len(results) == 0:
        print("Sem detecções para avaliar.")
        return

    coco_dt = coco.loadRes(out_json)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    student_onnx = "/proj/aurora/Smartness/checkpoints-GFLV1/epoch_12.onnx"
    teacher_onnx = "/proj/aurora/Smartness/checkpoints-GFLV1/teacher_model.onnx"  # Ajuste o caminho
    coco_img_dir = "/proj/aurora/Smartness/data/coco/val2017"
    coco_ann_file = "/proj/aurora/Smartness/data/coco/annotations/instances_val2017.json"
    out_json = "/proj/aurora/Smartness/custom_test/GFLV1-12_epochs-results.json"

    run_inference(
        student_onnx_path=student_onnx,
        teacher_onnx_path=teacher_onnx,
        coco_img_dir=coco_img_dir,
        coco_ann_file=coco_ann_file,
        input_shape=(640, 640),
        out_json=out_json,
        use_CQI=True,  # Altere para False para usar apenas o student
        cqi_threshold=5.0  # Ajuste conforme necessário
    )