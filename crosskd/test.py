from mmdet.apis import inference_detector, init_detector
import mmcv

config = '/workspace/CrossKD/configs/crosskd/crosskd_r50_retinanet_r101_fpn_2x_coco.py'
checkpoint = '/proj/aurora/Smartness/checkpoints-2/epoch_24.pth'
image_path = '/proj/aurora/Smartness/data/coco/test2017/000000043380.jpg'

model = init_detector(config, checkpoint, device='cuda:0')
result = inference_detector(model, image_path)


print(result)