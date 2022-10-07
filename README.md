# nnexpress

## Export
This guide explains how to export a trained YOLOv5 and Yolov7 models from PyTorch to tflite and quantized tflite.
We assume that the models are trained and stored by the Yolov5 repository [YOLOv5](https://github.com/ultralytics/yolov5) or the Yolov7 repository [YOLOv7](https://github.com/WongKinYiu/yolov7.git)

### Yolov5 tflite export

`export.py` is used to export pytorch .pt and .pth to tflite format.

```bash
python export.py --include tflite --weights PATH_TO_PT_FILE
```
In order to export to quantized tflite, add --int8 and --data with the path to representative dataset for quantization. --max-int8-img-cnt is used for limiting the number of images used for quantization.

```bash
python export.py --include tflite --weights PATH_TO_PT_FILE --int8 --data PATH_TO_REP_DATASET --max-int8-img-cnt NUM_IMG_USED_FOR_QUANT
```

### Yolov7 tflite export

`export_v7.py` is used to export pytorch .pt and .pth to tflite format.

```bash
python export_v7.py --tflite --weights PATH_TO_PT_FILE
```
In order to export to quantized tflite, add --int8 and --data with the path to representative dataset for quantization. --max-int8-img-cnt is used for limiting the number of images used for quantization.

```bash
python export_v7.py --tflite --weights PATH_TO_PT_FILE --int8 --data PATH_TO_REP_DATASET  --max-int8-img-cnt NUM_IMG_USED_FOR_QUANT
```
