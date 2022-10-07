import tensorflow as tf
import numpy as np

def quantize(data, quant_scale, quant_offset):
    quant_data=data/quant_scale
    quant_data=quant_data+quant_offset
    np.clip(quant_data, 0, 255)
    return quant_data

def dequantize(quant_data, dequant_scale, dequant_offset):
    data=quant_data.astype(np.float32)-dequant_offset.astype(np.float32)
    data=data*dequant_scale
    return data

def representative_dataset(image_size,num_images=100):
    for _ in range(num_images):
        data = np.random.rand(1, image_size[0],image_size[1],image_size[2])        
        yield [data.astype(np.float32)]

def modelQuantizeImages(model, dataset_generator,per_axis_disable=True):
    """
    Keras model post-training quantization procedure
    [in] model - keras model to quantize
    [in] dataset_generator - function to produce dataset
    """
    assert len(model.inputs) == 1, "Model must have single input"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_quantizer = True
    converter._experimental_disable_per_channel=per_axis_disable
    return converter.convert()

    
