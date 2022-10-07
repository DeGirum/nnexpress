import nnexpress.interpreter as n2xi
import numpy as np
import time
from quant_utils import quantize, dequantize
'''
This file contains utilities to 
1. run an n2x file 
2. run the sw n2x file on SW and hw n2x file on SW and compare results, and
3. estimate expected frames-per-second (FPS) performance of an n2x file
'''
def run_n2x(n2x_path):
    try:
        interpreter = n2xi.Interpreter(model_path=n2x_path)
        interpreter.allocate_tensors()

        # Get input details
        input_details = interpreter.get_input_details()
        for input_number in range(len(input_details)):
            input_quant_scale = input_details[input_number]['quantization_parameters']['scales'][0]
            input_quant_offset = input_details[input_number]['quantization_parameters']['zero_points'][0]

            input_img_array = np.random.randint(0,255,input_details[input_number]['shape'])
            quantized_input=quantize(input_img_array,input_quant_scale,input_quant_offset)
            interpreter.set_tensor(input_details[input_number]['index'], \
                                    quantized_input.astype(input_details[input_number]['dtype']))
        interpreter.invoke()
        return 'success'
    except Exception as e:
        return str(e)


def compare_n2x_sw_hw(n2x_sw_path,n2x_hw_path):
    match=True
    #Initilaize HW and SW interpreters
    sw_interpreter = n2xi.Interpreter(model_path=n2x_sw_path)
    hw_interpreter = n2xi.Interpreter(model_path=n2x_hw_path)
    
    #Allocate Tensors
    sw_interpreter.allocate_tensors()
    hw_interpreter.allocate_tensors()

    # Get input and output tensors.
    sw_input_details = sw_interpreter.get_input_details()
    sw_output_details = sw_interpreter.get_output_details()
    hw_input_details = hw_interpreter.get_input_details()
    hw_output_details = hw_interpreter.get_output_details()
    
    #Set each input to some random array
    for input_number in range(len(sw_input_details)):
        sw_input_quant_scale = sw_input_details[input_number]['quantization_parameters']['scales'][0]
        sw_input_quant_offset = sw_input_details[input_number]['quantization_parameters']['zero_points'][0]
        hw_input_quant_scale = hw_input_details[input_number]['quantization_parameters']['scales'][0]
        hw_input_quant_offset = hw_input_details[input_number]['quantization_parameters']['zero_points'][0]

        input_img_array = np.random.randint(0,255,sw_input_details[input_number]['shape'])
        sw_quantized_input=quantize(input_img_array,sw_input_quant_scale,sw_input_quant_offset)
        sw_interpreter.set_tensor(sw_input_details[input_number]['index'], \
                                  sw_quantized_input.astype(sw_input_details[input_number]['dtype']))
        hw_quantized_input=quantize(input_img_array,hw_input_quant_scale,hw_input_quant_offset)
        hw_interpreter.set_tensor(hw_input_details[input_number]['index'], \
                                  hw_quantized_input.astype(hw_input_details[input_number]['dtype']))
    #Run HW and SW interpreters
    sw_interpreter.invoke()
    hw_interpreter.invoke()
    
    #Compare all sw outputs with hw outputs
    for output_number in range(len(sw_output_details)):
        sw_output_quant_scale = sw_output_details[output_number]['quantization_parameters']['scales'][0]
        sw_output_quant_offset = sw_output_details[output_number]['quantization_parameters']['zero_points'][0]

        hw_output_quant_scale = hw_output_details[output_number]['quantization_parameters']['scales'][0]
        hw_output_quant_offset = hw_output_details[output_number]['quantization_parameters']['zero_points'][0]

        sw_output_data = sw_interpreter.get_tensor(sw_output_details[output_number]['index'])
        sw_output_float=dequantize(sw_output_data,sw_output_quant_scale,sw_output_quant_offset)

        hw_output_data = hw_interpreter.get_tensor(hw_output_details[output_number]['index'])
        hw_output_float=dequantize(hw_output_data,hw_output_quant_scale,hw_output_quant_offset)
        if not np.allclose(sw_output_data,hw_output_data,1e-3,1e-5):
            match=False
    return match

def estimate_fps(n2x_path,num_iterations=100):
    
    interpreter = n2xi.Interpreter(model_path=n2x_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for input_number in range(len(input_details)):
        input_quant_scale = input_details[input_number]['quantization_parameters']['scales'][0]
        input_quant_offset = input_details[input_number]['quantization_parameters']['zero_points'][0]

        input_img_array = np.random.randint(0,255,input_details[input_number]['shape'])
        quantized_input=quantize(input_img_array,input_quant_scale,input_quant_offset)
        interpreter.set_tensor(input_details[input_number]['index'], \
                                  quantized_input.astype(input_details[input_number]['dtype']))


    start_time=time.time_ns()
    for i in range(num_iterations):
        interpreter.invoke()
    end_time=time.time_ns()
    time_elpased=(end_time-start_time)*1e-9
    fps=num_iterations/time_elpased
    return fps