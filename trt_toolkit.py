import pycuda.autoinit
import pycuda.driver as cuda

import tensorrt as trt
from tensorrt import Builder

def create_trt_engine(trt_logger, onnx_file, engine_file, batch_size=1, fp16_mode=False, save_engine=False):
    # create tools
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    builder = Builder(trt_logger)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, trt_logger)
    
    # setup builder
    builder.max_workspace_size = 1 << 20
    builder.max_batch_size = batch_size
    builder.fp16_mode = fp16_mode
    
    # parse onnx
    '''
    with open(onnx_file, 'rb') as onnx_model:
        parser.parse(onnx_model.read())
    '''
    parser.parse_from_file(onnx_file)
        
    # build trt engine
    engine = builder.build_cuda_engine(network)
    
    if save_engine:
        # save trt engine
        with open(engine_file, 'wb') as trt_model:
            trt_model.write(engine.serialize())
        
    return engine

def load_trt_engine(trt_logger, engine_file):
    if os.path.exists(engine_file):
        with open(engine_file, 'rb') as trt_model:
            runtime = trt.Runtime(trt_logger)
            
            return runtime.deserialize_cuda_engine(trt_model.read())

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mem means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
        
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    output_shapes = []
    
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers.
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_shapes.append(engine.get_binding_shape(binding))
            
    return inputs, outputs, bindings, stream, output_shapes

def post_process(h_outputs, shape_of_outputs):
    h_outputs = h_outputs.reshape(*shape_of_outputs)
    
    return h_outputs

def infer(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs.
    return [out.host for out in outputs]
    