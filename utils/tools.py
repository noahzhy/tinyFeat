import onnx
import torch
from thop import clever_format, profile


def set_seed(seed=0):
    """
    Set random seed for reproducibility.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model, input_size=(1, 3, 224, 224), cpu=True):
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    macs, params = profile(model, (dummy_input, ), verbose=False)
    #-------------------------------------------------------------------------------#
    #   flops * 2 because profile does not consider convolution as two operations.
    #-------------------------------------------------------------------------------#
    flops = macs * 2
    macs, flops, params = clever_format([macs, flops, params], "%.2f ")
    print(f'\33[30mTotal MACs:   {macs:>8}\33[0m')
    print(f'\33[32mTotal FLOPs:  {flops:>8}\33[0m')
    print(f'\33[34mTotal Params: {params:>8}\33[0m')
    return macs, flops, params


def export2onnx(model, input_size=(1, 3, 224, 224), model_name="mobilenetv4_small.onnx", opt_version=20):
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    x = torch.randn(input_size).to(device)
    torch.onnx.export(model, x,
        model_name,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=opt_version,
    )
    print(f'Model exported to {model_name}')


def onnx2fp16(onnx_model_path, fp16_model_path, keep_io_types=True):
    from onnx import load_model, save_model
    from onnxmltools.utils import float16_converter

    onnx_model = load_model(onnx_model_path)
    trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=keep_io_types)
    save_model(trans_model, fp16_model_path)
    return fp16_model_path


def simplify_onnx(original_model, simplified_model):
    from onnxsim import simplify
    model = onnx.load(original_model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simplified_model)
