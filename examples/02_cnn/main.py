#%% Import and setup model

## System Imports
import os
import copy

## Other imports
from tqdm import tqdm

## Local Imports
from utils import generate_dataloader

### PyTorch Imports ###
import torch
import torch.nn as nn

### Brevitas Import ###
from brevitas.fx import symbolic_trace
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.calibrate import calibration_mode
from brevitas.export import export_onnx_qop
from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas.export import export_torch_qop
from brevitas.export import export_qonnx

import brevitas.nn as qnn

#%% Setup configuration and data loaders

# DATASETS = os.environ.get('DATASETS')
DATASETS = "/usr/scratch/sassauna1/ml_datasets/"
DTYPE = torch.float
DEVICE = "cpu"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INP_SHAPE = 224
RESIZE_SHAPE = 256

assert DATASETS is not None, 'DATASETS environment variable not set'

calib_loader = generate_dataloader(
    os.path.join(DATASETS, "ILSVRC2012/val"),
    batch_size=64,
    num_workers=8,
    resize_shape=RESIZE_SHAPE,
    center_crop_shape=INP_SHAPE,
    subset_size=100)

val_loader = generate_dataloader(
    dir=os.path.join(DATASETS, "ILSVRC2012/val"),
    batch_size=64,
    num_workers=8,
    resize_shape=RESIZE_SHAPE,
    center_crop_shape=INP_SHAPE,
    subset_size=10)

ref_input = torch.ones(1, 3, INP_SHAPE, INP_SHAPE, device=DEVICE, dtype=DTYPE)

#%% Get the model from torchvision
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
model = model.to(DTYPE)
model = model.to(DEVICE)

dtype = next(model.parameters()).dtype
device = next(model.parameters()).device

print(model)
print("Device: ", device)
print("Dtype: ", dtype)

#%% Prepare for quantization
import torch.nn.functional as F
from brevitas.nn import QuantEltwiseAdd
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph import TorchFunctionalToModule
import operator

model.eval()
model.to(device)
model = preprocess_for_quantize(model, equalize_iters=20, equalize_scale_computation='range')

# FN_TO_MODULE_MAP = ((torch.add, qnn.QuantEltwiseAdd), (operator.add, qnn.QuantEltwiseAdd), )
# model = TorchFunctionalToModule(fn_to_module_map=FN_TO_MODULE_MAP).apply(model)

model = AdaptiveAvgPoolToAvgPool().apply(model, ref_input)

print(model)
print(model.graph.print_tabular())

#%% Quantize model activation
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

COMPUTE_LAYER_MAP = {
    nn.AvgPool2d: (qnn.TruncAvgPool2d, {
        'return_quant_tensor': True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            # 'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'output_quant': Int8ActPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True,
            # 'input_bit_width': 8,
            'output_bit_width': 8,
            'weight_bit_width': 8}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            # 'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'output_quant': Int8ActPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True,
            # 'input_bit_width': 8,
            'output_bit_width': 8,
            'weight_bit_width': 8}),}

QUANT_ACT_MAP = {
    nn.ReLU: (
        qnn.QuantReLU,
        {
            # 'input_quant': Int8ActPerTensorFloat,
            # 'input_bit_width': 8,
            'act_quant': Uint8ActPerTensorFloat,
            'return_quant_tensor': True,
            'bit_width': 7}),}

QUANT_IDENTITY_MAP = {
    'signed': (
        qnn.QuantIdentity, {
            'act_quant': Int8ActPerTensorFloat, 'return_quant_tensor': True, 'bit_width': 7}),
    'unsigned': (
        qnn.QuantIdentity,
        {
            'act_quant': Uint8ActPerTensorFloat,
            'return_quant_tensor': True,  'bit_width': 7
        }),}

model_quant = quantize(
    copy.deepcopy(model),
    compute_layer_map=COMPUTE_LAYER_MAP,
    quant_act_map=QUANT_ACT_MAP,
    quant_identity_map=QUANT_IDENTITY_MAP)

print(model_quant)
print(model_quant.graph.print_tabular())

# %% Calibrate model
model_quant.eval()
model_quant.to(device)

with torch.no_grad():
    with calibration_mode(model_quant):
        for i, (images, target) in enumerate(tqdm(calib_loader)):
            images = images.to(device)
            images = images.to(dtype)
            model_quant(images)

# %% Export model
model_quant.eval()
model_quant.to(device)

# %% Export QCDQ model
# export_onnx_qcdq(model_quant, args=ref_input, export_path="02_quant_model_qcdq.onnx")
# export_torch_qcdq(model_quant, args=ref_input, export_path="02_quant_model_qcdq.pt")

# %% Export QOP model
export_onnx_qop(model_quant, args=ref_input, export_path="02_quant_model_qop.onnx")
# export_torch_qop(model_quant, input_t=ref_input, export_path="02_quant_model_qop.pt")

# %% Export QONNX model
# export_qonnx(model_quant, args=ref_input, export_path="02_quant_model_qonnx.onnx")
