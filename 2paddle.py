from x2paddle.convert import pytorch2paddle
import torch
import torch.nn.functional as F
from PIL import Image
from zmq import device
from util.util import save_images
from models.DAHRN.CDNet import BASECDNet, CDNet
from util.util import load_by_path
from torchvision.transforms import transforms
def transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return transforms.Compose(transform_list)
model = CDNet('hrnet18')
model = load_by_path(model, 'checkpoints/bisai_dahrn_1008/DAHRN_96_F1_1_0.92546_net_F.pth')
model.eval()
device = 'cuda:0'
model = model.to(device)
image1_path = 'checkpoints/bisai_dahrn_1008/val_133_F1_1_0.92568/val_12_1_A.png'
image2_path = 'checkpoints/bisai_dahrn_1008/val_133_F1_1_0.92568/val_12_1_B.png'
x1 = Image.open(image1_path).convert('RGB')
x2 = Image.open(image2_path).convert('RGB')
trans = transform()
A = trans(x1).unsqueeze(0)
B = trans(x2).unsqueeze(0)
torch_output = model(A.to(device),B.to(device)).cpu()
pred_L = F.interpolate(torch_output, size=A.shape[2:], mode='bilinear',align_corners=True)
pred_L = torch.argmax(pred_L, dim=1, keepdim=True).long()
save_images(pred_L, 'samples/output',['val.png'])
import os
import sys
# from PIL import Image
import numpy as np
import onnxruntime as ort
# # print(ort.get_device())


# ###########3. 先转换为onnx
# jit_type = "trace"    #转换类型
export_onnx_file = "epoch_96.onnx"	#输出文件
# torch.onnx.export(model,
#                     (A.to(device),B.to(device)),
#                     export_onnx_file,
#                     opset_version=11,   #opset_version 9不支持多输出
#                     verbose=False,
#                     do_constant_folding=True,	# 是否执行常量折叠优化
#                     input_names=['x1','x2'],	# 输入名
#                     output_names=["c"],	# 输出名
#                     dynamic_axes={'x1':{0: 'batch'},'x2':{0: 'batch'},'c':{0: 'batch'}}
#                     )

# ort_session = ort.InferenceSession(export_onnx_file, providers=['CPUExecutionProvider'])
# o_outputs = ort_session.run(None, {'x1':A.numpy(),'x2':B.numpy()})

# print('torch VS onnx diff ----', 'max: ', abs(torch_output[0].detach().numpy()-o_outputs[0]).max(), 'min: ', abs((torch_output[0].detach().numpy()-o_outputs[0]).min()))
import paddle
from pd_model.x2paddle_dahrnet import ONNXModel
paddle.disable_static()
params = paddle.load('pd_model/epoch_96.pdparams')
p_model = ONNXModel()
p_model.set_dict(params, use_structured_name=True)
p_model.eval()
p_out = p_model(paddle.to_tensor(A.numpy(), dtype='float32'),paddle.to_tensor(B.numpy(), dtype='float32'))
pred = paddle.argmax(p_out, axis=1)
pred_show = pred[0].cpu().numpy()*255
image_pil = Image.fromarray(pred_show.astype(np.uint8))
image_pil.save('paddle.png')
print('torch VS paddle diff ----', 'max: ', abs(torch_output[0].detach().numpy()-p_out[0].cpu().numpy()).max(), 'min: ', abs((torch_output[0].detach().numpy()-p_out[0].cpu().numpy()).min()))