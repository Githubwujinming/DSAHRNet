from PIL import Image
import cv2
from models.DAHRN.CDNet import CDNet
from util.util import load_by_path
from grad_cam import *
def grad_cam_cd(model, target_layer, saved_path,img1_path, img2_path,gradcam='GradCAM',use_cuda=True):
    # model = BASECDNet('hrnet18')
    model = model.eval()
    model = load_by_path(model, saved_path)
    model = CDModelOutputWrapper(model)
    # target_layer = [model.model.sca]
    cam = CDGradCamUtil(model,target_layer,gradcam=gradcam,use_cuda=use_cuda)
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    cam(img1, img2)
    cam.cd_save_cam()

def grad_cam_seg(model, target_layer,saved_path,img1_path, img2_path,gradcam='GradCAM', use_cuda=True):
    model = model.eval()
    model = load_by_path(model, saved_path)
    model = CDModelOutputWrapper(model)
    cam = CDGradCamUtil(model,target_layer,gradcam=gradcam,use_cuda=use_cuda,branch='seg')
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    cam(img1, img2)
    cam.cd_save_cam()

if __name__ == '__main__':
    img1_path = "/data/LEVIR-CD_cropped256/test/A/test_5_0.png"
    img2_path = "/data/LEVIR-CD_cropped256/test/B/test_5_0.png"
    model_path = "CUBECDN_178_F1_1_0.92102_net_F.pth"
    model = CDNet('hrnet18')
    target_layer = [model.cbuilder.cdblock0]
    gradcam = 'GradCAM'
    grad_cam_cd(model,target_layer,model_path,img1_path,img2_path,gradcam=gradcam)
    