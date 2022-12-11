
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from .cd_warpper import *
from .grad_cams import *
from util.crop_img import check_dir
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

sem_classes = ['unchange','change']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
change_category = sem_class_to_idx["change"]

GRADCAM = {
    'GradCAM':GradCAM,
    'HiResCAM':HiResCAM,
    'GradCAMElementWise':GradCAMElementWise,
    'GradCAMPlusPlus':GradCAMPlusPlus,
    'XGradCAM':XGradCAM,
    # 'AblationCAM':AblationCAM,
    # 'ScoreCAM':ScoreCAM,
    'EigenCAM':EigenCAM,
    'EigenGradCAM':EigenGradCAM,
    'LayerCAM':LayerCAM,
}

class CDGradCamUtil:
    def __init__(self, model, target_layers,gradcam='GradCAM',
                use_cuda=False, output_dir='grad_imgs',branch='cd') -> None:
        self.use_cuda=use_cuda
        self.branch = branch
        if use_cuda:
            model = model.cuda()
        self.model = model
        self.target_layers = target_layers 
        self.gradcam = gradcam
        self.output_dir = output_dir
        check_dir(output_dir)
        
    def forward(self,img1, img2):
        self.rgb_img1 = np.array(img1, dtype=np.float32)/255
        self.rgb_img2 = np.array(img2, dtype=np.float32)/255
        img1_tensor = preprocess_image(self.rgb_img1,mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        img2_tensor = preprocess_image(self.rgb_img2,mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if self.use_cuda:
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
        output = self.model((img1_tensor,img2_tensor))
        normalized_masks = F.softmax(output, dim=1).cpu()
        change_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        change_mask_float = np.float32(change_mask == change_category)
        change_mask_uint8 = 255*np.uint8(change_mask == change_category)
        self.pred_img = Image.fromarray(change_mask_uint8).resize(img1_tensor.shape[2:])
        targets = [CDSegmentationTarget(change_category, change_mask_float, self.use_cuda)]
        
        with GRADCAM[self.gradcam](model=self.model,
             target_layers=self.target_layers,
             use_cuda=self.use_cuda,branch=self.branch) as cam:
            self.grayscale_cams = cam(input_tensor=[img1_tensor, img2_tensor],
                                targets=targets,eigen_smooth=True)
    def cd_save_cam(self,):
        self.pred_img.save(os.path.join(self.output_dir,'cd_pred.jpg'))
        if self.branch == 'cd':
            self.grayscale_cams = self.grayscale_cams[0,:]
            heatmap = cv2.applyColorMap(np.uint8(255 * self.grayscale_cams),  cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.output_dir,'cd_%s_heatmap.jpg'%self.gradcam),heatmap)
            cam_image1 = show_cam_on_image(self.rgb_img1, self.grayscale_cams, use_rgb=True)
            cam_image2 = show_cam_on_image(self.rgb_img2, self.grayscale_cams, use_rgb=True)
            Image.fromarray(cam_image1).save(os.path.join(self.output_dir,'cd_%s_img1.jpg'%self.gradcam))
            Image.fromarray(cam_image2).save(os.path.join(self.output_dir,'cd_%s_img2.jpg'%self.gradcam))
        else:
            for i in range(len(self.grayscale_cams)):
                grayscale_cam = self.grayscale_cams[i][0,:]
                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam),  cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(self.output_dir,'seg_%s_heatmap%d.jpg'%(self.gradcam,i+1)),heatmap)
                rgb_img = getattr(self,'rgb_img%d'%(i+1))
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                Image.fromarray(cam_image).save(os.path.join(self.output_dir,'seg_%s_img%d.jpg'%(self.gradcam,i+1)))


    def __call__(self, img1, img2):
        return self.forward(img1,img2)

# if __name__ == '__main__':
#     from PIL import Image
#     from models.DTHRCDN.CDNet import BASECDNet
#     model = BASECDNet('hrnet')
#     saved_path = 'checkpoints/SYSU_rf_basecdn_hrnet18_c1_hybrid_SCAT_0913/CUBECDN_test_60_F1_1_0.83629_net_F.pth'
#     target_layer = [model.cbuilder.cddecoder1]
#     cam = CDGradCamUtil(model,target_layer,saved_path)
#     img1 = Image.open('checkpoints/SYSU_rf_basecdn_hrnet18_c1_hybrid_SCAT_0913/val_latest/images/12160_A.png')
#     img2 = Image.open('checkpoints/SYSU_rf_basecdn_hrnet18_c1_hybrid_SCAT_0913/val_latest/images/12160_B.png')
#     rgb_img1 = np.float32(img1) / 255
#     rgb_img2 = np.float32(img2) / 255
#     grad_gray = cam(rgb_img1, rgb_img2)