import torch
import itertools
from .base_model import BaseModel
import torch.nn.functional as F
from ChangeFormer.losses import hybrid_loss
from .FCN.FCSModel import getFCSModel
# baseline0 
class FCSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.arch = opt.arch
        self.istest = opt.istest
        if opt.phase == 'test':
            self.istest = True
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['f']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['A', 'B', 'L', 'pred_L_show']  # visualizations for A and B
        if opt.phase == 'val':
            self.visual_names.append('comp')
        if self.istest:
            self.visual_names = ['A', 'B', 'pred_L_show']
        self.visual_features = ['feat_A', 'feat_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['F']
        else:  # during test time, only load Gs
            self.model_names = ['F']
        self.ds=1
        # define networks (both Generators and discriminators)
        self.n_class = 2# for bce loss 
        # self.netF = backbone.define_F(in_c=3, f_c=opt.f_c, type=opt.arch).to(self.device)
        self.netF = getFCSModel(in_c=3, net=opt.arch).to(self.device)
        if self.isTrain:
            # define loss functions
            self.criterionF = hybrid_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netF.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.A = input['A'].to(self.device)
        self.B = input['B'].to(self.device)
        if not self.istest:
            self.L = input['L'].to(self.device).long()
        self.image_paths = input['A_paths']
        if self.isTrain:
            self.L_s = self.L.float()
            self.L_s = F.interpolate(self.L_s, size=torch.Size([self.A.shape[2]//self.ds, self.A.shape[3]//self.ds]),mode='nearest')
            # self.L_s[self.L_s == 1] = -1  # change
            # self.L_s[self.L_s == 0] = 1  # no change

    def test(self, val=False):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            pred = self.pred_L.long()
            if val:  # score
                self.compare()
                from util.metrics import RunningMetrics
                metrics = RunningMetrics(self.n_class)

                metrics.update(self.L.detach().cpu().numpy(), pred.detach().cpu().numpy())
                scores = metrics.get_cm()
                return scores
            else:
                return pred


    def forward(self):
        if self.arch=='FCEF':
            input_data = torch.cat((self.A, self.B),1)
            self.feature, self.dist = self.netF(input_data)
        else:
            self.feature, self.dist = self.netF(self.A, self.B)
        
        self.dist = F.interpolate(self.dist, size=self.A.shape[2:], mode='bilinear',align_corners=True)
        self.pred_L = torch.argmax(self.dist, dim=1, keepdim=True)
        self.pred_L_show = self.pred_L.long()
        return self.pred_L

    def backward(self):
        """Calculate the loss for generators F and L"""
        # print(self.weight)
        self.loss_f = self.criterionF(self.dist, self.L_s)

        self.loss = self.loss_f
        if torch.isnan(self.loss):
           print(self.image_paths)

        self.loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute feat and dist

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
