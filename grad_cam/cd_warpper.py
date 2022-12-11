import torch

class CDModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(CDModelOutputWrapper, self).__init__()
        self.model = model.eval()
    def forward(self, x):
        return self.model(x[0],x[1])[-1]
        
class CDSegmentationTarget:
    def __init__(self, category, mask, use_cuda):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if use_cuda:
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
