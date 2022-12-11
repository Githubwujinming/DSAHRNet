# ----------------------------------------
# Written by Chen Pan
# ----------------------------------------

from FCCDN.FCCDN import *


model_dict = {
		"FCS": FCS,
		"DED": DED,
		"FCCDN": FCCDN,
}

class Config():
    def __init__(self):
        self.MODEL_NAME = 'FCCDN'
        self.MODEL_OUTPUT_STRIDE = 16
        self.BAND_NUM = 3
        self.USE_SE = True

defalt_cfg = Config()
def GenerateNet(cfg=defalt_cfg):
    return model_dict[cfg.MODEL_NAME](
            os = cfg.MODEL_OUTPUT_STRIDE,
            num_band = cfg.BAND_NUM,
            use_se = cfg.USE_SE,
        )
