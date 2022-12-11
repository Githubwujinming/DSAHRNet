import numpy as np
from .cd_base_cam import CDBaseCAM as BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd',
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform, branch=branch)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))


class HiResCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd',
                 reshape_transform=None):
        super(
            HiResCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform, branch=branch)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = grads * activations

        if eigen_smooth:
            print(
                "Warning: HiResCAM's faithfulness guarantees do not hold if smoothing is applied")
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam


class GradCAMElementWise(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd',
                 reshape_transform=None):
        super(
            GradCAMElementWise,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform, branch=branch)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = np.maximum(grads * activations, 0)

        if eigen_smooth:
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam

class XGradCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False, branch='cd',
            reshape_transform=None):
        super(
            XGradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform, branch=branch)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights



class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd', 
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform, branch=branch)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


class LayerCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False, branch='cd',
            reshape_transform=None):
        super(
            LayerCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform, branch=branch)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        spatial_weighted_activations = np.maximum(grads, 0) * activations

        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd',
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       branch=branch,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)


class EigenGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd',
                 reshape_transform=None):
        super(EigenGradCAM, self).__init__(model, target_layers, use_cuda,
                                           reshape_transform, branch=branch)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(grads * activations)


class RandomCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, branch='cd',
                 reshape_transform=None):
        super(
            RandomCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform, branch=branch)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.random.uniform(-1, 1, size=(grads.shape[0], grads.shape[1]))


