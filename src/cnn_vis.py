import torch
import torch.nn.functional as F
import torch.optim as optim

import utils, mnist

EXT = 'png'


class StraightForward:

    def __init__(self, model, images):
        self.model = model
        self.model.eval()
        self.images = images

    def visualise_layer(self, selected_layer, save=False, scale=1, pad_value=0):

        if selected_layer > len(self.model):
            raise ValueError('the model has {} layers, but select {}.'.format(len(self.model), selected_layer))

        x = self.images
        if selected_layer == 0:
            # visualise original image
            im_path = None if not save else '../result/sf_original.{}'.format(EXT)
            utils.show_tensor(x, 1, filename=im_path)
        else:
            for index, layer in enumerate(self.model):
                x = layer(x)
                if index + 1 == selected_layer:
                    break
            x = x.permute(1, 0, 2, 3)

            im_path = None if not save else '../result/sf_layer_{}.{}'.format(str(selected_layer), EXT)
            utils.show_tensor(x, 10, filename=im_path, scale=scale, pad_value=pad_value)

    def visualise_filter(self, selected_layer=1, save=False, scale=1, pad_value=0):

        if selected_layer > len(self.model):
            raise ValueError('the model has {} layers, but select {}.'.format(len(self.model), selected_layer))

        # Only support vis first Conv's filter
        layer = self.model[selected_layer-1]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError('layer {} is not a nn.Conv2d.'.format(selected_layer))
        x = layer.weight

        im_path = None if not save else '../result/sf_filter_{}.{}'.format(str(selected_layer), EXT)
        utils.show_tensor(x, 10, filename=im_path,scale=scale, pad_value=pad_value)


class VanillaBackprop:

    def __init__(self, model, images, tags):
        self.model = model
        self.model.eval()
        self.images = images
        self.tags = tags
        self.gradients = None

    def hook_layers(self, selected_layer):
        def hook_function(module, grad_input, grad_output):
            self.gradients = grad_input[0]

        layer = self.model.features._modules[str(selected_layer-1)]
        layer.register_backward_hook(hook_function)

    def visualise(self, selected_layer, save=False, scale=1, pad_value=0):

        if selected_layer > len(self.model.features):
            raise ValueError('the model has {} layers, but select {}.'.format(len(self.model), selected_layer))

        self.hook_layers(selected_layer)

        self.model.zero_grad()
        x = self.images
        model_output = self.model(x)

        one_hot_output = torch.zeros(1, model_output.shape[-1])
        one_hot_output[0][self.tags[0].item()] = 1
        model_output.backward(gradient=one_hot_output)

        # x = F.relu(x)
        x = self.gradients.permute(1, 0, 2, 3)
        x = utils.tensor_norm(x)
        im_path = None if not save else '../result/vb_layer_{}.{}'.format(str(selected_layer), EXT)
        utils.show_tensor(x, 10, filename=im_path, scale=scale, pad_value=pad_value)


class GuidedBackprop:

    def __init__(self, model, images, tags):
        self.model = model
        self.model.eval()
        self.images = images
        self.tags = tags
        self.gradients = None

    def hook_layers(self, selected_layer):
        def hook_function(module, grad_input, grad_output):
            self.gradients = grad_input[0]

        layer = self.model.features._modules[str(selected_layer-1)]
        layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module, grad_input, grad_output):
            # If there is a negative gradient, change it to zero
            modified_grad_input = F.relu(grad_input[0])
            return (modified_grad_input,)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)

    def visualise(self, selected_layer, save=False, scale=1, pad_value=0):

        if selected_layer > len(self.model.features):
            raise ValueError('the model has {} layers, but select {}.'.format(len(self.model), selected_layer))

        self.update_relus()
        self.hook_layers(selected_layer)

        self.model.zero_grad()
        x = self.images
        model_output = self.model(x)

        one_hot_output = torch.zeros(1, model_output.shape[-1])
        one_hot_output[0][self.tags[0].item()] = 1
        model_output.backward(gradient=one_hot_output)

        x = self.gradients.permute(1, 0, 2, 3)
        x = utils.tensor_norm(x)
        im_path = None if not save else '../result/gb_layer_{}.{}'.format(str(selected_layer), EXT)
        utils.show_tensor(x, 10, filename=im_path, scale=scale, pad_value=pad_value)


class GradCAM:

    def __init__(self, model, images, tags):
        self.model = model
        self.model.eval()
        self.images = images
        self.tags = tags
        self.gradients = None
        self.feature_map = None

    def hook_layers(self, selected_layer):

        def forward_hook_function(module, input, output):
            self.feature_map = output

        def backward_hook_function(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        layer = self.model.features._modules[str(selected_layer-1)]
        layer.register_forward_hook(forward_hook_function)
        layer.register_backward_hook(backward_hook_function)

    def visualise(self, selected_layer, save=False, scale=1, pad_value=0):

        if selected_layer > len(self.model.features):
            raise ValueError('the model has {} layers, but select {}.'.format(len(self.model), selected_layer))

        self.hook_layers(selected_layer)

        self.model.zero_grad()
        x = self.images
        model_output = self.model(x)

        one_hot_output = torch.zeros(1, model_output.shape[-1])
        one_hot_output[0][self.tags[0].item()] = 1
        model_output.backward(gradient=one_hot_output)

        weights = torch.sum(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.mul(weights, self.feature_map)
        cam = torch.sum(cam, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = utils.tensor_norm(cam)

        im_path = None if not save else '../result/gcam_layer_{}.{}'.format(str(selected_layer), EXT)
        utils.apply_heatmap_on_image(self.images.detach().numpy(), cam.detach().numpy(), filename=im_path, scale=scale, pad_value=pad_value)


class InvertedRepresentation:
    def __init__(self, model, images):
        self.model = model
        self.model.eval()
        self.images = images

    def alpha_norm(self, input_matrix, alpha):

        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):

        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

    def euclidian_loss(self, org_matrix, target_matrix):

        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    def get_output_from_specific_layer(self, x, selected_layer):

        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index + 1 == selected_layer:
                break
        return x

    def visualise(self, selected_layer, save=False, scale=1, pad_value=0):

        target = self.get_output_from_specific_layer(self.images, selected_layer)

        opt_img = torch.randn(self.images.shape, requires_grad=True)

        optimizer = optim.SGD([opt_img], lr=1e3)

        # Alpha regularization parametrs
        # Parameter alpha, which is actually sixth norm
        alpha_reg_alpha = 6
        # The multiplier, lambda alpha
        alpha_reg_lambda = 1e-7

        # Total variation regularization parameters
        # Parameter beta, which is actually second norm
        tv_reg_beta = 2
        # The multiplier, lambda beta
        tv_reg_lambda = 1e-8

        for i in range(201):
            optimizer.zero_grad()
            # Get the output from the model after a forward pass until target_layer
            # with the generated image (randomly generated one, NOT the real image)
            output = self.get_output_from_specific_layer(opt_img, selected_layer)
            # Calculate euclidian loss
            euc_loss = 1e-1 * self.euclidian_loss(target.detach(), output)
            # Calculate alpha regularization
            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)
            # Calculate total variation regularization
            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img,
                                                                            tv_reg_beta)
            # Sum all to optimize
            loss = euc_loss + reg_alpha + reg_total_variation
            # Step
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy())

            if i % 40 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1/3

        im_path = None if not save else '../result/ir_layer_{}.{}'.format(str(selected_layer), EXT)
        utils.show_tensor(opt_img, 1, filename=im_path, scale=scale, pad_value=pad_value)


class DeepDream:

    def __init__(self, model, images):
        self.model = model
        self.model.eval()
        self.images = images
        self.conv_output = None

    def hook_layer(self, selected_layer, selected_filter):
        def hook_function(module, grad_input, grad_output):
            self.conv_output = grad_output[0, selected_filter]

        layer = self.model.features._modules[str(selected_layer - 1)]
        layer.register_forward_hook(hook_function)

    def visualise(self, selected_layer, selected_filter, save=False, scale=1, pad_value=0):

        self.hook_layer(selected_layer, selected_filter)
        import copy
        input = self.images.detach().clone()
        input.requires_grad = True
        optimizer = optim.SGD([input], lr=3, weight_decay=1e-4)
        for i in range(1, 251):
            optimizer.zero_grad()
            x = input
            for index, layer in enumerate(self.model.features):
                x = layer(x)
                if index + 1 == selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))

        image = self.images.detach()
        image *= utils.MNIST_std
        image += utils.MNIST_mean
        input = utils.tensor_norm(input.detach())
        x = torch.cat((image*255, input*255), dim=0)

        im_path = None if not save else '../result/dd_layer{}_filter{}.{}'.format(str(selected_layer), str(selected_filter), EXT)
        utils.show_tensor(x, 2, filename=im_path, scale=scale, pad_value=pad_value)


if __name__ == '__main__':

    pretrained_model = mnist.LeNet5(pretrained=True)
    # One picture at a time
    images, tags = utils.load_test_images([10])

    # cnn_layer = 1
    # vis = StraightForward(pretrained_model.features, images)
    # vis.visualise_layer(cnn_layer, save=True, scale=2)
    # vis.visualise_filter(save=True, scale=6)

    # cnn_layer = 5
    # vis = VanillaBackprop(pretrained_model, images, tags)
    # vis.visualise(cnn_layer, save=True, scale=4)

    # cnn_layer = 5
    # vis = GuidedBackprop(pretrained_model, images, tags)
    # vis.visualise(cnn_layer, save=True, scale=4, pad_value=1)

    # cnn_layer = 5
    # vis = GradCAM(pretrained_model, images, tags)
    # vis.visualise(cnn_layer, save=True, scale=6, pad_value=1)

    # cnn_layer = 3
    # vis = InvertedRepresentation(pretrained_model, images)
    # vis.visualise(cnn_layer, save=True, scale=6)

    # cnn_layer = 3
    # cnn_filter = 0
    # vis = DeepDream(pretrained_model, images)
    # vis.visualise(cnn_layer, cnn_filter, save=True, scale=6, pad_value=1)

