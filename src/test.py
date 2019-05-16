import torch
import torch.optim as optim
import torch.nn.functional as F

import mnist, utils


class InvertedRepresentation:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def visualise(self, input_shape, target_class, save=False):

        # target = self.get_output_from_specific_layer(self.images, selected_layer)
        # target = torch.

        opt_img = torch.randn(input_shape, requires_grad=True)
        # utils.show_tensor(opt_img, 1)

        optimizer = optim.SGD([opt_img], lr=1e3, momentum=0.5)
        # optimizer = optim.Adam([opt_img])

        for i in range(10):
            optimizer.zero_grad()

            output = self.model(opt_img)

            # loss = F.nll_loss(output, target_class)
            loss = F.mse_loss(output, target_class)

            loss.backward()
            optimizer.step()
            # Generate image every 5 iterations
            if i % 1 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy())

            # # Reduce learning rate every 40 iterations
            # if i % 400 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/3

        x = opt_img.detach()
        x *= utils.MNIST_mean
        x += utils.MNIST_std
        utils.show_tensor(x, 5, scale=4)


if __name__ == '__main__':

    pretrained_model = mnist.LeNet5(pretrained=True)
    # One picture at a time
    images, tags = utils.load_test_images([10])

    target_class = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    vis = InvertedRepresentation(pretrained_model)
    vis.visualise((10, 1, 28, 28), target_class)

