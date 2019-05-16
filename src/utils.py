import copy
from math import ceil

import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


MNIST_mean = 0.1307
MNIST_std = 0.3081


def load_test_images(indexes):
    """Load mnist test images

    Args:
        indexes: type (list)

    Returns: (imgs, tags), imgs: type (list) PIL Image, tags: type (Tensor)
    """
    mnist = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_mean,), (MNIST_std,))
    ]))

    imgs = list()
    tags = list()
    for i in indexes:
        img, tag = mnist[i]
        imgs.append(img)
        tags.append(tag)
    imgs = torch.stack(imgs)
    imgs.requires_grad_(True)
    tags = torch.LongTensor(tags)

    return imgs, tags


def tensor_norm(x):
    x -= x.min()
    x /= x.max()
    return x


def images2tensor(pil_imgs):
    """Transform images to torch.Tensor

    Args:
        PIL_imgs (list):
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    imgs_arr = np.float32(list(map(np.float32, pil_imgs)))
    imgs_arr = imgs_arr.transpose(3, 1, 2)  # Convert array to D,W,H
    # Convert to float tensor
    imgs_ten = torch.from_numpy(imgs_arr).float()
    # Convert to Pytorch variable
    imgs_ten = torch.Tensor(imgs_ten, requires_grad=True)
    return imgs_ten


def tensor2images(x):

    imgs = list()
    for i in range(x.shape[0]):
        recreated_im = copy.copy(x.data.numpy()[i])
        recreated_im /= MNIST_std
        recreated_im -= MNIST_mean
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)

        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        imgs.append(Image.fromarray(recreated_im))

    return imgs


def save_images(imgs, path):
    """Save images

    Args:
        imgs: type (list) PIL Image
        path: type (str)
    """
    paths = ['{}_{}.jpg'.format(path, i+1) for i in range(len(imgs))]
    for img, path in zip(imgs, paths):
        if not isinstance(img, Image.Image):
            raise ValueError('input is {}, not Image.Image'.format(type(img)))
        img.save(path)


def show_images(imgs):
    """Show PIL images

    Args:
        imgs:
    """
    n = len(imgs)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            raise ValueError('input is ' + type(img) + ' not Image.Image')
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def apply_heatmap_on_image(imgs, map, filename=None, scale=1, pad_value=0):
    """Apply heatmap on image

    Args:
        imgs:
        map:
        filename:

    Returns:
    """
    n, c, h, w = imgs.shape
    imgs[0][0] *= MNIST_std
    imgs[0][0] += MNIST_mean

    import matplotlib.cm as cm
    image = cm.get_cmap('gray')(imgs[0][0])

    img = Image.fromarray(np.uint8(imgs[0][0]*255))

    map = np.uint8(Image.fromarray(map[0][0]*255).resize((h, w), Image.BILINEAR))/255

    import matplotlib.cm as mpl_color_map
    color_map = mpl_color_map.get_cmap('hot')
    heatmap = color_map(map)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap[:, :, 3] = 0.3
    heatmap = Image.fromarray(np.uint8(heatmap*255))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", img.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, img.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    image = np.uint8(Image.fromarray(np.uint8(image*255)).convert('RGB')).transpose((2, 0, 1))
    heatmap = np.uint8(heatmap.convert('RGB')).transpose((2, 0, 1))
    heatmap_on_image = np.uint8(heatmap_on_image.convert('RGB')).transpose((2, 0, 1))
    x = np.stack([image, heatmap, heatmap_on_image])
    x = torch.Tensor(x)
    grid = torchvision.utils.make_grid(x, nrow=3, normalize=True, padding=1, pad_value=pad_value)

    plt.figure(figsize=(3, 1))
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()

    if filename is not None:
        im = format_np_output(grid.detach().numpy())
        im = Image.fromarray(im)
        im = im.resize([x * scale for x in im.size])
        im.save(filename)


def show_tensor(x, col, padding=1, filename=None, scale=1, pad_value=0):
    """
    Args:
        x: type (Tensor), shaoe (n, channel, w, h)
        col: type (int), Number of images displayed in each row of the grid
        padding: type (int)
        filename: type (str)
        scale: type (int)
    """
    n, c, h, w = x.shape
    if not c == 1 and not c == 3:
        raise ValueError('channel is {}, should (1 or 3).'.format(c))

    row = ceil(n / col)
    grid = torchvision.utils.make_grid(x, nrow=col, normalize=True, padding=padding, pad_value=pad_value)
    plt.figure(figsize=(col, row))
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()

    if filename is not None:
        # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = format_np_output(grid.detach().numpy())
        im = Image.fromarray(im)
        im = im.resize([x*scale for x in im.size])
        im.save(filename)


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
        # im = im.resize([x*2 for x in im.size])
    im.save(path)

