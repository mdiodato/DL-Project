import torch
from models.networks.architecture import VGG19StyleAndContent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use VGG19 to extract style and content features from the intermediate layers.
vgg = VGG19StyleAndContent().to(device)
style_weights = [0.1, 0.2, 0.4, 0.8, 1.6]


def get_gram_matrix(feature):
    """
    Compute the gram matrix by converting to 2D tensor and doing dot product
    feature: (batch, channel, height, width)
    """
    b, c, h, w = feature.size()
    feature = feature.view(b*c, h*w)
    gram = torch.mm(feature, feature.t())
    return gram


def get_losses(real_img, guide_img, fake_img):
    """
    Calculate the content loss between guide image and fake image
    and the style loss between real image and fake image
    (parameters from https://arxiv.org/pdf/1904.11617.pdf)
    """
    # Get features from VGG19
    _, guide_content = vgg(guide_img)
    real_style, _ = vgg(real_img)
    fake_style, fake_content = vgg(fake_img)

    # Compute content loss
    content_loss = torch.mean((guide_content - fake_content) ** 2)

    # Compute style loss
    style_loss = 0
    for i in range(len(real_style)):
        b, c, h, w = real_style[i].shape
        real_gram = get_gram_matrix(real_style[i])
        fake_gram = get_gram_matrix(fake_style[i])
        layer_style_loss = style_weights[i] * torch.mean((real_gram - fake_gram) ** 2)
        style_loss += layer_style_loss / (c * h * w)

    return style_loss, content_loss
