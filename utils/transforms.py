import torch
import torchvision.transforms as transforms
# from randaugment import RandAugment

def transforms_train():
    return transforms.Compose([
            transforms.RandomResizedCrop(160, scale=(0.90, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NormalizePerImage(),
        ])

def transforms_test():
    return transforms.Compose([
            transforms.CenterCrop(152),
            transforms.Resize(160),
            transforms.ToTensor(),
            NormalizePerImage(),
        ])


class NormalizePerImage(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        # assert torch.prod(torch.std(tensor, dim=(1, 2), keepdim=True)) == 0

        m = torch.mean(tensor, dim=(1, 2), keepdim=True).expand_as(tensor)
        sd = torch.std(tensor, dim=(1, 2), keepdim=True).expand_as(tensor)
        return torch.div(torch.subtract(tensor, m), sd)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
