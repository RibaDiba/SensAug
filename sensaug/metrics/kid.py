"""Calculate Kernel Inception Distance (KID) between two datasets."""

from typing import Callable, Iterator, List, Optional, Tuple, Union

from pathlib import Path
from functools import partial

import cv2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms

from sensaug.dataset.bp_gpu import (
    rgb_gpu_perturb,
    hsv_gpu_perturb,
    blur_gpu_perturb,
    noise_gpu_perturb,
)

IMG_EXTENSIONS = {"png", "jpg"}


class ImagePathDataset(Dataset):
    """Dataset for loading images from a directory."""

    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.images = [
            file for ext in IMG_EXTENSIONS for file in Path(path).rglob(f"*.{ext}")
        ]
        self.transforms = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Tensor:
        path = self.images[i]
        img: Tensor = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class KID:
    """Calculate Kernel Inception Distance (KID) metric.
    TODO: Modify to use Runner class."""

    def __init__(self, path: str, batch_size: int = 25, device="cuda"):
        dataset = ImagePathDataset(path)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

        self.device = device

        self.kid = KernelInceptionDistance(
            subsets=len(self.dataloader),
            subset_size=batch_size,
            reset_real_features=False,
            normalize=False,
        ).to(device)

        for batch in iter(self.dataloader):
            batch = batch.to(device)
            self.kid.update(batch.permute(0, 3, 1, 2), real=True)

    def compute(
        self, perturbation: str, level: Union[float, int]
    ) -> Tuple[Tensor, Tensor]:
        """Compute KID mean and standard deviation of a perturbation.

        Args:
            perturbation: Perturbation to apply to images
            level: Level of perturbation

        Returns:
            KID mean
            KID standard deviation
        """
        self.kid.reset()

        perturb = self.__get_perturb(perturbation, level)

        for batch in iter(self.dataloader):
            batch = perturb(batch).to(self.device)
            self.kid.update(batch.permute(0, 3, 1, 2), real=False)

        return self.kid.compute()

    def __get_perturb(
        self, perturbation: str, level: Union[float, int]
    ) -> Callable[[Tensor], Tensor]:
        """Get perturbation function.

        Args:
            perturbation: Perturbation to apply to images
            level: Level of perturbation

        Returns:
            Perturbation function
        """
        if perturbation == "blur":
            return partial(blur_gpu_perturb, kernel_size=level)
        if perturbation == "noise":
            return partial(noise_gpu_perturb, sigma=level)

        direction_str, channel_str = perturbation.split("_")
        direction = 1 if direction_str == "lighter" else 0

        if channel_str in "RGB":
            channel = "RGB".index(channel_str)
            function = rgb_gpu_perturb
        else:
            channel = "HSV".index(channel_str)
            function = hsv_gpu_perturb

        return partial(function, channel=channel, direction=direction, alpha=level)


def calculate_kid_from_path(
    path1: str, path2: str, batch_size: int = 25
) -> Tuple[Tensor, Tensor]:
    """Calculate KID between two datasets given their paths.

    Args:
        path1: Path of first dataset
        path2: Path of second dataset
        batch_size: Batch size

    Returns:
        KID mean
        KID standard deviation
    """
    images1 = __load_images(path1, batch_size)
    images2 = __load_images(path2, batch_size)

    kid = KernelInceptionDistance(subset_size=batch_size, normalize=True)
    for image in images1:
        kid.update(image, real=True)
    for image in images2:
        kid.update(image, real=False)

    return kid.compute()


def __load_images(path: str, batch_size: int = 25) -> Iterator[Tensor]:
    """Load images from a directory.

    Args:
        path: Path of directory
        batch_size: Batch size

    Returns:
        Iterator of images
    """
    dataset = ImagePathDataset(path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return iter(dataloader)
