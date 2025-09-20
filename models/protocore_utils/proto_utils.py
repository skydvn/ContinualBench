import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.v2 as T
import numpy as np

class AugmentationMixer:
    """
    Utility class to generate mixed batches from original inputs using various augmentation strategies:
      - Concatenate (original + multiple augmented copies)
      - MixUp
      - CutMix
    """
    def __init__(self, transform=None, mode='concat', num_views=2, alpha=1.0, device='cpu'):
        """
        Args:
            transform (callable): torchvision transform for augmentations.
            mode (str): 'concat', 'mixup', or 'cutmix'.
            num_views (int): Number of augmented views per image (only for 'concat').
            alpha (float): Beta distribution parameter for MixUp / CutMix.
            device (str): Device for tensors.
        """
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])

        self.transform = T.Compose([
            T.RandomResizedCrop(size=(32, 32)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.ToDtype(torch.float32, scale=True),  # Tensor-friendly
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


        self.mode = mode
        self.num_views = num_views
        self.alpha = alpha
        self.device = device

    # =========================
    # 1. Concatenate Augmentation
    # =========================
    def _concat(self, inputs, labels):
        """
        Create multiple augmented views and concatenate with original inputs.
        """
        B = inputs.size(0)
        aug_list = []

        for _ in range(self.num_views):
            aug = torch.stack([self.transform(img.cpu()) for img in inputs], dim=0).to(self.device)
            aug_list.append(aug)

        # Combine original + all augmented views
        mixed_inputs = torch.cat([inputs] + aug_list, dim=0)
        mixed_labels = labels.repeat(self.num_views + 1)

        return mixed_inputs, mixed_labels

    # =========================
    # 2. MixUp Augmentation
    # =========================
    def _mixup(self, inputs, labels):
        if self.alpha <= 0:
            return inputs, labels, labels, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        y_a, y_b = labels, labels[index]

        return mixed_inputs, y_a, y_b, lam

    # =========================
    # 3. CutMix Augmentation
    # =========================
    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def _cutmix(self, inputs, labels):
        if self.alpha <= 0:
            return inputs, labels, labels, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).to(self.device)

        shuffled_inputs = inputs[index, :]
        shuffled_labels = labels[index]

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[:, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))

        return inputs, labels, shuffled_labels, lam

    # =========================
    # Main entry point
    # =========================
    def __call__(self, inputs, labels):
        """
        Generate augmented batch depending on the selected mode.
        """
        if self.mode == 'concat':
            return self._concat(inputs, labels)

        elif self.mode == 'mixup':
            return self._mixup(inputs, labels)

        elif self.mode == 'cutmix':
            return self._cutmix(inputs, labels)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
