
from torchvision import transforms as T


def make_transforms(cfg,is_train):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    if is_train:
        return T.Compose([
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])