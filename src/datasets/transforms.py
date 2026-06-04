import os
from random import random
from PIL import Image
import cv2
import json
import numpy as np
from torchvision import transforms as T


def make_transforms_inf_dataaug(cfg, is_train, view=None, meta_dir=None):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    
    additional_transforms = cfg.data.get("additional_transforms") or []
    if isinstance(additional_transforms, str):
        additional_transforms = [additional_transforms]
    
    ad_augment =  []
    for t in additional_transforms:
        if t == "maskonly":
            ad_augment.append(MaskOnly())
        elif t == "gaussian_blur":
            ad_augment.append(GaussianBlur())
        elif t == "shuffle_mask_pixels":
            ad_augment.append(ShuffleMaskPixels())
        elif t == "greyscale_view":
            assert view is not None and meta_dir is not None, \
                "greyscale_view transform requires view and meta_dir arguments"
            ad_augment.append(GrayscaleView(view=view, meta_dir=meta_dir))
        elif t == "no_aug" : pass
        else:
            raise ValueError(f"Unknown additional transform '{t}'")
    
    return T.Compose([
        *ad_augment,
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),
        T.Normalize(mean, std),
    ])



def make_transforms(cfg, is_train, view=None, meta_dir=None):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    additional_transforms = cfg.data.get("additional_transforms") or []
    if isinstance(additional_transforms, str):
        additional_transforms = [additional_transforms]
    
    ad_augment =  []
    for t in additional_transforms:
        if t == "maskonly":
            ad_augment.append(MaskOnly())
        elif t == "gaussian_blur":
            ad_augment.append(GaussianBlur())
        elif t == "shuffle_mask_pixels":
            ad_augment.append(ShuffleMaskPixels())
        elif t == "greyscale_view":
            assert view is not None and meta_dir is not None, \
                "greyscale_view transform requires view and meta_dir arguments"
            ad_augment.append(GrayscaleView(view=view, meta_dir=meta_dir))
        else:
            raise ValueError(f"Unknown additional transform '{t}'")
    if len(ad_augment) > 0:
        ad_augment = [RandomXOR(ad_augment)]

    else :
        ad_augment = []
    
    if is_train:
        return T.Compose([
            *ad_augment,
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x[:3]),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),
        T.Normalize(mean, std),
    ])


class RandomXOR:
    """
    Applique soit transform_a soit transform_b .... — jamais les deux ni aucun.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms
        self.nb_transforms = len(self.transforms)
    def __call__(self, img):
        r = random()
        for t in range(self.nb_transforms):
            if r < (t+1)/self.nb_transforms:
                return self.transforms[t](img)
        return self.transforms[-1](img)
    

class MaskOnly:
    def __call__(self, img: Image.Image) -> Image.Image:
        rgba           = np.array(img.convert("RGBA"))
        mask           = rgba[:, :, 3] > 0
        result         = rgba[:, :, :3].copy()
        result[~mask]  = 255
        result[mask] = 0
        return Image.fromarray(result, mode="RGB")

class GaussianBlur: 
    def __init__(self):
        self.sigma = 40
        self. ksize = int(6 * self.sigma + 1)
    def __call__(self, img: Image.Image):
        img = np.array(img)
        mask = (img[:,:,3]>0).astype(np.float32)
        ksize   = self. ksize 
        img_masked   = img * mask[:, :, None]

        blurred_img  = cv2.GaussianBlur(img_masked, (ksize, ksize), sigmaX=self.sigma)
        blurred_mask = cv2.GaussianBlur(mask,       (ksize, ksize), sigmaX=self.sigma)

        result = blurred_img /  np.maximum(blurred_mask, 1e-6)[:, :, None]

        img[mask > 0] = result[mask > 0]
        img[mask == 0] = 0
        return Image.fromarray(img[:, :, :3], mode="RGB")

class ShuffleMaskPixels:
    def __call__(self, img: Image.Image) -> Image.Image:
        rgba = np.array(img)
        mask = rgba[:, :, 3] > 0
        idx  = np.where(mask)
        n    = len(idx[0])
        if n == 0:
            return img
        pixels                          = rgba[idx[0], idx[1], :3].copy()
        rgba[idx[0], idx[1], :3]        = pixels[np.random.permutation(n)]
        return Image.fromarray(rgba[:, :, :3], mode="RGB")

class GrayscaleView:
    """
    GrayscaleView + luminance normalization + histogram matching.
    One instance per view, parameterized by view name.
    """

    def __init__(self, view: str, meta_dir: str):
        """
        view     : "Back" | "Side" | "Belly"
        meta_dir : path to the folder containing the json and npy files
        """
        meta_path = os.path.join(meta_dir, f"meta_data_grayscale_{view}.json")
        hist_path = os.path.join(meta_dir, f"target_hist_grayscale_{view}.npy")

        with open(meta_path, "r") as f:
            parameters = json.load(f)

        self.target_hist = np.load(hist_path)
        self.M           = parameters["M"]
        self.S           = parameters["S"]

    def __call__(self, img):
        arrays, mask, g_channel = self.rgb_to_ggg(img)
        arrays                  = self.lum_match(arrays, self.M, self.S)
        arrays                  = self.hist_match(arrays, self.target_hist)
        g_channel[~mask]        = 0
        g_channel[mask]         = arrays
        out_uint8               = g_channel.astype(np.uint8)
        out_rgb                 = np.stack([out_uint8, out_uint8, out_uint8], axis=-1)
        return Image.fromarray(out_rgb, mode="RGB")
    
    def rgb_to_ggg(self, img) -> np.ndarray:
        img = np.array(img) 
        mask = img[:,:, 3]>0
        g_channel = img[:, :, 1]
        arr = g_channel[mask]
        return arr, mask, g_channel  


    def lum_match(self,arrays,M,S):
        target_mean = M
        target_std  = S
        m, s = arrays.mean(), arrays.std()
        if s < 1e-8:
            results = np.full_like(arrays, target_mean)
        else:
            Z = (arrays - m) / s
            E = Z * target_std + target_mean
            results = np.clip(E, 0, 255)
        return results

    def _exact_hist_match(self,source: np.ndarray,
                      target_hist: np.ndarray) -> np.ndarray:
        flat = source.ravel().astype(np.float64)
        N = flat.size

        noise = np.random.uniform(0, 1e-9, size=N)
        sort_idx = np.argsort(flat + noise, kind='stable')

        new_values = np.empty(N, dtype=np.float64)
        pos = 0
        for value, count in enumerate(target_hist):
            new_values[sort_idx[pos:pos + count]] = value
            pos += count

        return new_values.reshape(source.shape)

    def hist_match(self,arrays, target_hist) -> list[np.ndarray]:
        matched = self._exact_hist_match(arrays, target_hist)
        results= np.clip(matched, 0, 255)
        return results 