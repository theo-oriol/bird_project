from dotenv import load_dotenv
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import os 
load_dotenv()

def rgb_to_ggg(path: Path) -> np.ndarray:
    """
    Charge une image RGB et retourne un tableau 2D float64
    correspondant au seul canal vert (modèle de luminance aviaire).
    """
    img = np.array(Image.open(path).convert("RGBA"))
    mask = img[:,:, 3]>0
    g_channel = img[:, :, 1]
    img = g_channel[mask]
    arr = np.array(img, dtype=np.float64)
    return arr   # canal G uniquement, tableau 2D


def lum_match(arrays):

    target_mean = np.mean([a.mean() for a in arrays])
    target_std  = np.mean([a.std()  for a in arrays])

    results = []
    for arr in arrays:
        m, s = arr.mean(), arr.std()
        if s < 1e-8:
            results.append(np.full_like(arr, target_mean))
        else:
            Z = (arr - m) / s
            E = Z * target_std + target_mean
            results.append(np.clip(E, 0, 255))
    return results


def _compute_target_histogram(arrays: list[np.ndarray],
                               n_bins: int = 256) -> np.ndarray:
    """
    Calcule l'histogramme cible = moyenne des histogrammes du set
    (fonction avgHist de SHINE, suivie de tarhist).

    Retourne un tableau de n_bins entiers dont la somme = N pixels
    de la première image (toutes supposées de même taille).
    """
    N = arrays[0].size
    avg = np.zeros(n_bins, dtype=np.float64)
    for arr in arrays:
        counts, _ = np.histogram(arr.ravel(), bins=n_bins, range=(0, 255))
        avg += counts
    avg /= len(arrays)

    # Arrondir pour que la somme = N exactement
    target = np.round(avg).astype(np.int64)
    diff = N - target.sum()
    # Ajuster le bin le plus peuplé pour absorber l'erreur d'arrondi
    target[np.argmax(target)] += diff
    return target



def _exact_hist_match(source: np.ndarray,
                      target_hist: np.ndarray) -> np.ndarray:
    """
    Algorithme exact de correspondance d'histogramme (Table 1 de Willenbockel).

    Step 0 : histogramme cible H = {h0, h1, ..., h255}
    Step 1 : trier les pixels de l'image source par valeur croissante
             (en cas d'égalité, ordre aléatoire)
    Step 2 : assigner les h0 premiers pixels la valeur 0,
             les h1 suivants la valeur 1, etc.
    """
    flat = source.ravel().astype(np.float64)
    N = flat.size

    # Step 1 : tri avec bruit aléatoire pour départager les ex-æquo
    noise = np.random.uniform(0, 1e-9, size=N)
    sort_idx = np.argsort(flat + noise, kind='stable')

    # Step 2 : assignation des nouvelles valeurs selon l'histogramme cible
    new_values = np.empty(N, dtype=np.float64)
    pos = 0
    for value, count in enumerate(target_hist):
        new_values[sort_idx[pos:pos + count]] = value
        pos += count

    return new_values.reshape(source.shape)


def hist_match(arrays: list[np.ndarray], view: str, output_dir: Path) -> list[np.ndarray]:
    """
    Équivalent de histMatch : égalisation exacte des histogrammes.

    Si target_hist est None, la cible est la moyenne des histogrammes
    du set (comportement par défaut de SHINE).
    """
    target_hist = _compute_target_histogram(arrays)

    ###############################
    np.save(output_dir / f"target_hist_grayscale_{view}", target_hist)
    ###############################
    results = []
    for arr in arrays:
        matched = _exact_hist_match(arr, target_hist)
        results.append(np.clip(matched, 0, 255))
    return results


def shine_ggg_pipeline(image_paths: list[Path],
                       steps: list[str], view: str, output_dir: Path) -> list[np.ndarray]:

    print("=== Étape 0 : Extraction du canal G ===")
    arrays = [rgb_to_ggg(p) for p in image_paths]
    print(f"  {len(arrays)} images chargées, taille : {arrays[0].shape}\n")
    if "lum" in steps:
        print("=== Étape 1 : lumMatch (normalisation μ et σ) ===")
        M = np.mean([a.mean() for a in arrays])
        S = np.mean([a.std()  for a in arrays])
        print(f"  Moyenne cible M = {M:.2f},  Écart-type cible S = {S:.2f}")

        ###############################
        PARAMETERS["M"] = M
        PARAMETERS["S"] = S
        ###############################

        print("NOMBRE D'IMAGE : ",len(arrays))
        arrays = lum_match(arrays)
        print()

    if "hist" in steps:
        print("=== Étape 2 : histMatch (égalisation exacte d'histogramme) ===")
        arrays = hist_match(arrays, view=v, output_dir=output_dir)
        print("  Histogrammes équalisés.\n")

    return arrays



if __name__ == "__main__":
    SOURCE_IMG = Path(os.environ["SOURCE_IMG"] )
    VIEW = ["Back", "Belly", "Side"]
    for v in VIEW:
        TARGET_IMG = f"V0_NEW_Segmented-Aves-{v}-224-RGBA"
        PARAMETERS = {}
        images_path = SOURCE_IMG/TARGET_IMG
        
        list_of_images = [im for im in images_path.iterdir() if im.is_file() and im.suffix == ".png"]
        steps = ["lum", "hist"]
        print(f"Processing view '{v}' with {len(list_of_images)} images...")
        output_path = Path(f"configs/meta_data/grayscale/{TARGET_IMG}")
        os.makedirs(output_path, exist_ok=True)
        shine_ggg_pipeline(
            image_paths=list_of_images,
            steps=steps,
            view=v,
            output_dir=output_path,
        )

        import json
        with open(output_path / f'meta_data_grayscale_{v}.json', 'w') as f:
            json.dump(PARAMETERS, f)


