"""
dataset_loader.py
-----------------
Handles loading of satellite images and their corresponding binary ground truth
masks from the expected folder structure:

    dataset/
    ├── images/   (satellite .jpg / .png files)
    └── masks/    (binary .png mask files — white = building, black = background)

All images and masks are resized to IMG_SIZE × IMG_SIZE and returned as
normalised float32 NumPy arrays.
"""

import os
import numpy as np
import cv2

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE      = 256          # Target spatial resolution (width = height)
DATASET_DIR   = "dataset"    # Root folder that contains images/ and masks/
IMAGES_DIR    = os.path.join(DATASET_DIR, "images")
MASKS_DIR     = os.path.join(DATASET_DIR, "masks")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# ─────────────────────────────────────────────────────────────────────────────


def _sorted_files(directory: str) -> list[str]:
    """
    Return a sorted list of supported image file paths inside *directory*.

    Sorting guarantees that image[i] always corresponds to mask[i] when both
    directories share identical filenames (common convention for segmentation
    datasets).
    """
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """
    Read a single image from *path*, resize it to (IMG_SIZE, IMG_SIZE), and
    return a float32 array with pixel values in [0, 1].

    Parameters
    ----------
    path      : Full path to the image file.
    grayscale : If True the image is loaded as single-channel grayscale
                (used for masks).

    Returns
    -------
    np.ndarray of shape (IMG_SIZE, IMG_SIZE, 1) for grayscale images or
                        (IMG_SIZE, IMG_SIZE, 3) for RGB images.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img  = cv2.imread(path, flag)

    if img is None:
        raise FileNotFoundError(f"[dataset_loader] Could not read file: {path}")

    # Resize to the target resolution using bilinear interpolation for images
    # and nearest-neighbour for masks (preserves crisp binary boundaries).
    interp = cv2.INTER_NEAREST if grayscale else cv2.INTER_LINEAR
    img    = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=interp)

    # Convert BGR → RGB (OpenCV loads in BGR by default)
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Expand dims to get a channel axis, then normalise to [0, 1]
    if grayscale:
        img = np.expand_dims(img, axis=-1)          # (H, W) → (H, W, 1)

    return img.astype(np.float32) / 255.0


def binarise_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Ensure the mask is strictly binary (0 or 1).

    Some datasets use anti-aliased masks or JPEG compression artefacts.
    Thresholding eliminates intermediate values.

    Parameters
    ----------
    mask      : Float32 mask array in [0, 1] with shape (H, W, 1).
    threshold : Pixel values above this are set to 1, others to 0.

    Returns
    -------
    Binary float32 array with the same shape as *mask*.
    """
    return (mask > threshold).astype(np.float32)


def load_dataset(
    images_dir: str = IMAGES_DIR,
    masks_dir:  str = MASKS_DIR,
    verbose:    bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load every image / mask pair from *images_dir* / *masks_dir*.

    Filenames are matched by sort order — both directories must contain files
    with identical base names (e.g. ``image_001.png`` ↔ ``image_001.png``).

    Parameters
    ----------
    images_dir : Path to the directory containing satellite images.
    masks_dir  : Path to the directory containing ground-truth masks.
    verbose    : Print progress information when True.

    Returns
    -------
    images : float32 array of shape (N, IMG_SIZE, IMG_SIZE, 3)
    masks  : float32 binary array of shape (N, IMG_SIZE, IMG_SIZE, 1)
    """
    image_files = _sorted_files(images_dir)
    mask_files  = _sorted_files(masks_dir)

    if len(image_files) == 0:
        raise RuntimeError(f"[dataset_loader] No images found in: {images_dir}")
    if len(image_files) != len(mask_files):
        raise RuntimeError(
            f"[dataset_loader] Image / mask count mismatch: "
            f"{len(image_files)} images vs {len(mask_files)} masks."
        )

    if verbose:
        print(f"[dataset_loader] Found {len(image_files)} image-mask pairs.")

    images, masks = [], []

    for idx, (img_name, msk_name) in enumerate(zip(image_files, mask_files)):
        img_path = os.path.join(images_dir, img_name)
        msk_path = os.path.join(masks_dir,  msk_name)

        image = load_image(img_path,  grayscale=False)
        mask  = load_image(msk_path,  grayscale=True)
        mask  = binarise_mask(mask)

        images.append(image)
        masks.append(mask)

        if verbose and (idx + 1) % 50 == 0:
            print(f"  Loaded {idx + 1}/{len(image_files)} pairs …")

    if verbose:
        print("[dataset_loader] Dataset loaded successfully.")

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    images, masks = load_dataset()
    print(f"Images shape : {images.shape}  dtype={images.dtype}")
    print(f"Masks  shape : {masks.shape}   dtype={masks.dtype}")
    print(f"Pixel range  : images [{images.min():.2f}, {images.max():.2f}]  "
          f"masks [{masks.min():.2f}, {masks.max():.2f}]")
