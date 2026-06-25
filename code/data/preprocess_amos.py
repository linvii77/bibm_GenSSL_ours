"""
Preprocess AMOS22 dataset into .npy format for training.

Expected raw dataset structure:
  ./Datasets/amos22/
      imagesTr/amos_XXXX.nii.gz
      labelsTr/amos_XXXX.nii.gz

Output:
  ./AMOS_data/npy/amos_XXXX_image.npy
  ./AMOS_data/npy/amos_XXXX_label.npy

Download AMOS22 from: https://amos22.grand-challenge.org/
The dataset has 500 CT scans with 15 abdominal organ labels.
We use cases amos_0001 ~ amos_0500 (CT only, excluding MRI cases 0501+).

Usage:
    python code/data/preprocess_amos.py
"""
import os
import glob
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from utils.config import Config

config = Config("amos_0.05")
base_dir = config.base_dir  # ./Datasets/amos22


def read_nifti(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def write_txt(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    """Convert raw NIfTI to resized .npy files."""
    npy_dir = os.path.join(config.save_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)

    resize_shape = (
        config.patch_size[0] + config.patch_size[0] // 4,
        config.patch_size[1] + config.patch_size[1] // 4,
        config.patch_size[2] + config.patch_size[2] // 4,
    )  # (80, 160, 160) for patch=(64,128,128)

    image_paths = sorted(glob.glob(os.path.join(base_dir, 'imagesTr', 'amos_*.nii.gz')))
    # Keep only CT scans (amos_0001 ~ amos_0500); skip MRI (amos_0501+)
    image_paths = [p for p in image_paths if int(os.path.basename(p).split('_')[1].split('.')[0]) <= 500]

    print(f"Found {len(image_paths)} CT scans. Preprocessing to {npy_dir} ...")

    for img_path in tqdm(image_paths):
        case_id = os.path.basename(img_path).replace('.nii.gz', '')  # e.g. amos_0001
        label_path = os.path.join(base_dir, 'labelsTr', f'{case_id}.nii.gz')

        if not os.path.exists(label_path):
            print(f"  WARNING: label not found for {case_id}, skipping.")
            continue

        out_image = os.path.join(npy_dir, f'{case_id}_image.npy')
        out_label = os.path.join(npy_dir, f'{case_id}_label.npy')
        if os.path.exists(out_image) and os.path.exists(out_label):
            continue  # already processed

        image = read_nifti(img_path)
        label = read_nifti(label_path).astype(np.int8)

        image_t = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        label_t = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)

        image_t = F.interpolate(image_t, size=resize_shape, mode='trilinear', align_corners=False)
        label_t = F.interpolate(label_t, size=resize_shape, mode='nearest')

        np.save(out_image, image_t.squeeze().numpy())
        np.save(out_label, label_t.squeeze().numpy().astype(np.int8))

    print("Done.")


def process_split_fully(train_ratio=0.8, eval_ratio=0.1, seed=0):
    """
    Split preprocessed cases into train / eval / test.
    Default: 80% train, 10% eval, 10% test  (out of ~200 labeled cases 0001-0200).
    The labeled portion of AMOS22 is cases 0001-0200; 0201-0500 have no labels.
    """
    npy_dir = os.path.join(config.save_dir, 'npy')
    split_dir = os.path.join(config.save_dir, 'split_txts')

    labeled_ids = sorted([
        os.path.basename(p).replace('_image.npy', '')
        for p in glob.glob(os.path.join(npy_dir, 'amos_*_image.npy'))
        if int(os.path.basename(p).split('_')[1]) <= 200
    ])
    unlabeled_ids = sorted([
        os.path.basename(p).replace('_image.npy', '')
        for p in glob.glob(os.path.join(npy_dir, 'amos_*_image.npy'))
        if int(os.path.basename(p).split('_')[1]) > 200
    ])

    rng = np.random.default_rng(seed)
    labeled_ids = rng.permutation(labeled_ids).tolist()

    n = len(labeled_ids)
    n_eval = max(1, int(n * eval_ratio))
    n_test = max(1, int(n * (1 - train_ratio - eval_ratio)))
    n_train = n - n_eval - n_test

    train_ids = sorted(labeled_ids[:n_train])
    eval_ids  = sorted(labeled_ids[n_train:n_train + n_eval])
    test_ids  = sorted(labeled_ids[n_train + n_eval:])

    write_txt(train_ids,     os.path.join(split_dir, 'train.txt'))
    write_txt(eval_ids,      os.path.join(split_dir, 'eval.txt'))
    write_txt(test_ids,      os.path.join(split_dir, 'test.txt'))
    write_txt(unlabeled_ids, os.path.join(split_dir, 'unlabeled.txt'))

    print(f"Split: {len(train_ids)} train / {len(eval_ids)} eval / {len(test_ids)} test / {len(unlabeled_ids)} unlabeled")


def process_split_semi(labeled_ratio=0.05, seed=0):
    """Create labeled/unlabeled split from train.txt for semi-supervised training."""
    split_dir = os.path.join(config.save_dir, 'split_txts')
    train_ids = np.loadtxt(os.path.join(split_dir, 'train.txt'), dtype=str).tolist()

    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(train_ids).tolist()

    n_labeled = max(1, int(len(train_ids) * labeled_ratio))
    labeled_ids   = sorted(train_ids[:n_labeled])
    unlabeled_ids = sorted(train_ids[n_labeled:])

    tag = f'{labeled_ratio:.0%}'.replace('%', 'p')
    write_txt(labeled_ids,   os.path.join(split_dir, f'labeled_{tag}.txt'))
    write_txt(unlabeled_ids, os.path.join(split_dir, f'unlabeled_{tag}.txt'))
    print(f"Semi split ({labeled_ratio*100:.0f}%): {len(labeled_ids)} labeled / {len(unlabeled_ids)} unlabeled")


if __name__ == '__main__':
    process_npy()
    process_split_fully()
    process_split_semi(labeled_ratio=0.05)
