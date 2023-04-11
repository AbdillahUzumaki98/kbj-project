import os
import shutil
import cv2
import pywt
import random
from tqdm import tqdm
from skimage import io

os.chdir("../dataset")

is_aug = False

source_path = "1_2_aug_included" if is_aug else "1_1_image_only"
source_dirs = os.listdir(source_path)

dest_path = "2_0_ready"
dest_prep_path = "2_1_swt-2"

IMAGE_SIZE = (224, 224)


def check_or_clear_folder(path, is_need_clear):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if is_need_clear:
            shutil.rmtree(path)
            os.makedirs(path)



check_or_clear_folder(dest_path, is_need_clear=True)
check_or_clear_folder(dest_prep_path, is_need_clear=True)



def copy_data(data, data_type, label_name):
    dest_full_path = os.path.join(dest_path, data_type, label_name)
    dest_prep_full_path = os.path.join(dest_prep_path, data_type, label_name)
    check_or_clear_folder(dest_full_path, is_need_clear=False)
    check_or_clear_folder(dest_prep_full_path, is_need_clear=False)

    index = random.randint(0, len(data) - 1)
    file_path = os.path.join(source_path, label_name, data[index])
    im = io.imread(file_path)
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    
    h, w = im.shape[:2]
    im = cv2.resize(im, (w, IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA if h > IMAGE_SIZE[0] else cv2.INTER_CUBIC)
    im = cv2.resize(im, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA if w > IMAGE_SIZE[1] else cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(dest_full_path, data[index]), im)

    prep_im = im.copy()
    for i in range(3):
        c = prep_im[:, :, i]  # take each channel and after that do wavelet
        levels = pywt.swt2(c, "haar", level=2)
        approx, _ = levels[-1]  # approximation, (horizontal, vertical, diagonal) - take last level
        prep_im[:, :, i] = approx
    
    cv2.imwrite(os.path.join(dest_prep_full_path, data[index]), prep_im)
    data.pop(index)



for folder in source_dirs:
    filenames = os.listdir(os.path.join(source_path, folder))
    total = len(filenames)

    train_frac = .8
    val_test_frac = (1 - train_frac) / 2

    print(f"Generating data train-{folder}")
    for _ in tqdm(range(int(total * train_frac))):
        copy_data(filenames, "train", folder)
    
    print("\n")

    print(f"Generating data validation-{folder}")
    for _ in tqdm(range(int(total * val_test_frac))):
        copy_data(filenames, "validation", folder)
    
    print("\n")

    print(f"Generating data test-{folder}")
    for _ in tqdm(range(int(total * val_test_frac))):
        copy_data(filenames, "test", folder)
    
    print("\n")


print("COMPLETE!")