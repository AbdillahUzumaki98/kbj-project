import os
import shutil
import cv2
import pywt
import random
from skimage import io

os.chdir("../dataset")

is_aug = True

source_path = "1_2_aug_included" if is_aug else "1_1_image_only"
source_dirs = os.listdir(source_path)

dest_path = "2_0_ready"
dest_prep_path = "2_1_swt-2"
dest_y_cb_cr_path = "2_2_y_cb_cr"

IMAGE_SIZE = (400, 400)


def check_or_clear_folder(path, is_need_clear):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if is_need_clear:
            print(f"Delete {path}")
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"Done - Delete {path}\n\n")



check_or_clear_folder(dest_path, is_need_clear=True)
check_or_clear_folder(dest_prep_path, is_need_clear=True)
check_or_clear_folder(dest_y_cb_cr_path, is_need_clear=True)

img_names = []
for filename in os.listdir("1_1_image_only/original"):
    img_names.append(filename.split("_")[0])



def copy_data(names, data_type):
    index = random.randint(0, len(names) - 1)
    
    name = names[index]
    print(f"Generating {name}-{data_type}")
    
    for label_name in source_dirs:
        dest_full_path = os.path.join(dest_path, data_type, label_name)
        dest_prep_full_path = os.path.join(dest_prep_path, data_type, label_name)
        dest_y_cb_cr_full_path = os.path.join(dest_y_cb_cr_path, data_type, label_name)
        check_or_clear_folder(dest_full_path, is_need_clear=False)
        check_or_clear_folder(dest_prep_full_path, is_need_clear=False)
        check_or_clear_folder(dest_y_cb_cr_full_path, is_need_clear=False)

        filenames = os.listdir(os.path.join(source_path, label_name))
        filtered = []
        for filename in filenames:
            if name in filename:
                filtered.append(filename)

        for data in filtered:
            file_path = os.path.join(source_path, label_name, data)
            im = io.imread(file_path)
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            
            dest_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
            source_area = im.shape[0] * im.shape[1]
            im = cv2.resize(im, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA if source_area > dest_area else cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(dest_full_path, data), im)

            prep_im = im.copy()
            for i in range(3):
                c = prep_im[:, :, i]  # take each channel and after that do wavelet
                levels = pywt.swt2(c, "haar", level=2)
                approx, _ = levels[-1]  # approximation, (horizontal, vertical, diagonal) - take last level
                prep_im[:, :, i] = approx
            
            cv2.imwrite(os.path.join(dest_prep_full_path, data), prep_im)

            y_cb_cr = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
            cv2.imwrite(os.path.join(dest_y_cb_cr_full_path, data), y_cb_cr)
    
    names.pop(index)



total = len(img_names)
train_frac = .8
val_test_frac = (1 - train_frac) / 2

for _ in range(int(total * train_frac)):
    copy_data(img_names, "train")

print("\n")

for _ in range(int(total * val_test_frac)):
    copy_data(img_names, "validation")

print("\n")

for _ in range(int(total * val_test_frac)):
    copy_data(img_names, "test")

print("\nCOMPLETE!")