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
dest_swt_2_path = "2_1_swt-2"
dest_dwt_path = "2_2_dwt"
dest_y_cb_cr_path = "2_3_y_cb_cr"
dest_y_cb_cr_swt_2_path = "2_4_y_cb_cr_swt-2"
dest_y_cb_cr_dwt_path = "2_5_y_cb_cr_dwt"

IMAGE_SIZE = (256, 256)


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
check_or_clear_folder(dest_swt_2_path, is_need_clear=True)
check_or_clear_folder(dest_dwt_path, is_need_clear=True)
check_or_clear_folder(dest_y_cb_cr_path, is_need_clear=True)
check_or_clear_folder(dest_y_cb_cr_swt_2_path, is_need_clear=True)
check_or_clear_folder(dest_y_cb_cr_dwt_path, is_need_clear=True)

img_names = []
for filename in os.listdir("1_1_image_only/original"):
    img_names.append(filename.split("_")[0])



def swt(prep_im):
    for i in range(3):
        c = prep_im[:, :, i]  # take each channel and after that do wavelet
        levels = pywt.swt2(c, "haar", level=2)
        approx, _ = levels[-1]  # approximation, (horizontal, vertical, diagonal) - take last level
        prep_im[:, :, i] = approx
    return prep_im



def dwt(prep_im):
    copy_im = prep_im.copy()
    dest_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    source_area = copy_im.shape[0] * copy_im.shape[1]
    copy_im = cv2.resize(copy_im, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA if source_area > dest_area else cv2.INTER_CUBIC)
    for i in range(3):
        c = prep_im[:, :, i]  # take each channel and after that do wavelet
        levels = pywt.dwt2(c, "haar")
        approx, _ = levels  # approximation, (horizontal, vertical, diagonal)
        copy_im[:, :, i] = approx
    return copy_im



def copy_data(names, data_type):
    index = random.randint(0, len(names) - 1)
    
    name = names[index]
    print(f"Generating {name}-{data_type}")
    
    for label_name in source_dirs:
        dest_full_path = os.path.join(dest_path, data_type, label_name)
        dest_swt_2_full_path = os.path.join(dest_swt_2_path, data_type, label_name)
        dest_dwt_full_path = os.path.join(dest_dwt_path, data_type, label_name)
        dest_y_cb_cr_full_path = os.path.join(dest_y_cb_cr_path, data_type, label_name)
        dest_y_cb_cr_swt_2_full_path = os.path.join(dest_y_cb_cr_swt_2_path, data_type, label_name)
        dest_y_cb_cr_dwt_full_path = os.path.join(dest_y_cb_cr_dwt_path, data_type, label_name)
        
        check_or_clear_folder(dest_full_path, is_need_clear=False)
        check_or_clear_folder(dest_swt_2_full_path, is_need_clear=False)
        check_or_clear_folder(dest_dwt_full_path, is_need_clear=False)
        check_or_clear_folder(dest_y_cb_cr_full_path, is_need_clear=False)
        check_or_clear_folder(dest_y_cb_cr_swt_2_full_path, is_need_clear=False)
        check_or_clear_folder(dest_y_cb_cr_dwt_full_path, is_need_clear=False)

        filenames = os.listdir(os.path.join(source_path, label_name))
        filtered = []
        for filename in filenames:
            if name in filename:
                filtered.append(filename)

        for data in filtered:
            file_path = os.path.join(source_path, label_name, data)
            orig_im = io.imread(file_path)
            orig_im = cv2.cvtColor(orig_im, cv2.COLOR_RGBA2BGR)
            
            dest_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
            source_area = orig_im.shape[0] * orig_im.shape[1]
            im = cv2.resize(orig_im, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA if source_area > dest_area else cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(dest_full_path, data), im)

            swt_2_im = im.copy()
            swt_2_im = swt(swt_2_im)
            cv2.imwrite(os.path.join(dest_swt_2_full_path, data), swt_2_im)

            dwt_im = orig_im.copy()
            dwt_im = dwt(dwt_im)
            cv2.imwrite(os.path.join(dest_dwt_full_path, data), dwt_im)

            y_cb_cr = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
            cv2.imwrite(os.path.join(dest_y_cb_cr_full_path, data), y_cb_cr)

            y_cb_cr_swt_2 = y_cb_cr.copy()
            y_cb_cr_swt_2 = swt(y_cb_cr_swt_2)
            cv2.imwrite(os.path.join(dest_y_cb_cr_swt_2_full_path, data), y_cb_cr_swt_2)

            y_cb_cr_dwt = cv2.cvtColor(orig_im, cv2.COLOR_BGR2YCR_CB)
            y_cb_cr_dwt = dwt(y_cb_cr_dwt)
            cv2.imwrite(os.path.join(dest_y_cb_cr_dwt_full_path, data), y_cb_cr_dwt)
    
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