import os
import shutil
from tqdm import tqdm

os.chdir("../dataset")

asli_path = "1_0_CoMoFoD_small_v2"
asli = os.listdir(asli_path)

is_aug_included = True

new_path = "1_2_aug_included" if is_aug_included else "1_1_image_only"

if not os.path.exists(new_path):
    os.makedirs(new_path)
else:
    print("Folder already exist, deleting the exist one...\n\n")
    shutil.rmtree(new_path)
    os.makedirs(new_path)


print(f"Start copy to {new_path}")


def copy_image(filename, split_name):
    class_label = "original" if split_name[1].lower() == "O".lower() else "forged"
        
    class_path = os.path.join(new_path, class_label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    shutil.copy2(os.path.join(asli_path, filename), class_path)



for i in tqdm(range(len(asli))):
    filename = asli[i]
    name_only, extension = filename.rsplit(".")
    
    if extension.lower() == "txt".lower():
        continue


    split_name_only = name_only.split("_")
    
    if is_aug_included:
        if split_name_only[1].lower() == "M".lower() or split_name_only[1].lower() == "B".lower():
            continue

        copy_image(filename, split_name_only)
    else:
        if split_name_only[1].lower() == "M".lower() or split_name_only[1].lower() == "B".lower() or len(split_name_only) > 2:
            continue

        copy_image(filename, split_name_only)



print("Done!")