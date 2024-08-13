import os
import shutil
import random
import tempfile

def create_split_directories(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def split_dataset(source_img_dir, source_lbl_dir, dest_img_dir_train, dest_img_dir_val, dest_lbl_dir_train, dest_lbl_dir_val, split_ratio=0.2):
    files = [f for f in os.listdir(source_img_dir) if f.endswith('.png')]
    random.shuffle(files)
    split_idx = int(len(files) * (1 - split_ratio))
    
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for file in train_files:
        shutil.copy(os.path.join(source_img_dir, file), dest_img_dir_train)
        lbl_file = file.replace('.png', '.txt')  # Ensure label file has .txt extension
        lbl_src_path = os.path.join(source_lbl_dir, lbl_file)
        if os.path.exists(lbl_src_path):
            shutil.copy(lbl_src_path, dest_lbl_dir_train)
        else:
            print(f"Warning: Label file {lbl_src_path} not found for image {file}")

    for file in val_files:
        shutil.copy(os.path.join(source_img_dir, file), dest_img_dir_val)
        lbl_file = file.replace('.png', '.txt')  # Ensure label file has .txt extension
        lbl_src_path = os.path.join(source_lbl_dir, lbl_file)
        if os.path.exists(lbl_src_path):
            shutil.copy(lbl_src_path, dest_lbl_dir_val)
        else:
            print(f"Warning: Label file {lbl_src_path} not found for image {file}")

# Create a temporary directory to hold the original files
with tempfile.TemporaryDirectory() as temp_dir:
    temp_img_dir = os.path.join(temp_dir, "images")
    temp_lbl_dir = os.path.join(temp_dir, "labels")
    shutil.copytree(r"C:\Users\daksh\Vamsi Python\Deep Learning\PatMovementImages\obj_train_data\images\train", temp_img_dir)
    shutil.copytree(r"C:\Users\daksh\Vamsi Python\Deep Learning\PatMovementImages\obj_train_data\labels\train", temp_lbl_dir)

    # Create the split directories
    base_path = r"C:\Users\daksh\Vamsi Python\Deep Learning\PatMovementImages\obj_train_data"
    create_split_directories(os.path.join(base_path, "images"), ["train", "val"])
    create_split_directories(os.path.join(base_path, "labels"), ["train", "val"])

    # Split the dataset
    split_dataset(
        source_img_dir=temp_img_dir,
        source_lbl_dir=temp_lbl_dir,
        dest_img_dir_train=os.path.join(base_path, "images", "train"),
        dest_img_dir_val=os.path.join(base_path, "images", "val"),
        dest_lbl_dir_train=os.path.join(base_path, "labels", "train"),
        dest_lbl_dir_val=os.path.join(base_path, "labels", "val"),
        split_ratio=0.2
    )
