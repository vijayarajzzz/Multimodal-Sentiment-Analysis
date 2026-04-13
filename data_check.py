import os

DATASET_PATH = r"C:\ML_Project\waste_ai_project\dataset"

for split in ["train", "val", "test"]:
    print(f"\n📂 {split.upper()} SET")
    split_path = os.path.join(DATASET_PATH, split)
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            print(f"{class_name}: {len(os.listdir(class_path))} images")