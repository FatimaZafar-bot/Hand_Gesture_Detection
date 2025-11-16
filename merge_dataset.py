import os
import shutil

SOURCE = r"C:\Users\Hp\Desktop\HandGesture\leapGestRecog"
TARGET = r"C:\Users\Hp\Desktop\HandGesture\final_dataset"

gesture_classes = [
    "01_palm", "02_I", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]

os.makedirs(TARGET, exist_ok=True)
for g in gesture_classes:
    os.makedirs(os.path.join(TARGET, g), exist_ok=True)

for user in os.listdir(SOURCE):
    user_path = os.path.join(SOURCE, user)

    if not os.path.isdir(user_path):
        continue

    print("Processing user:", user)

    for gesture in gesture_classes:
        src = os.path.join(user_path, gesture)
        dst = os.path.join(TARGET, gesture)

        if os.path.isdir(src):
            for img in os.listdir(src):
                src_img = os.path.join(src, img)
                dst_img = os.path.join(dst, f"{user}_{img}")  # avoid duplicates
                shutil.copy(src_img, dst_img)

print("\n🎉 Dataset successfully merged into final_dataset!")
print("Final dataset path:", TARGET)
