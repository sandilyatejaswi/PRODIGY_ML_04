import numpy as np

gesture_names = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down"
]

np.save("gesture_names.npy", gesture_names)

print("gesture_names.npy fixed!")