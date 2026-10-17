"""Eight mapped NTU RGB+D action groups used by the supplied training notebook."""

ACTION_GROUPS = {
    "walking": [59, 60],
    "standing": [1, 2, 3, 4, 10, 14, 15, 18, 19, 20, 21, 22, 23, 28, 31, 32, 33, 35, 36, 38, 39, 40, 41, 49, 56, 58],
    "sitting": [8, 11, 12, 29, 30],
    "bending": [5, 6, 13, 25, 37],
    "falling": [43],
    "getting_up": [9],
    "stumbling": [42],
    "abnormal_movement": [44, 45, 46, 47, 48],
}

CLASS_NAMES = list(ACTION_GROUPS.keys())
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
ACTION_TO_CLASS = {
    action_id: class_name
    for class_name, action_ids in ACTION_GROUPS.items()
    for action_id in action_ids
}
NUM_CLASSES = len(CLASS_NAMES)
SEQUENCE_LENGTH = 30
FEATURE_DIM = 51

if NUM_CLASSES != 8:
    raise RuntimeError(f"Expected 8 classes, found {NUM_CLASSES}")
