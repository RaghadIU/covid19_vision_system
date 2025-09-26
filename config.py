from pathlib import Path

# ---------- الجذر الرئيسي للمشروع ----------
PROJECT_ROOT = Path(__file__).parent

# ---------- المجلدات الأساسية ----------
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "data_sets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
TRAINING_DIR = PROJECT_ROOT / "training"

# ---------- مجلدات البيانات ----------
IMAGE_DATA_DIR = DATASETS_DIR / "image_data"
VIDEO_DATA_DIR = DATASETS_DIR / "video_data"
MASK_PERSON_TEST_DIR = IMAGE_DATA_DIR / "mask_person_test"
RANDOM_OBJECTS_DIR = IMAGE_DATA_DIR / "random_objects"

# ---------- مسارات النماذج YOLO ----------
YOLOV8_MASK = MODELS_DIR / "yolov8n_mask.pt"
YOLOV8_PERSON = MODELS_DIR / "yolov8n_person.pt"
YOLOV11_MASK = MODELS_DIR / "yolov11n_mask.pt"
YOLOV11_PERSON = MODELS_DIR / "yolov11n_person.pt"

# ---------- التأكد من وجود المجلدات ----------
for directory in [
    MODELS_DIR, DATASETS_DIR, SCRIPTS_DIR, TESTS_DIR, TRAINING_DIR,
    IMAGE_DATA_DIR, MASK_PERSON_TEST_DIR, RANDOM_OBJECTS_DIR, VIDEO_DATA_DIR
]:
    directory.mkdir(exist_ok=True)
