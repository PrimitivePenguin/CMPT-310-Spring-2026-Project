IMAGE_SIZE = 48

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

RAW_TRAIN_DIR = "data/raw/train"
RAW_TEST_DIR = "data/raw/test"
PROCESSED_DIR = "data/processed"

OUTPUT_FEATURE_SIZE = 256