DATA_ROOT = "./data/makeathon-challenge"

S2_TRAIN = f"{DATA_ROOT}/sentinel-2/train"
S2_TEST = f"{DATA_ROOT}/sentinel-2/test"

S1_TRAIN = f"{DATA_ROOT}/sentinel-1/train"
S1_TEST = f"{DATA_ROOT}/sentinel-1/test"

AEF_TRAIN = f"{DATA_ROOT}/aef-embeddings/train"
AEF_TEST = f"{DATA_ROOT}/aef-embeddings/test"

LABELS_ROOT = f"{DATA_ROOT}/labels/train"
RADD_DIR = f"{LABELS_ROOT}/radd"
GLADL_DIR = f"{LABELS_ROOT}/gladl"
GLADS2_DIR = f"{LABELS_ROOT}/glads2"

TRAIN_META = f"{DATA_ROOT}/metadata/train_tiles.geojson"
TEST_META = f"{DATA_ROOT}/metadata/test_tiles.geojson"

FUSED_LABELS_DIR = "./outputs/fused_labels"
PREDICTIONS_DIR = "./outputs/predictions"
SUBMISSION_DIR = "./submission"

TILE_SIZE = 1002
PIXEL_M = 10
DEFOR_START_YEAR = 2020
