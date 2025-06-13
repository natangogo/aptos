NUM_CLASSES = 35
BASIC_CHANNELS = 32
# PATH_APTOS_VIDEO = r"F:/APTOS/aptos_ophnet_new2/aptos_videos"
PATH_APTOS_VIDEO = r"F:/APTOS/aptos_ophnet_new3/fixed_videos"
PATH_APTOS_CSV = r"F:/APTOS/aptos_ophnet_new2/APTOS_train-val_annotation.csv"
EPOCHS = 10
FRAME_EXTRACT_INTERVAL_SEC = 1
FPS_STEP = 1.0
SAVE_PATH = "./output"

# train
BATCH_NUM_TRAIN = 1
NUM_CLASSES_TRAIN = 35
NUM_WORKS_TRAIN = 2

# validation
BATCH_NUM_VAL = 1
NUM_WORKS_VAL = 2

# test
BATCH_DATA_TEST = 16
NUM_CLASSES_TEST = 35
# NUM_CLASSES_TEST = 10
SKIP_BATCH_TEST = 100
# worksとは、データを読み込むためのサブプロセス(作業員)の数のこと
NUM_WORKS_TEST = 4
