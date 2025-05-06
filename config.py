""" General configurations for main.py """

# Input video path
INPUT_VIDEO_PATH = "uploads/videos/uploaded_video.mp4"

# Inference video path
OUTPUT_VIDEO_PATH = "./output/videos/results.mp4"

# True to collect 2d projection data
COLLECT_DATA = True

# Collected data path
COLLECT_DATA_PATH = "players_data.csv"
COLLECT_BALL_DATA_PATH = "AiGit/AiGit/players_data.csv"

# Maximum number of frames to be analysed
MAX_FRAMES = None

# Players tracker
PLAYERS_TRACKER_MODEL = "./weights/players_detection/yolov8m.pt"
PLAYERS_TRACKER_BATCH_SIZE = 8
PLAYERS_TRACKER_ANNOTATOR = "rectangle_bounding_box"
PLAYERS_TRACKER_LOAD_PATH = None    #"./cache/players_detections.json"
PLAYERS_TRACKER_SAVE_PATH = "./cache/players_detections.json"

# Ball tracker
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"
BALL_TRACKER_BATCH_SIZE = 8
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = 400
BALL_TRACKER_LOAD_PATH = None    #"./cache/ball_detections.json"
BALL_TRACKER_SAVE_PATH = "./cache/ball_detections.json"

# Court keypoints tracker
KEYPOINTS_TRACKER_MODEL = "./weights/court_keypoints_detection/best.pt"
KEYPOINTS_TRACKER_BATCH_SIZE = 8
KEYPOINTS_TRACKER_MODEL_TYPE = "yolo"
KEYPOINTS_TRACKER_LOAD_PATH = None    #"./cache/keypoints_detections.json"
KEYPOINTS_TRACKER_SAVE_PATH = "./cache/keypoints_detections.json"