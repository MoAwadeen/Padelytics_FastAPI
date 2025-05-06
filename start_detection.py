import timeit
import supervision as sv
from config import *
from json_maker import analyze_player_performance
from trackers import (
    PlayerTracker,
    BallTracker,
    KeypointsTracker,
    TrackingRunner,
)

def start_detection():
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps,
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    # Instantiate trackers
    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    players_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL,
        polygon_zone=None,
        batch_size=PLAYERS_TRACKER_BATCH_SIZE,
        annotator=PLAYERS_TRACKER_ANNOTATOR,
        show_confidence=True,
        load_path=PLAYERS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_TRACKER_SAVE_PATH,
    )

    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker,
            ball_tracker,
            keypoints_tracker,
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
        collect_data=COLLECT_DATA,
    )

    runner.run()

    runner.export_ball_data(COLLECT_BALL_DATA_PATH)

    if COLLECT_DATA:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        data.to_csv(COLLECT_DATA_PATH)

    t2 = timeit.default_timer()
    print("Duration (min): ", (t2 - t1) / 60)

    players_file = "C:/Users/modyy/Grad Project/Ai/AiGit/AiGit/players_data.csv"
    ball_file = "C:/Users/modyy/Grad Project/Ai/AiGit/AiGit/output/datasets/ball_data.csv"
    players_list = ["player1", "player2", "player3", "player4"]
    result = analyze_player_performance(players_file, ball_file, players_list)

if __name__ == "__main__":
    start_detection()