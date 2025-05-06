from typing import Optional
from tqdm import tqdm
import timeit
from copy import deepcopy
from pathlib import Path
import cv2
import supervision as sv

from trackers.players_tracker.players_tracker import Players
from trackers.ball_tracker.ball_tracker import Ball
from trackers.keypoints_tracker.keypoints_tracker import Keypoints
from trackers.tracker import Tracker
from analytics import ProjectedCourt, DataAnalytics
from trackers.ball_tracker.export_ball_data import export_ball_data_to_excel  # الإضافة الجديدة

class TrackingRunner:

    """
    Abstraction that implements a memory efficient pipeline to run
    a sequence of trackers over a sequence of video frames

    Attributes:
        trackers: sequence of trackers of interest
        video_path: source video path
        inference_path: path where to save the inference results
        start: indicates the starting position from which video should generate frames
        stride: indicates the interval at which frames are returned
        end: indicates the ending position at which video should stop generating frames.
             If None, video will be read to the end.   
        collect_data: True to collect data from projected court
    """

    def __init__(
        self, 
        trackers: list[Tracker],
        video_path: str | Path,
        inference_path: str | Path,
        start: int = 0,
        end: Optional[int] = None,
        collect_data: bool = False, 
    ) -> None:
    
        self.video_path = video_path
        self.inference_path = inference_path
        self.start = start
        self.stride = 1
        self.end = end
        self.video_info = sv.VideoInfo.from_video_path(video_path=video_path)

        if self.end is None:
            self.total_frames = self.video_info.total_frames
        else:
            self.total_frames = self.end - self.start

        self.trackers = {}
        self.is_fixed_keypoints = False
        for tracker in trackers:
            self.trackers[str(tracker)] = tracker.video_info_post_init(self.video_info)

            if isinstance(tracker, Keypoints):
                self.is_fixed_keypoints = False  

        if self.is_fixed_keypoints:
            print("-"*40)
            print("runner: Using fixed court keypoints")
            print("-"*40)

        self.projected_court = ProjectedCourt(self.video_info)
        if collect_data:
            print("runner: Ready for data collection")
            self.data_analytics = DataAnalytics()
        else:
            self.data_analytics = None
    
    def restart(self) -> None:
        """
        Restart all trackers and data
        """
        for tracker in self.trackers.values():
            tracker.restart()
        
        if self.data_analytics:
            self.data_analytics.restart()

    def draw_and_collect_data(self) -> None:
        """
        Draw tracker results and 2D court projections across all video frames.
        Collect data for further analysis.
        """

        print(f"runner: Writing results into {str(self.inference_path)}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.inference_path,
            fourcc,
            float(self.video_info.fps),
            self.video_info.resolution_wh,
        )

        frame_generator = sv.get_video_frames_generator(
            self.video_path,
            start=self.start,
            stride=self.stride,
            end=self.end,
        )

        for frame_index, frame in tqdm(enumerate(frame_generator)):
    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.putText(
                frame_rgb,
                f"Frame: {frame_index + 1}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                1,
            )

            players_detection = None
            ball_detection = None
            keypoints_detection = None
            for tracker in self.trackers.values():
                
                try:
                    prediction = tracker.results[frame_index]
                except IndexError as e:
                    print(f"runner: {str(tracker)} frame {frame_index}")
                    raise(e)
                
                frame_rgb = prediction.draw(frame_rgb, **tracker.draw_kwargs())

                if tracker.object() == Players:
                    players_detection = deepcopy(prediction)
                elif tracker.object() == Ball:
                    ball_detection = deepcopy(prediction)
                elif tracker.object() == Keypoints:
                    keypoints_detection = deepcopy(prediction)
               
            output_frame, self.data_analytics = self.projected_court.draw_projections_and_collect_data(
                frame_rgb,
                keypoints_detection=keypoints_detection,
                players_detection=players_detection,
                ball_detection=ball_detection,
                data_analytics=self.data_analytics,
                is_fixed_keypoints=self.is_fixed_keypoints,
            )

            if self.data_analytics is not None:
                self.data_analytics.step(1)

            out.write(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
        
        out.release()

        if self.data_analytics is not None:
            self.data_analytics.frames = self.data_analytics.frames[:-1]

        print("runner: Done.") 

    def export_ball_data(self, output_file: str) -> None:
        """
        Export ball tracking data to an Excel file, including players' positions.
        """

        if "ball_tracker" not in self.trackers or "players_tracker" not in self.trackers:
            print("runner: ball_tracker or players_tracker not available. Skipping export.")
            return

        ball_detections = self.trackers["ball_tracker"].results.predictions
        players_results = self.trackers["players_tracker"].results.predictions

        players_per_frame = {}
        for frame_id, players_obj in enumerate(players_results):
            for player in players_obj.players:
                x1, y1, x2, y2 = player.xyxy
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2

                if frame_id not in players_per_frame:
                    players_per_frame[frame_id] = {}
                players_per_frame[frame_id][player.id] = (x, y)

        export_ball_data_to_excel(
            ball_detections,
            players_per_frame,
            fps=self.video_info.fps,
            output_file=output_file,
        )

        print(f"runner: Ball data exported to {output_file}")

    def run(self) -> None:
        """
        Run trackers object prediction for every frame in the frame generator
        """

        print(f"runner: Running {self.total_frames} frames")

        for tracker in self.trackers.values():

            if len(tracker) != 0:
                print(f"{tracker.__str__()}: {len(tracker)} predictions stored")
                continue

            tracker.to(tracker.DEVICE)
            print(f"{str(tracker)}: Running on {tracker.DEVICE} ...")

            frame_generator = sv.get_video_frames_generator(
                self.video_path,
                start=self.start,
                stride=self.stride,
                end=self.end,
            )

            t0 = timeit.default_timer()

            tracker.predict_and_update(
                frame_generator, 
                total_frames=self.total_frames,
            )
            t1 = timeit.default_timer()

            tracker.to("cpu")

            print(f"{str(tracker)}: {t1 - t0} inference time.")

            tracker.save_predictions()
        
        self.draw_and_collect_data()

        if self.data_analytics is not None:
            from config import COLLECT_BALL_DATA_PATH
            self.export_ball_data(COLLECT_BALL_DATA_PATH)
