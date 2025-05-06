import pandas as pd
import numpy as np
import math
from trackers.ball_tracker.ball_tracker import Ball

def compute_speed(p1, p2, fps):
    if p1 is None or p2 is None:
        return 0.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    return distance * fps

def compute_angle(p1, p2):
    if p1 is None or p2 is None:
        return None
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def detect_closest_player(ball_pos, players_pos):
    if not players_pos:
        return None, None
    distances = {
        pid: np.linalg.norm(np.array(ball_pos) - np.array(ppos))
        for pid, ppos in players_pos.items()
    }
    closest_pid = min(distances, key=distances.get)
    return closest_pid, distances[closest_pid]

def export_ball_data_to_excel(ball_detections: list[Ball], players_per_frame: dict[int, dict[int, tuple[float, float]]], fps: float = 30.0, output_file: str = "ball_data.xlsx"):
    SKIP_LAST_FRAMES = 5
    rows = []

    for i in range(1, len(ball_detections)):
        b0, b1 = ball_detections[i - 1], ball_detections[i]
        speed = compute_speed(b0.xy, b1.xy, fps)
        angle = compute_angle(b0.xy, b1.xy)

        players_pos = players_per_frame.get(b1.frame, {})
        closest_player, min_dist = detect_closest_player(b1.xy, players_pos)
        hit_player = closest_player

        power = None
        if i >= 2:
            speed_before = compute_speed(ball_detections[i - 2].xy, b0.xy, fps)
            power = abs(speed - speed_before)

        if b1.frame < ball_detections[-SKIP_LAST_FRAMES].frame:
            rows.append({
                "Frame": b1.frame,
                "Ball X": b1.xy[0],
                "Ball Y": b1.xy[1],
                "Speed": speed,
                "Angle": angle,
                "Min Distance to Player": min_dist,
                "Closest Player ID": closest_player,
                "Hit Player ID": hit_player,
                "Hit Power": power,
            })

    df = pd.DataFrame(rows)
    df["Hit Player ID"] = df["Hit Player ID"].fillna(-1)

    MIN_FRAME_GAP = 20
    filtered_rows = []
    last_kept_frame = -MIN_FRAME_GAP - 1
    current_id = None
    current_group = []

    for _, row in df.iterrows():
        pid = row["Hit Player ID"]
        frame = row["Frame"]
        if pid == current_id:
            current_group.append(row)
        else:
            if current_id != -1 and current_group:
                group_df = pd.DataFrame(current_group)
                best_row = group_df.loc[group_df["Min Distance to Player"].idxmin()]
                if best_row["Frame"] - last_kept_frame >= MIN_FRAME_GAP:
                    keep_frame = best_row["Frame"]
                    last_kept_frame = keep_frame
                else:
                    keep_frame = None
                for r in current_group:
                    r_copy = r.copy()
                    if r["Frame"] == keep_frame:
                        filtered_rows.append(r_copy)
                    else:
                        r_copy["Hit Player ID"] = np.nan
                        r_copy["Hit Power"] = np.nan
                        filtered_rows.append(r_copy)
            elif current_id == -1 and current_group:
                filtered_rows.extend(current_group)

            current_group = [row]
            current_id = pid

    if current_group:
        if current_id != -1:
            group_df = pd.DataFrame(current_group)
            best_row = group_df.loc[group_df["Min Distance to Player"].idxmin()]
            if best_row["Frame"] - last_kept_frame >= MIN_FRAME_GAP:
                keep_frame = best_row["Frame"]
                last_kept_frame = keep_frame
            else:
                keep_frame = None
            for r in current_group:
                r_copy = r.copy()
                if r["Frame"] == keep_frame:
                    filtered_rows.append(r_copy)
                else:
                    r_copy["Hit Player ID"] = np.nan
                    r_copy["Hit Power"] = np.nan
                    filtered_rows.append(r_copy)
        else:
            filtered_rows.extend(current_group)

    final_hits = []
    last_hit_frame_per_player = {}

    for row in filtered_rows:
        player_id = row["Hit Player ID"]
        frame = row["Frame"]

        if pd.notna(player_id) and player_id != -1:
            last_frame = last_hit_frame_per_player.get(player_id, -MIN_FRAME_GAP - 1)
            if frame - last_frame >= MIN_FRAME_GAP:
                final_hits.append(row)
                last_hit_frame_per_player[player_id] = frame
            else:
                row["Hit Player ID"] = np.nan
                row["Hit Power"] = np.nan
                final_hits.append(row)
        else:
            final_hits.append(row)

    final_df = pd.DataFrame(final_hits)
    final_df.to_csv("output/datasets/ball_data.csv", index=False)