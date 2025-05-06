import json
import pandas as pd

def analyze_player_performance(players_file_path, ball_file_path, players, output_file='player_analysis.json'):
    # Load the player and ball data
    players_df = pd.read_csv(players_file_path)
    players_df.rename(columns={"frame": "Frame"}, inplace=True)
    ball_df = pd.read_csv(ball_file_path)
    ball_df.rename(columns={"frame": "Frame"}, inplace=True)

    # Define data
    players_data = players_df.copy()
    ball_data = ball_df.copy()

    # Flip Y for correct top-down view (do this once)
    for player in players:
        players_data[f'{player}_y'] = -players_data[f'{player}_y']

    # Initialize output dictionary with specified key order
    output_data = {
        "trajectories": {},
        "animation": [],
        "heatmaps": {},
        "distance_total": {},
        "distance_avg_per_frame": {},
        "average_speed": {},
        "max_speed": {},
        "average_acceleration": {},
        "zone_presence_percentages": {},
        "radar_performance": {},
        "ball_trajectory": {},
        "ball_speed_over_time": {},
        "hit_count_per_player": {},
        "ball_hit_locations": {},
        "top_3_strongest_hits": [],
        "role": {},
        "role_advice": {},
        "reaction_time_efficiency": {},
        "reaction_advice": {},
        "shot_effectiveness": {},
        "shot_advice": {},
        "team_hits": {"Team1": 0, "Team2": 0},
        "player_contribution": {},
        "player_contribution_advice": {},
        "stamina_drop_time": {},
        "stamina_advice": {}
    }

    # Trajectories
    output_data["trajectories"] = {
        player: {
            "x": players_data[f"{player}_x"].tolist(),
            "y": players_data[f"{player}_y"].tolist()  # Already flipped
        }
        for player in players
    }

    # Animation data
    output_data["animation"] = [
        {
            "Frame": int(players_data['Frame'].iloc[i]),
            **{player: {
                "x": float(players_data[f"{player}_x"].iloc[i]),
                "y": float(players_data[f"{player}_y"].iloc[i])
            } for player in players}
        }
        for i in range(len(players_data))
    ]

    # Heatmaps
    output_data["heatmaps"] = {
        player: {
            "x": players_data[f"{player}_x"].tolist(),
            "y": players_data[f"{player}_y"].tolist()
        }
        for player in players
    }

    # Distance and speed metrics
    output_data["distance_total"] = {
        player: float(players_data[f'{player}_distance'].sum())
        for player in players
    }
    output_data["distance_avg_per_frame"] = {
        player: float(players_data[f'{player}_distance'].mean())
        for player in players
    }
    output_data["average_speed"] = {
        player: float(players_data[f'{player}_Vnorm1'].mean())
        for player in players
    }
    output_data["max_speed"] = {
        player: float(players_data[f'{player}_Vnorm1'].max())
        for player in players
    }
    output_data["average_acceleration"] = {
        player: float(players_data[f'{player}_Anorm1'].mean())
        for player in players
    }

    # Zone presence
    zones = {
        "Attack Zone": lambda y: (y >= -5) & (y <= 5),
        "Defense Zone": lambda y: (y < -5) | (y > 5),
    }
    output_data["zone_presence_percentages"] = {zone: {player: 0 for player in players} for zone in zones}
    total_frames = len(players_data)

    for player in players:
        y_positions = players_data[f'{player}_y']
        for zone_name, condition in zones.items():
            count = condition(y_positions).sum()
            percentage = (count / total_frames) * 100
            output_data["zone_presence_percentages"][zone_name][player] = round(percentage, 2)

    # Radar performance
    metrics = {
        "Avg Speed": [players_data[f"{p}_Vnorm1"].mean() for p in players],
        "Max Speed": [players_data[f"{p}_Vnorm1"].max() for p in players],
        "Acceleration": [players_data[f"{p}_Anorm1"].mean() for p in players],
        "Attack Zone %": [
            ((players_data[f"{p}_y"].between(-5, 5)).sum() / len(players_data)) * 100 for p in players
        ],
        "Distance": [players_data[f"{p}_distance"].sum() for p in players],
    }
    output_data["radar_performance"] = {
        "metrics": list(metrics.keys()),
        "players": {
            player: {
                metric: round(metrics[metric][i], 2)
                for metric in metrics
            }
            for i, player in enumerate(players)
        }
    }

    # Ball trajectory
    output_data["ball_trajectory"] = {
        "x": ball_data["Ball X"].tolist(),
        "y": ball_data["Ball Y"].tolist()
    }

    # Ball speed over time
    output_data["ball_speed_over_time"] = {
        "frame": ball_data["Frame"].tolist(),
        "speed": ball_data["Speed"].tolist()
    }

    # Hit count per player
    valid_hits = ball_data.dropna(subset=['Hit Player ID']).copy()
    valid_hits['Hit Player ID'] = valid_hits['Hit Player ID'].astype(int)
    hit_counts = valid_hits['Hit Player ID'].value_counts().sort_index()
    output_data["hit_count_per_player"] = {
        f"player{pid}": int(count) for pid, count in hit_counts.items()
    }

    # Ball hit locations
    hit_positions = ball_data.dropna(subset=['Hit Player ID']).copy()
    hit_positions['Hit Player ID'] = hit_positions['Hit Player ID'].astype(int)
    output_data["ball_hit_locations"] = {
        f"player{pid}": {
            "x": hit_positions[hit_positions["Hit Player ID"] == pid]["Ball X"].tolist(),
            "y": hit_positions[hit_positions["Hit Player ID"] == pid]["Ball Y"].tolist()
        } for pid in [1, 2, 3, 4]
    }

    # Top 3 strongest hits
    top_3_hits = valid_hits.nlargest(3, 'Speed')[['Hit Player ID', 'Speed']]
    output_data["top_3_strongest_hits"] = [
        {
            "player": f"player{int(row['Hit Player ID'])}",
            "speed": round(row["Speed"], 3)
        } for _, row in top_3_hits.iterrows()
    ]

    for player in players:
        player_id = int(player[-1])
        player_key = f"player{player_id}"

        avg_speed = players_data[f'{player}_Vnorm1'].mean()
        avg_acceleration = players_data[f'{player}_Anorm1'].mean()
        hit_count = ball_data[ball_data['Hit Player ID'] == player_id].shape[0]

        # Classifying roles
        if avg_speed > 5 and avg_acceleration > 0.5 and hit_count > 20:
            role = 'Aggressor'
        elif avg_speed <= 5 and avg_acceleration <= 0.5 and hit_count < 20:
            role = 'Defensive Anchor'
        else:
            role = 'Support'

        output_data["role"][player_key] = role

        # Role-based advice
        if role == 'Aggressor':
            role_advice = "Maintain aggressive positioning to exploit fast breaks and keep pressure on the opposition."
        elif role == 'Defensive Anchor':
            role_advice = "Focus on maintaining defensive positioning and providing cover for teammates."
        else:
            role_advice = "Stay flexible to switch between attacking and defending as needed, providing support where required."

        output_data["role_advice"][player_key] = role_advice

    valid_hits = ball_data[(ball_data['Closest Player ID'] != -1) & (ball_data['Hit Player ID'] != -1)]

    for player in players:
        player_id = int(player[-1])
        player_key = f"player{player_id}"

        player_closest = valid_hits[valid_hits['Closest Player ID'] == player_id]
        player_hits = player_closest[player_closest['Hit Player ID'] == player_id]

        efficiency = (player_hits.shape[0] / player_closest.shape[0]) * 100 if player_closest.shape[0] > 0 else 0
        output_data["reaction_time_efficiency"][player_key] = round(efficiency, 2)

        if efficiency > 80:
            reaction_advice = "Excellent reaction time! Keep up the good work."
        else:
            reaction_advice = "Reaction time can be improved. Work on improving anticipation and positioning."

        output_data["reaction_advice"][player_key] = reaction_advice

    for player in players:
        player_id = int(player[-1])
        player_key = f"player{player_id}"

        player_hits = ball_data[ball_data['Hit Player ID'] == player_id]
        hit_power = player_hits['Hit Power'].values
        speed = player_hits['Speed'].values

        valid_data = pd.DataFrame({'Hit Power': hit_power, 'Speed': speed})
        valid_data = valid_data.dropna()
        valid_data = valid_data[valid_data['Hit Power'] != -1]
        valid_data = valid_data[valid_data['Speed'] != -1]

        if valid_data.shape[0] > 1:
            correlation = valid_data.corr().loc['Hit Power', 'Speed']
            output_data["shot_effectiveness"][player_key] = round(correlation, 2)
        else:
            output_data["shot_effectiveness"][player_key] = None

        if output_data["shot_effectiveness"][player_key] is not None:
            if output_data["shot_effectiveness"][player_key] > 0.8:
                advice = "Excellent shot effectiveness! Your power is contributing well to ball speed."
            elif output_data["shot_effectiveness"][player_key] < 0.3:
                advice = "Low shot effectiveness. Work on improving your shot technique for better ball speed."
            else:
                advice = "Moderate shot effectiveness. A little more focus on shot power could help increase speed."
        else:
            advice = "No shot data available for analysis."

        output_data["shot_advice"][player_key] = advice

    valid_hits = ball_data[ball_data['Hit Player ID'] != -1].dropna(subset=['Hit Player ID'])
    hit_counts = valid_hits['Hit Player ID'].value_counts()

    team_1_players = [1, 2]
    team_2_players = [3, 4]

    team_1_hits = hit_counts[team_1_players].sum() if any(pid in hit_counts for pid in team_1_players) else 0
    team_2_hits = hit_counts[team_2_players].sum() if any(pid in hit_counts for pid in team_2_players) else 0

    output_data["team_hits"]["Team1"] = int(team_1_hits)
    output_data["team_hits"]["Team2"] = int(team_2_hits)

    for player_id in team_1_players:
        player_key = f"player{player_id}"
        if player_id in hit_counts and team_1_hits > 0:
            player_hits = hit_counts[player_id]
            player_contribution = (player_hits / team_1_hits) * 100
            output_data["player_contribution"][player_key] = round(player_contribution, 2)
        else:
            output_data["player_contribution"][player_key] = 0.0

    for player_id in team_2_players:
        player_key = f"player{player_id}"
        if player_id in hit_counts and team_2_hits > 0:
            player_hits = hit_counts[player_id]
            player_contribution = (player_hits / team_2_hits) * 100
            output_data["player_contribution"][player_key] = round(player_contribution, 2)
        else:
            output_data["player_contribution"][player_key] = 0.0

    # Define generate_contribution_advice before calling it
    def generate_contribution_advice(contribution_pct, role, player_key, output_data):
        # Initialize player_contribution_advice if not present
        if "player_contribution_advice" not in output_data:
            output_data["player_contribution_advice"] = {}

        if role == 'Aggressor':
            if contribution_pct >= 50:
                advice = "Outstanding aggression! Your high impact drives the team's offense. Keep pushing fast breaks and pressuring opponents."
            elif 30 <= contribution_pct < 50:
                advice = "Solid aggressive play! You're influencing the game well. Sharpen your attacking moves to dominate even more."
            elif 15 <= contribution_pct < 30:
                advice = "Decent effort, but your aggression needs more bite. Focus on timing your attacks to disrupt the opposition."
            elif 5 <= contribution_pct < 15:
                advice = "Your aggressive play is too quiet. Step up by initiating more plays and pressuring opponents consistently."
            else:
                advice = "Minimal impact as an Aggressor. Work on explosive movements and confidence in attacking to boost your presence."

        elif role == 'Defensive Anchor':
            if contribution_pct >= 50:
                advice = "Rock-solid defending! You're a wall, shutting down threats. Keep anchoring and guiding the backline."
            elif 30 <= contribution_pct < 50:
                advice = "Strong defensive work! You're holding the line well. Improve positioning to block more plays."
            elif 15 <= contribution_pct < 30:
                advice = "You're defending, but need more authority. Focus on reading plays early to intercept threats."
            elif 5 <= contribution_pct < 15:
                advice = "Your defensive presence is weak. Strengthen your positioning and communication to anchor effectively."
            else:
                advice = "Barely noticeable as a Defensive Anchor. Train on reading opponents and maintaining coverage to solidify your role."

        else:  # Flexible/Support role
            if contribution_pct >= 50:
                advice = "Diverse Excellence! You're seamlessly switching roles, massively impacting both attack and defense."
            elif 30 <= contribution_pct < 50:
                advice = "Great flexibility! You're supporting well across roles. Refine transitions to boost your overall influence."
            elif 15 <= contribution_pct < 30:
                advice = "You're contributing, but your flexibility needs work. Practice switching between attack and defense faster."
            elif 5 <= contribution_pct < 15:
                advice = "Your support is minimal. Take more initiative in both attacking and defending to be a true all-rounder."
            else:
                advice = "Very low impact in a flexible role. Focus on readiness and adaptability to contribute meaningfully."

        # Assign advice to output_data
        output_data["player_contribution_advice"][player_key] = advice
        return output_data

    # Generate contribution advice for each player
    for player in players:
        player_id = int(player[-1])
        player_key = f"player{player_id}"
        contribution_pct = output_data["player_contribution"][player_key]
        role = output_data["role"][player_key]
        output_data = generate_contribution_advice(contribution_pct, role, player_key, output_data)

    ball_data = ball_data.dropna(subset=['Speed'])
    ball_data['Time (s)'] = ball_data['Frame'] / 25
    ball_data['Time (min)'] = ball_data['Time (s)'] / 60

    for player_id in ball_data['Hit Player ID'].unique():
        if pd.isna(player_id) or player_id == -1:
            continue

        player_key = f"player{int(player_id)}"
        player_data = ball_data[ball_data['Hit Player ID'] == player_id]

        speed_diff = player_data['Speed'].diff()
        threshold = -0.1

        drop_points = player_data[speed_diff < threshold]

        if not drop_points.empty:
            first_drop_time = drop_points['Time (min)'].iloc[0]
            output_data["stamina_drop_time"][player_key] = round(first_drop_time, 2)
            if first_drop_time > 30:
                advice = "Good stamina!."
            else:
                advice = "Practice stamina under match-like conditions."
        else:
            output_data["stamina_drop_time"][player_key] = None
            advice = "No significant stamina drop."

        output_data["stamina_advice"][player_key] = advice

    # Ensure all players are included for all player-specific metrics
    for player in players:
        player_id = int(player[-1])
        player_key = f"player{player_id}"

        # Visualization defaults
        if player_key not in output_data["trajectories"]:
            output_data["trajectories"][player_key] = {"x": [], "y": []}
        if player_key not in output_data["heatmaps"]:
            output_data["heatmaps"][player_key] = {"x": [], "y": []}
        if player_key not in output_data["distance_total"]:
            output_data["distance_total"][player_key] = 0.0
        if player_key not in output_data["distance_avg_per_frame"]:
            output_data["distance_avg_per_frame"][player_key] = 0.0
        if player_key not in output_data["average_speed"]:
            output_data["average_speed"][player_key] = 0.0
        if player_key not in output_data["max_speed"]:
            output_data["max_speed"][player_key] = 0.0
        if player_key not in output_data["average_acceleration"]:
            output_data["average_acceleration"][player_key] = 0.0
        if player_key not in output_data["zone_presence_percentages"]["Attack Zone"]:
            output_data["zone_presence_percentages"]["Attack Zone"][player_key] = 0.0
            output_data["zone_presence_percentages"]["Defense Zone"][player_key] = 0.0
        if player_key not in output_data["radar_performance"]["players"]:
            output_data["radar_performance"]["players"][player_key] = {
                metric: 0.0 for metric in metrics
            }
        if player_key not in output_data["hit_count_per_player"]:
            output_data["hit_count_per_player"][player_key] = 0
        if player_key not in output_data["ball_hit_locations"]:
            output_data["ball_hit_locations"][player_key] = {"x": [], "y": []}

        # Previous analyses defaults
        if player_key not in output_data["role"]:
            output_data["role"][player_key] = "Unknown"
            output_data["role_advice"][player_key] = "No role data available."
        if player_key not in output_data["reaction_time_efficiency"]:
            output_data["reaction_time_efficiency"][player_key] = 0.0
            output_data["reaction_advice"][player_key] = "No reaction data available."
        if player_key not in output_data["shot_effectiveness"]:
            output_data["shot_effectiveness"][player_key] = None
            output_data["shot_advice"][player_key] = "No shot data available for analysis."
        if player_key not in output_data["player_contribution"]:
            output_data["player_contribution"][player_key] = 0.0
        if player_key not in output_data["player_contribution_advice"]:
            output_data["player_contribution_advice"][player_key] = "No contribution data available."
        if player_key not in output_data["stamina_drop_time"]:
            output_data["stamina_drop_time"][player_key] = None
            output_data["stamina_advice"][player_key] = "No significant stamina drop."

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"All analyses saved to '{output_file}'")

    return output_data