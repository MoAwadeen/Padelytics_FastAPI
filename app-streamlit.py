import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Load your data
players_data = pd.read_csv("D:/Personal/G-Project/Padelytics/output/datasets/players_data.csv")
ball_data = pd.read_csv("D:/Personal/G-Project/Padelytics/output/datasets/ball_data.csv")

players = ['player1', 'player2', 'player3', 'player4']
colors = ['blue', 'orange', 'green', 'red']

# Title
st.title("Padelytics Dashboard")

# Sidebar for player selection
selected_player = st.sidebar.selectbox("Select a Player", players)

# Tabs
tabs = st.tabs([
    "Player Movement",
    "Distance & Speed",
    "Zones Analysis",
    "Player Performance",
    "Ball Analysis"
])

# ----------------------------
# Player Movement Tab
# ----------------------------
with tabs[0]:
    st.header("Player Trajectories")
    fig_traj = px.line(players_data, x=f'{selected_player}_y', y=f'{selected_player}_x',
                       labels={'x': 'Court Width (Y)', 'y': 'Court Length (X)'})
    fig_traj.update_yaxes(autorange='reversed')
    fig_traj.update_layout(title=f"Trajectory of {selected_player.capitalize()}")
    st.plotly_chart(fig_traj)

    st.header("Player Position Heatmaps")
    fig_heat = go.Figure()
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds']
    for i, player in enumerate(players):
        fig_heat.add_trace(go.Histogram2dContour(
            x=players_data[f'{player}_y'],
            y=players_data[f'{player}_x'],
            colorscale=colormaps[i],
            contours_coloring='fill',
            opacity=0.5,
            showscale=False,
            name=player
        ))
    fig_heat.update_layout(title="Player Position Heatmaps")
    fig_heat.update_yaxes(autorange='reversed')
    st.plotly_chart(fig_heat)

    st.header("Player Movement by Zone")
    fig_zone = go.Figure()
    zones = {"Attack": (lambda y: (y >= -5) & (y <= 5)),
             "Defense": (lambda y: (y < -5) | (y > 5))}
    colors_zones = {"Attack": "green", "Defense": "red"}
    time = players_data['frame']
    for player in players:
        y_values = players_data[f'{player}_y']
        for zone, cond in zones.items():
            mask = cond(y_values)
            fig_zone.add_trace(go.Scatter(
                x=time[mask],
                y=-y_values[mask],
                mode='markers',
                marker=dict(color=colors_zones[zone], size=5),
                name=f'{player.capitalize()} - {zone}'
            ))
    fig_zone.update_layout(title="Player Movement Over Time by Zone")
    st.plotly_chart(fig_zone)

# ----------------------------
# Distance & Speed Tab
# ----------------------------
with tabs[1]:
    st.header("Total Distance Covered")
    totals = [players_data[f'{p}_distance'].sum() for p in players]
    fig_total_dist = px.bar(x=players, y=totals, labels={'x': 'Player', 'y': 'Total Distance (m)'}, 
                            color=players, color_discrete_sequence=colors)
    fig_total_dist.update_layout(title='Total Distance Covered by Each Player')
    st.plotly_chart(fig_total_dist)

    st.header("Average Distance per Frame")
    avgs = [players_data[f'{p}_distance'].mean() for p in players]
    fig_avg_dist = px.bar(x=players, y=avgs, labels={'x': 'Player', 'y': 'Average Distance per Frame (m)'},
                          color=players, color_discrete_sequence=colors)
    fig_avg_dist.update_layout(title='Average Distance per Frame')
    st.plotly_chart(fig_avg_dist)

    st.header("Average Speed per Player")
    avgs_speed = [players_data[f'{p}_Vnorm1'].mean() for p in players]
    fig_avg_speed = px.bar(x=players, y=avgs_speed, labels={'x': 'Player', 'y': 'Average Speed (units/s)'},
                           color=players, color_discrete_sequence=colors)
    fig_avg_speed.update_layout(title='Average Speed per Player')
    st.plotly_chart(fig_avg_speed)

    st.header("Max Speed per Player")
    maxs_speed = [players_data[f'{p}_Vnorm1'].max() for p in players]
    fig_max_speed = px.bar(x=players, y=maxs_speed, labels={'x': 'Player', 'y': 'Max Speed (units/s)'},
                           color=players, color_discrete_sequence=colors)
    fig_max_speed.update_layout(title='Max Speed per Player')
    st.plotly_chart(fig_max_speed)

    st.header("Average Acceleration per Player")
    avgs_acc = [players_data[f'{p}_Anorm1'].mean() for p in players]
    fig_avg_acc = px.bar(x=players, y=avgs_acc, labels={'x': 'Player', 'y': 'Average Acceleration (units/s²)'},
                         color=players, color_discrete_sequence=colors)
    fig_avg_acc.update_layout(title='Average Acceleration per Player')
    st.plotly_chart(fig_avg_acc)

# ----------------------------
# Zones Analysis Tab
# ----------------------------
with tabs[2]:
    st.header("Time Distribution in Zones")
    attack = [(players_data[f'{p}_y'].between(-5, 5)).sum() / len(players_data) * 100 for p in players]
    defense = [100 - a for a in attack]
    fig_zone_dist = go.Figure()
    fig_zone_dist.add_trace(go.Bar(x=players, y=attack, name='Attack Zone', marker_color='green'))
    fig_zone_dist.add_trace(go.Bar(x=players, y=defense, name='Defense Zone', marker_color='red'))
    fig_zone_dist.update_layout(barmode='stack', title='Percentage of Time in Attack vs Defense Zones')
    st.plotly_chart(fig_zone_dist)

# ----------------------------
# Player Performance Tab
# ----------------------------
with tabs[3]:
    st.header("Player Performance Radar Chart")
    metrics = {
        'Avg Speed': [players_data[f'{p}_Vnorm1'].mean() for p in players],
        'Max Speed': [players_data[f'{p}_Vnorm1'].max() for p in players],
        'Acceleration': [players_data[f'{p}_Anorm1'].mean() for p in players],
        'Attack Zone %': [(players_data[f'{p}_y'].between(-5, 5)).sum() / len(players_data) * 100 for p in players],
        'Distance': [players_data[f'{p}_distance'].sum() for p in players]
    }
    df_metrics = pd.DataFrame(metrics, index=players)
    fig_radar = go.Figure()
    for player in players:
        fig_radar.add_trace(go.Scatterpolar(
            r=df_metrics.loc[player],
            theta=df_metrics.columns,
            fill='toself',
            name=player
        ))
    fig_radar.update_layout(title='Player Performance Radar Chart', polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_radar)

# ----------------------------
# Ball Analysis Tab
# ----------------------------
with tabs[4]:
    st.header("Ball Trajectory")
    fig_ball_traj = px.line(ball_data, x='Ball X', y='Ball Y')
    fig_ball_traj.update_layout(title='Ball Trajectory')
    st.plotly_chart(fig_ball_traj)

    st.header("Ball Speed Over Time")
    fig_ball_speed = px.line(ball_data, x='Frame', y='Speed')
    fig_ball_speed.update_layout(title='Ball Speed Over Time')
    st.plotly_chart(fig_ball_speed)

    st.header("Number of Hits per Player")
    hit_counts = ball_data['Hit Player ID'].dropna().astype(int).value_counts().sort_index()
    fig_hits = px.bar(x=[f'Player {i}' for i in hit_counts.index], y=hit_counts.values,
                      labels={'x': 'Player', 'y': 'Hit Count'})
    fig_hits.update_layout(title='Number of Hits per Player')
    st.plotly_chart(fig_hits)

    st.header("Hit Locations")
    hit_positions = ball_data.dropna(subset=['Hit Player ID'])
    hit_positions['Ball Y'] = -hit_positions['Ball Y']
    fig_hit_locations = px.scatter(hit_positions, x='Ball X', y='Ball Y', color=hit_positions['Hit Player ID'].astype(str))
    fig_hit_locations.update_layout(title='Ball Hit Locations')
    st.plotly_chart(fig_hit_locations)

    st.header("Top Strongest Hits")
    top_hits = ball_data.dropna(subset=['Speed', 'Hit Player ID']).nlargest(2, 'Speed')
    fig_top_hits = px.bar(
        x=[f'Player {int(pid)}' for pid in top_hits['Hit Player ID']],
        y=top_hits['Speed'],
        labels={'x': 'Player', 'y': 'Speed (m/s)'},
        color_discrete_sequence=['red', 'orange']
    )
    fig_top_hits.update_layout(title='Top 2 Strongest Ball Hits')
    st.plotly_chart(fig_top_hits)
