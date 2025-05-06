import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Load your data
players_data = pd.read_csv("D:/Personal/G-Project/Padelytics/output/datasets/players_data.csv")
ball_data = pd.read_csv("D:/Personal/G-Project/Padelytics/output/datasets/ball_data.csv")

players = ['player1', 'player2', 'player3', 'player4']
colors = ['blue', 'orange', 'green', 'red']

# Initialize the app
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Padelytics Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Player Movement', children=[
            html.Div([
                html.H3("Player Trajectories"),
                dcc.Dropdown(id='trajectory-player', options=[{'label': p, 'value': p} for p in players], value=players[0]),
                dcc.Graph(id='trajectory-graph'),

                html.H3("Player Position Heatmaps"),
                dcc.Graph(id='heatmap-graph'),

                html.H3("Player Movement by Zone"),
                dcc.Graph(id='zone-movement-graph'),
            ], style={'width': '90%', 'margin': 'auto'})
        ]),

        dcc.Tab(label='Distance & Speed', children=[
            html.Div([
                html.H3("Total Distance Covered"),
                dcc.Graph(id='total-distance-graph'),

                html.H3("Average Distance per Frame"),
                dcc.Graph(id='avg-distance-frame-graph'),

                html.H3("Average Speed per Player"),
                dcc.Graph(id='avg-speed-graph'),

                html.H3("Max Speed per Player"),
                dcc.Graph(id='max-speed-graph'),

                html.H3("Average Acceleration per Player"),
                dcc.Graph(id='avg-acceleration-graph')
            ], style={'width': '90%', 'margin': 'auto'})
        ]),

        dcc.Tab(label='Zones Analysis', children=[
            html.Div([
                html.H3("Time Distribution in Zones"),
                dcc.Graph(id='zone-distribution-graph')
            ], style={'width': '90%', 'margin': 'auto'})
        ]),

        dcc.Tab(label='Player Performance', children=[
            html.Div([
                html.H3("Player Performance Radar Chart"),
                dcc.Graph(id='radar-graph')
            ], style={'width': '90%', 'margin': 'auto'})
        ]),

        dcc.Tab(label='Ball Analysis', children=[
            html.Div([
                html.H3("Ball Trajectory"),
                dcc.Graph(id='ball-trajectory-graph'),

                html.H3("Ball Speed Over Time"),
                dcc.Graph(id='ball-speed-graph'),

                html.H3("Number of Hits per Player"),
                dcc.Graph(id='hits-count-graph'),

                html.H3("Hit Locations"),
                dcc.Graph(id='hit-locations-graph'),

                html.H3("Top Strongest Hits"),
                dcc.Graph(id='strongest-hits-graph')
            ], style={'width': '90%', 'margin': 'auto'})
        ])
    ])
])

#####################
# Callbacks Section #
#####################

# Player Trajectories
@app.callback(
    Output('trajectory-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_trajectory(player):
    fig = px.line(players_data, x=f'{player}_y', y=f'{player}_x', labels={'x': 'Court Width (Y)', 'y': 'Court Length (X)'})
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title=f"Trajectory of {player.capitalize()}", height=500)
    return fig

# Heatmaps
@app.callback(
    Output('heatmap-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_heatmap(_):
    fig = go.Figure()
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds']
    for i, player in enumerate(players):
        fig.add_trace(go.Histogram2dContour(
            x=players_data[f'{player}_y'],
            y=players_data[f'{player}_x'],
            colorscale=colormaps[i],
            contours_coloring='fill',
            opacity=0.5,
            showscale=False,
            name=player
        ))
    fig.update_layout(title="Player Position Heatmaps", height=600)
    fig.update_yaxes(autorange='reversed')
    return fig

# Zone Movement
@app.callback(
    Output('zone-movement-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_zone_movement(_):
    fig = go.Figure()
    zones = {"Attack": (lambda y: (y >= -5) & (y <= 5)), "Defense": (lambda y: (y < -5) | (y > 5))}
    colors_zones = {"Attack": "green", "Defense": "red"}
    time = players_data['frame']
    for player in players:
        y_values = players_data[f'{player}_y']
        for zone, cond in zones.items():
            mask = cond(y_values)
            fig.add_trace(go.Scatter(
                x=time[mask],
                y=-y_values[mask],
                mode='markers',
                marker=dict(color=colors_zones[zone], size=5),
                name=f'{player.capitalize()} - {zone}'
            ))
    fig.update_layout(title="Player Movement Over Time by Zone", height=600)
    return fig

# Total Distance Covered
@app.callback(
    Output('total-distance-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_total_distance(_):
    totals = [players_data[f'{p}_distance'].sum() for p in players]
    fig = px.bar(x=players, y=totals, labels={'x': 'Player', 'y': 'Total Distance (m)'}, color=players, color_discrete_sequence=colors)
    fig.update_layout(title='Total Distance Covered by Each Player')
    return fig

# Average Distance per Frame
@app.callback(
    Output('avg-distance-frame-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_avg_distance(_):
    avgs = [players_data[f'{p}_distance'].mean() for p in players]
    fig = px.bar(x=players, y=avgs, labels={'x': 'Player', 'y': 'Average Distance per Frame (m)'}, color=players, color_discrete_sequence=colors)
    fig.update_layout(title='Average Distance per Frame')
    return fig

# Average Speed
@app.callback(
    Output('avg-speed-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_avg_speed(_):
    avgs = [players_data[f'{p}_Vnorm1'].mean() for p in players]
    fig = px.bar(x=players, y=avgs, labels={'x': 'Player', 'y': 'Average Speed (units/s)'}, color=players, color_discrete_sequence=colors)
    fig.update_layout(title='Average Speed per Player')
    return fig

# Max Speed
@app.callback(
    Output('max-speed-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_max_speed(_):
    maxs = [players_data[f'{p}_Vnorm1'].max() for p in players]
    fig = px.bar(x=players, y=maxs, labels={'x': 'Player', 'y': 'Max Speed (units/s)'}, color=players, color_discrete_sequence=colors)
    fig.update_layout(title='Max Speed per Player')
    return fig

# Average Acceleration
@app.callback(
    Output('avg-acceleration-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_avg_acceleration(_):
    avgs = [players_data[f'{p}_Anorm1'].mean() for p in players]
    fig = px.bar(x=players, y=avgs, labels={'x': 'Player', 'y': 'Average Acceleration (units/s²)'}, color=players, color_discrete_sequence=colors)
    fig.update_layout(title='Average Acceleration per Player')
    return fig

# Zone Distribution
@app.callback(
    Output('zone-distribution-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_zone_distribution(_):
    attack = [(players_data[f'{p}_y'].between(-5, 5)).sum() / len(players_data) * 100 for p in players]
    defense = [100 - a for a in attack]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=players, y=attack, name='Attack Zone', marker_color='green'))
    fig.add_trace(go.Bar(x=players, y=defense, name='Defense Zone', marker_color='red'))
    fig.update_layout(barmode='stack', title='Percentage of Time in Attack vs Defense Zones')
    return fig

# Radar Chart
@app.callback(
    Output('radar-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_radar(_):
    metrics = {
        'Avg Speed': [players_data[f'{p}_Vnorm1'].mean() for p in players],
        'Max Speed': [players_data[f'{p}_Vnorm1'].max() for p in players],
        'Acceleration': [players_data[f'{p}_Anorm1'].mean() for p in players],
        'Attack Zone %': [(players_data[f'{p}_y'].between(-5, 5)).sum() / len(players_data) * 100 for p in players],
        'Distance': [players_data[f'{p}_distance'].sum() for p in players]
    }
    df = pd.DataFrame(metrics, index=players)
    fig = go.Figure()
    for player in players:
        fig.add_trace(go.Scatterpolar(r=df.loc[player], theta=df.columns, fill='toself', name=player))
    fig.update_layout(title='Player Performance Radar Chart', polar=dict(radialaxis=dict(visible=True)))
    return fig

# Ball Trajectory
@app.callback(
    Output('ball-trajectory-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_ball_trajectory(_):
    fig = px.line(ball_data, x='Ball X', y='Ball Y')
    fig.update_layout(title='Ball Trajectory')
    return fig

# Ball Speed
@app.callback(
    Output('ball-speed-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_ball_speed(_):
    fig = px.line(ball_data, x='Frame', y='Speed')
    fig.update_layout(title='Ball Speed Over Time')
    return fig

# Hits Count
@app.callback(
    Output('hits-count-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_hits_count(_):
    hit_counts = ball_data['Hit Player ID'].dropna().astype(int).value_counts().sort_index()
    fig = px.bar(x=[f'Player {i}' for i in hit_counts.index], y=hit_counts.values, labels={'x': 'Player', 'y': 'Hit Count'})
    fig.update_layout(title='Number of Hits per Player')
    return fig

# Hit Locations
@app.callback(
    Output('hit-locations-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_hit_locations(_):
    hit_positions = ball_data.dropna(subset=['Hit Player ID'])
    hit_positions.loc[:, 'Ball Y'] = -hit_positions['Ball Y']
    fig = px.scatter(hit_positions, x='Ball X', y='Ball Y', color=hit_positions['Hit Player ID'].astype(str))
    fig.update_layout(title='Ball Hit Locations')
    return fig

# Strongest Hits
@app.callback(
    Output('strongest-hits-graph', 'figure'),
    Input('trajectory-player', 'value')
)
def update_strongest_hits(_):
    top_hits = ball_data.dropna(subset=['Speed', 'Hit Player ID']).nlargest(2, 'Speed')
    fig = px.bar(
        x=[f'Player {int(pid)}' for pid in top_hits['Hit Player ID']],
        y=top_hits['Speed'],
        labels={'x': 'Player', 'y': 'Speed (m/s)'},
        color_discrete_sequence=['red', 'orange']
    )
    fig.update_layout(title='Top 2 Strongest Ball Hits')
    return fig


# Run
if __name__ == '__main__':
    app.run(debug=True)
