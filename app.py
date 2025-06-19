import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

map_fig = go.Figure(go.Scattergeo())

map_fig.update_layout(
    title_text="Sites Map",
    geo=dict(
        projection_type='orthographic',  
        showland=True,
        landcolor="rgb(217, 217, 217)",
        showocean=True,
        oceancolor="rgb(204, 229, 255)",
        bgcolor='rgba(0,0,0,0)', 
    ),
    margin={"r":0,"t":40,"l":0,"b":0}, 
)


series_fig = go.Figure()

series_fig.update_layout(
    title_text="Data products series",
    xaxis_title="Time",
    yaxis_title="Value",
    template="plotly_white", 
    yaxis=dict(
        tickmode='array',
        tickvals=[1, 2, 3], 
        ticktext=['BRAZ', 'SCRZ', 'AREQ'] 
    )
)


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id='globe-graph',
                        figure=map_fig
                    ),
                    width=6  
                ),
                dbc.Col(
                    dcc.Graph(
                        id='series-graph',
                        figure=series_fig
                    ),
                    width=6  
                ),
            ],
            className="mb-4",
        ),

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        dbc.Button("Да", color="success", className="me-3"), 
                        dbc.Button("Нет", color="danger"), 
                    ],
                    className="d-flex justify-content-center" 
                ),
                width=12 
            )
        )
    ],
    fluid=True 
)

if __name__ == '__main__':
    app.run(debug=True)