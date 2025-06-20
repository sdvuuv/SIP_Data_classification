import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import data_processing

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

modal = dbc.Modal(
    [
        dbc.ModalHeader("Выберите параметры для анализа"),
        dbc.ModalBody([
            dbc.Label("Дата исследования:"),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=datetime(2020, 1, 1),
                max_date_allowed=datetime.now(),
                display_format='YYYY-MM-DD',
                className="w-100"
            ),
            dbc.Label("Структура данных:", className="mt-3"),
            dcc.Dropdown(
                id='structure-dropdown',
                options=[{'label': 'Структура А', 'value': 'A'}],
                value='A'
            ),
            dbc.Alert("Пожалуйста, выберите дату.", color="warning", id="date-alert", is_open=False, duration=3000),
        ]),
        dbc.ModalFooter(
            dbc.Button("Загрузить и отобразить", id="submit-modal-button", color="primary")
        ),
    ],
    id="settings-modal", is_open=True, backdrop="static", keyboard=False,
)

app.layout = dbc.Container(
    [
        dcc.Store(id='map-data-store'),

        modal,

        dcc.Loading(
            id="loading-container",
            type="default",
            children=html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='globe-graph'), width=6),
                    dbc.Col(dcc.Graph(id='series-graph'), width=6), 
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Div([
                            dbc.Button("Да", id="yes-button", color="success", className="me-3"),
                            dbc.Button("Нет", id="no-button", color="danger"),
                        ], className="d-flex justify-content-center mt-4"),
                        width=12
                    ),
                    className="align-items-end"
                )
            ], id="main-content", style={'display': 'none'}) 
        )
    ],
    fluid=True
)



@app.callback(
    Output('map-data-store', 'data'),
    Output('settings-modal', 'is_open'),
    Output('date-alert', 'is_open'),
    Output('main-content', 'style'), 
    Input('submit-modal-button', 'n_clicks'),
    State('date-picker', 'date'),
    prevent_initial_call=True
)
def run_processing_and_store_data(n_clicks, selected_date):
    """
    Запускает тяжелые вычисления из data_processing.py и сохраняет результат.
    """
    if not selected_date:
        return no_update, True, True, {'display': 'none'}

    print(f"Запрос на обработку данных для даты: {selected_date}")
    study_date = datetime.strptime(selected_date.split('T')[0], '%Y-%m-%d')

    trajectories = data_processing.get_main_map_data(study_date)
    print(f"Обработка завершена. Найдено подходящих траекторий: {len(trajectories) if trajectories else 0}")
    
    anomaly_polygon_coords = [(0, 40), (120, 400), (120, 60), (0, 60)]

    data_to_store = {
        'trajectories': trajectories,
        'anomaly_polygon': anomaly_polygon_coords
    }
    
    return data_to_store, False, False, {'display': 'block'}


@app.callback(
    Output('globe-graph', 'figure'),
    Output('series-graph', 'figure'), 
    Input('map-data-store', 'data')
)
def update_graphs_from_stored_data(stored_data):
    if not stored_data:
        map_fig = go.Figure(go.Scattergeo())
        map_fig.update_layout(title_text="Sites Map")
        series_fig = go.Figure()
        series_fig.update_layout(title_text="Data products series", template="plotly_white")
        return map_fig, series_fig

    trajectories = stored_data.get('trajectories', [])
    anomaly_polygon_coords = stored_data.get('anomaly_polygon', [])

    map_fig = go.Figure()

    if anomaly_polygon_coords:
        poly_lons = [p[0] for p in anomaly_polygon_coords]
        poly_lats = [p[1] for p in anomaly_polygon_coords]
        poly_lons.append(poly_lons[0])
        poly_lats.append(poly_lats[0])
        map_fig.add_trace(go.Scattergeo(
            lon=poly_lons,
            lat=poly_lats,
            mode='lines',
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=2),
            name='Anomaly Area'
        ))

    site = data_processing.get_site_data_by_id('irkj')
    if site:
        map_fig.add_trace(go.Scattergeo(
            lon=[site['lon']],
            lat=[site['lat']],
            text=site['id'].upper(),
            mode='markers+text',
            marker=dict(color='blue', size=10, symbol='triangle-up'),
            textposition='top right',
            name=f"Station {site['id'].upper()}"
        ))

    if trajectories:
        for traj in trajectories:
            points = traj['points']
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]
            map_fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats, mode='lines',
                line=dict(width=2, color='orange'),
                name=f"Trajectory: {traj['id']}"
            ))
    
    map_fig.update_layout(
        title_text="Geophysical Effects Map",
        geo=dict(
            projection_type='orthographic',
            projection_rotation={'lon': 90, 'lat': 50},
            showland=True, landcolor="rgb(217, 217, 217)",
            showocean=True, oceancolor="rgb(204, 229, 255)",
        ),
        margin={"r":0, "t":40, "l":0, "b":0},
        legend=dict(x=0, y=1),
        uirevision='constant' 
    )

    series_fig = go.Figure()
    series_fig.update_layout(title_text="Data products series", template="plotly_white")

    return map_fig, series_fig


if __name__ == '__main__':
    app.run(debug=True)