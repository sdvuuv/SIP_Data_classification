import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import data_processing
import json
from pathlib import Path

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks="initial_duplicate")

modal = dbc.Modal(
    [
        dbc.ModalHeader("Выберите параметры для анализа"),
        dbc.ModalBody([
            dbc.Label("Дата исследования:"),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=datetime(2025, 5, 1),
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
        dcc.Store(id='all-segments-store'),
        dcc.Store(id='current-segment-index-store'),
        dcc.Store(id='study-date-store'),
        
        modal,

        dcc.Loading(
            id="loading-container",
            children=html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='globe-graph'), width=7),
                    dbc.Col(dcc.Graph(id='series-graph'), width=5), 
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Div([
                            dbc.Button("Да", id="yes-button", color="success", className="me-3"),
                            dbc.Button("Нет", id="no-button", color="danger"),
                        ], className="d-flex justify-content-center mt-4"),
                        width=12
                    )
                )
            ], id="main-content", style={'display': 'none'}) 
        )
    ],
    fluid=True
)


@app.callback(
    Output('all-segments-store', 'data'),
    Output('current-segment-index-store', 'data'),
    Output('study-date-store', 'data'), 
    Output('settings-modal', 'is_open'),
    Output('main-content', 'style'), 
    Input('submit-modal-button', 'n_clicks'),
    State('date-picker', 'date'),
    prevent_initial_call=True
)
def run_processing_and_init(n_clicks, selected_date):
    if not selected_date:
        return no_update, no_update, no_update, True, {'display': 'none'}
        
    study_date_dt = datetime.strptime(selected_date.split('T')[0], '%Y-%m-%d')
    
    all_segments, study_date_obj = data_processing.get_main_map_data(study_date_dt)
    
    study_date_str = study_date_obj.isoformat()
    
    return all_segments, 0, study_date_str, False, {'display': 'block'}


@app.callback(
    Output('globe-graph', 'figure'),
    Output('series-graph', 'figure'),
    Input('current-segment-index-store', 'data'),
    State('all-segments-store', 'data'),
    State('study-date-store', 'data') 
)
def update_graphs(current_index, all_segments, study_date_str):
    map_fig = go.Figure()
    series_fig = go.Figure()
    map_title = "Карта геофизических эффектов"
    series_title = "Данные временного ряда"

    anomaly_polygon_coords = data_processing.generate_equatorial_poly()
    poly_lons, poly_lats = zip(*anomaly_polygon_coords)
    map_fig.add_trace(go.Scattergeo(
        lon=poly_lons, lat=poly_lats, mode='lines', fill='toself', 
        fillcolor='rgba(255, 0, 0, 0.3)', name='Зона аномалии', showlegend=False
    ))
    
    if all_segments is None or current_index is None or study_date_str is None:
        return no_update, no_update

    if not all_segments:
        map_title = "Не найдено подходящих сегментов"
    elif current_index >= len(all_segments):
        map_title = "Маркировка завершена!"
        series_fig.update_layout(title="Нет данных для отображения")
    else:
        segment_to_display = all_segments[current_index]
        station_id = segment_to_display['station_id']
        map_title = f"Сегмент на проверке: {segment_to_display['id']}"
        
        # Отображение станции
        site = data_processing.get_site_data_by_id(station_id)
        if site:
            map_fig.add_trace(go.Scattergeo(
                lon=[site['lon']], lat=[site['lat']], text=site['id'].upper(), mode='markers+text', 
                marker=dict(color='blue', size=10, symbol='triangle-up'), textposition='top right', 
                name=f"Станция {site['id'].upper()}"
            ))

        # Отображение траектории
        points = segment_to_display.get('points', [])
        lats = [p['lat'] for p in points]; lons = [p['lon'] for p in points]
        map_fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, mode='lines', 
            line=dict(width=3, color='purple'), name=segment_to_display['id']
        ))
        
        intersections = segment_to_display.get('intersections', [])
        if intersections:
            int_lats = [p['lat'] for p in intersections]
            int_lons = [p['lon'] for p in intersections]
            
            hover_texts = []
            for i, p in enumerate(intersections):
                event_type = "Вход" if i % 2 == 0 else "Выход"
                event_num = i // 2 + 1
                time_obj = datetime.fromisoformat(p['time'])
                time_str = time_obj.strftime('%H:%M:%S')
                hover_texts.append(f"{event_type} {event_num}<br>Время: {time_str}")

            map_fig.add_trace(go.Scattergeo(
                lon=int_lons,
                lat=int_lats,
                mode='markers',
                marker=dict(
                    color='lime',
                    size=8,
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                hoverinfo='text',
                text=hover_texts,
                name='Точки пересечения'
            ))

        study_date = datetime.fromisoformat(study_date_str)
        series_data = data_processing.get_series_data_for_trajectory(
            study_date=study_date,
            station_id=segment_to_display['station_id'],
            satellite_id=segment_to_display['satellite_id'],
            product=data_processing.DataProducts.roti
        )
        if series_data:
            series_title = f"ROTI для {segment_to_display['id']}"
            times_dt = [datetime.fromisoformat(t) for t in series_data['time']]
            
            series_fig.add_trace(go.Scatter(
                x=times_dt, y=series_data['value'], mode='lines+markers',
                name=series_data['product_name']
            ))
            series_fig.update_layout(yaxis_title=f"{series_data['product_name']}, {series_data['product_units']}")
            
            if intersections:
                for i, intersection in enumerate(intersections):
                    is_entry = (i % 2 == 0)
                    line_color = "green" if is_entry else "red"
                    event_number = i // 2 + 1
                    annotation_text = f"Вход {event_number}" if is_entry else f"Выход {event_number}"
                    intersection_time_dt = datetime.fromisoformat(intersection['time'])
                    intersection_timestamp = intersection_time_dt.timestamp() * 1000
                    
                    series_fig.add_vline(
                        x=intersection_timestamp, line_width=2, line_dash="dash", line_color=line_color,
                        annotation_text=annotation_text, annotation_position="top left" if is_entry else "top right"
                    )

                series_fig.update_xaxes(
                    title="Время (UTC)", type="date", tickformat="%Y-%m-%d\n%H:%M:%S"
                )
        else:
            series_title = f"Нет данных для {segment_to_display['id']}"

    map_fig.update_layout(
        title=map_title, width=None, height=600,
        geo=dict(projection_type='orthographic', lonaxis_showgrid=True, lataxis_showgrid=True)
    )
    series_fig.update_layout(
        title=series_title, xaxis_title="Время (UTC)", width=None, height=600
    )
    
    return map_fig, series_fig

@app.callback(
    Output('current-segment-index-store', 'data', allow_duplicate=True),
    [Input('yes-button', 'n_clicks'),
     Input('no-button', 'n_clicks')],
    [State('current-segment-index-store', 'data'),
     State('all-segments-store', 'data')],
    prevent_initial_call=True
)
def process_annotation(yes_clicks, no_clicks, current_index, all_segments):
    if not any([yes_clicks, no_clicks]):
        return no_update

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    is_effect = (button_id == 'yes-button')
    
    if all_segments and current_index < len(all_segments):
        segment_to_save = all_segments[current_index]
        segment_to_save['is_effect'] = is_effect
        
        output_dir = Path("annotations")
        output_dir.mkdir(exist_ok=True)
        
        date_str = datetime.fromisoformat(segment_to_save['points'][0]['time']).strftime('%Y-%m-%d')
        file_path = output_dir / f"annotations_{date_str}.json"

        with open(file_path, 'a') as f:
            json_line = json.dumps(segment_to_save)
            f.write(json_line + '\n')
            
        print(f"Сегмент {segment_to_save['id']} сохранен в {file_path} с is_effect={is_effect}")

    return current_index + 1


if __name__ == '__main__':
    app.run(debug=True)