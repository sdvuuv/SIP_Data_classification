import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import data_processing
from data_processing import get_segment_from_cache 
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
    id="settings-modal", is_open=False, backdrop="static", keyboard=False,
)

app.layout = dbc.Container(
    [
        # У нас будет одно главное хранилище для состояния всей сессии
        dcc.Store(id='session-state-store', storage_type='session'),
        
        modal,

        dcc.Loading(
            id="loading-container",
            type="default",
            children=html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='globe-graph'), width=7),
                    dbc.Col(dcc.Graph(id='series-graph'), width=5), 
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Div([
                            # Добавляем disabled=True, чтобы кнопки были неактивны, пока нет сегмента
                            dbc.Button("Эффект есть", id="yes-button", color="success", className="me-3", disabled=True),
                            dbc.Button("Эффекта нет", id="no-button", color="danger", disabled=True),
                        ], className="d-flex justify-content-center mt-4"),
                        width=12
                    )
                )
            ], id="main-content", style={'visibility': 'hidden'}) # Используем visibility вместо display
        )
    ],
    fluid=True,
    id='app-layout'
)

@app.callback(
    Output('settings-modal', 'is_open', allow_duplicate=True),
    Output('main-content', 'style', allow_duplicate=True),
    Input('app-layout', 'children'), # Срабатывает при загрузке layout
    State('session-state-store', 'data')
)
def manage_visibility_on_load(_, session_state):
    # Если сессия уже существует в браузере и не завершена...
    if session_state and not session_state.get('is_finished', False):
        print("Обнаружена существующая сессия. Показываем интерфейс разметки.")
        # ...то мы НЕ открываем модальное окно и показываем основной контент.
        return False, {'visibility': 'visible'}
    else:
        # Иначе (первый запуск или сессия завершена) открываем модальное окно.
        print("Новая сессия. Показываем модальное окно.")
        return True, {'visibility': 'hidden'}

@app.callback(
    Output('session-state-store', 'data', allow_duplicate=True),
    Output('settings-modal', 'is_open', allow_duplicate=True),
    Output('main-content', 'style', allow_duplicate=True),
    Input('submit-modal-button', 'n_clicks'),
    State('date-picker', 'date'),
    prevent_initial_call=True
)
def start_new_session(n_clicks, selected_date):
    """
    Запускается по кнопке в модальном окне.
    1. Получает отфильтрованный список станций (быстро).
    2. Инициализирует состояние сессии в dcc.Store.
    3. Не ищет первый сегмент! Это сделают кнопки "Да/Нет".
    """
    if not selected_date:
        return no_update, True, no_update
        
    study_date_dt = datetime.strptime(selected_date.split('T')[0], '%Y-%m-%d')
    
    # Эта функция должна быть в data_processing.py и работать быстро
    filtered_stations = data_processing.get_filtered_stations(study_date_dt)
    
    # Инициализируем начальное состояние "конвейера"
    initial_state = {
        'study_date': study_date_dt.isoformat(),
        'station_list': filtered_stations,
        'current_station_idx': 0,
        'current_sat_idx': 0,
        'current_segment_data': None, # Пока нет сегмента для отображения
        'is_finished': False
    }
    
    # Закрываем модальное окно и показываем основной интерфейс
    return initial_state, False, {'visibility': 'visible'}

@app.callback(
    Output('session-state-store', 'data', allow_duplicate=True),
    Input('yes-button', 'n_clicks'),
    Input('no-button', 'n_clicks'),
    State('session-state-store', 'data'),
    prevent_initial_call=True
)
def process_annotation_and_find_next(yes_clicks, no_clicks, session_state):
    """
    Запускается по кнопкам "Да/Нет".
    1. Сохраняет разметку для *предыдущего* сегмента (если он был).
    2. Вызывает find_next_valid_segment для поиска *следующего*.
    3. Обновляет состояние сессии в dcc.Store.
    """
    if not session_state:
        return no_update

    # --- Шаг 1: Сохранение разметки для предыдущего сегмента ---
    if session_state.get('current_segment_data'):
        segment_to_annotate = session_state['current_segment_data']
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        is_effect = (button_id == 'yes-button')
        
        # Добавляем разметку к метаданным
        segment_to_annotate['has_effect'] = is_effect
        
        # Логика сохранения в файл
        output_dir = Path("annotations")
        output_dir.mkdir(exist_ok=True)
        date_str = datetime.fromisoformat(session_state['study_date']).strftime('%Y-%m-%d')
        file_path = output_dir / f"annotations_{date_str}.jsonl"
        with open(file_path, 'a') as f:
            f.write(json.dumps(segment_to_annotate) + '\n')
        print(f"Сегмент {segment_to_annotate['id']} сохранен с is_effect={is_effect}")

    # --- Шаг 2: Поиск следующего валидного сегмента ---
    print("\nИщу следующий сегмент...")
    next_segment_metadata, next_station_idx, next_sat_idx = data_processing.find_next_valid_segment(
        study_date=datetime.fromisoformat(session_state['study_date']),
        station_list=session_state['station_list'],
        current_station_idx=session_state['current_station_idx'],
        current_sat_idx=session_state['current_sat_idx']
    )

    # --- Шаг 3: Обновление состояния сессии ---
    if next_segment_metadata:
        # Найден новый сегмент
        session_state['current_segment_data'] = next_segment_metadata
        session_state['current_station_idx'] = next_station_idx
        session_state['current_sat_idx'] = next_sat_idx
        session_state['is_finished'] = False
    else:
        # Сегменты закончились!
        print("Разметка для данной даты завершена.")
        session_state['is_finished'] = True
        session_state['current_segment_data'] = None
    
    return session_state

@app.callback(
    Output('globe-graph', 'figure'),
    Output('series-graph', 'figure'),
    Output('yes-button', 'disabled'),
    Output('no-button', 'disabled'),
    Input('session-state-store', 'data')
)
def update_graphs(session_state):
    """
    Полностью "глупый" колбэк. Просто рисует то, что лежит в session_state.
    """
    if not session_state:
        return no_update, no_update, False, False

    map_fig = go.Figure()
    series_fig = go.Figure()
    map_title = "Карта геофизических эффектов"
    series_title = "Данные временного ряда"
    
    anomaly_polygon_coords = list(reversed(data_processing.generate_equatorial_poly()))
    poly_lons, poly_lats = zip(*anomaly_polygon_coords)
    map_fig.add_trace(go.Scattergeo(
        lon=poly_lons, lat=poly_lats, mode='lines', fill='toself', 
        fillcolor='rgba(255, 0, 0, 0.3)', name='Зона аномалии', showlegend=False
    ))
    

    
    segment_to_display_metadata = session_state.get('current_segment_data')
    is_finished = session_state.get('is_finished', False)
    study_date_str = session_state.get('study_date')
    study_date = datetime.fromisoformat(study_date_str) if study_date_str else None

    buttons_disabled = False

    if is_finished:
        map_title = "Маркировка завершена! Выберите новую дату."
        # Можно очистить и файловый кэш при завершении
        # data_processing.clear_geometry_cache()
    elif segment_to_display_metadata:
        # --- Логика отрисовки ОДНОГО сегмента ---
        event_id = segment_to_display_metadata['id']
        
        segment_full_data = get_segment_from_cache(event_id)
        
        if not segment_full_data:
            map_title = f"Ошибка: данные для {event_id} не найдены в кэше. Нажмите 'Нет' для пропуска."

        else:
            event_metadata = segment_to_display_metadata
            event_id = event_metadata['id']
            station_id = event_metadata['station_id']
            satellite_id = event_metadata['satellite_id']
            map_title = f"Сегмент на проверке: {event_id}"

            segment_to_display = get_segment_from_cache(event_id)
            if not segment_to_display:
                map_title = f"Ошибка: данные для сегмента {event_id} не найдены в кэше."
                # Возвращаем карту с ошибкой и пустой график
                map_fig.update_layout(title=map_title, geo=dict(projection_type='orthographic'))
                series_fig.update_layout(title="Ошибка загрузки данных")
                return map_fig, series_fig, False, False
            
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

            series_data = data_processing.get_series_data_for_trajectory(
                study_date=study_date,
                station_id=station_id, # Берем из метаданных
                satellite_id=satellite_id, # Берем из метаданных
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
                    buttons_disabled = False
            else:
                map_title = "Нажмите 'Да' или 'Нет', чтобы найти первый сегмент для разметки"
                buttons_disabled = False

    map_fig.update_layout(
        title=map_title, width=None, height=600,
        geo=dict(projection_type='orthographic', lonaxis_showgrid=True, lataxis_showgrid=True)
    )
    series_fig.update_layout(
        title=series_title, xaxis_title="Время (UTC)", width=None, height=600
    )
    
    return map_fig, series_fig, buttons_disabled, buttons_disabled

if __name__ == '__main__':
    app.run(debug=True)