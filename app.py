import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
DATA_STRUCTURE_OPTIONS = [
    {'label': 'equatorial anomaly', 'value': 'anomaly'},
    {'label': 'что-то там про ионосферу((', 'value': 'ionosphere'},
    {'label': 'plasma bubbles', 'value': 'bubbles'},
]

# Создание модального окна
modal = dbc.Modal(
    [
        dbc.ModalHeader("Выберите параметры"),
        dbc.ModalBody([
            dbc.Label("Дата:"),
            dcc.DatePickerSingle(
                id='date-picker',
                display_format='YYYY-MM-DD',
                className="w-100",
                date=datetime.now().date() 
            ),
            html.Hr(), 
            dbc.Label("Структура данных:"),
            dcc.Dropdown(
                id='structure-dropdown',
                options=DATA_STRUCTURE_OPTIONS,
                value='anomaly', 
                clearable=False
            ),
        ]),
        dbc.ModalFooter(
            dbc.Button("Подтвердить и сохранить", id="submit-modal-button", color="primary")
        ),
    ],
    id="settings-modal",
    is_open=True,    
    backdrop="static", 
    keyboard=False, 
)

map_fig = go.Figure(go.Scattergeo())
map_fig.update_layout(
    title_text="Sites Map",
    geo=dict(
        projection_type='orthographic',
        showland=True, landcolor="rgb(217, 217, 217)",
        showocean=True, oceancolor="rgb(204, 229, 255)",
    ),
    margin={"r":0,"t":40,"l":0,"b":0},
)

series_fig = go.Figure()
series_fig.update_layout(
    title_text="Data products series",
    xaxis_title="Time",
    yaxis_title="Value",
    template="plotly_white",
    yaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['BRAZ', 'SCRZ', 'AREQ'])
)

app.layout = dbc.Container(
    [
        dcc.Store(id='user-choices-store'),

        modal,

        html.Div(id='output-container', className="text-center my-3"),

        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='globe-graph', figure=map_fig), width=6),
                dbc.Col(dcc.Graph(id='series-graph', figure=series_fig), width=6),
            ],
            className="mb-4",
        ),

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        dbc.Button("Да", id="yes-button", color="success", className="me-3"),
                        dbc.Button("Нет", id="no-button", color="danger"),
                    ],
                    className="d-flex justify-content-center"
                ),
                width=12
            )
        )
    ],
    fluid=True
)

@app.callback(
    Output('user-choices-store', 'data'),         # Сохраняем данные сюда
    Output('settings-modal', 'is_open'),          # Управляем видимостью окна
    Output('output-container', 'children'),       # Выводим сообщение пользователю
    Input('submit-modal-button', 'n_clicks'),     # клик по кнопке
    State('date-picker', 'date'),                 # Получаем значение даты
    State('structure-dropdown', 'value'),         # Получаем значение из списка
    prevent_initial_call=True                     # Не запускать при загрузке
)
def save_user_choices(n_clicks, selected_date, selected_structure):
    """
    Эта функция выполняется при нажатии кнопки "Подтвердить".
    Она сохраняет выбор пользователя в dcc.Store и закрывает окно.
    """
    # Создаем словарь с данными, который будем хранить
    data_to_store = {
        'date': selected_date,
        'structure': selected_structure
    }

    # Создаем сообщение для пользователя
    feedback_message = f"Данные сохранены! Выбрано: Дата - {selected_date}, Структура - '{selected_structure}'"
    print(feedback_message) 

    return data_to_store, False, html.P(feedback_message, className="text-success fw-bold")


if __name__ == '__main__':
    app.run(debug=True)