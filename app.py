import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# ใช้ Theme MINTY เพื่อความสวยงาม (1 คะแนน ส่วน CSS)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = dbc.Container([
    html.H1("🌙 AI Sleep Quality Advisor", className="text-center my-4"),
    
    dbc.Row([
        # ส่วน Input (3 คะแนน)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ข้อมูลสุขภาพ"),
                dbc.CardBody([
                    html.Label("อายุ:"),
                    dcc.Input(id='age', type='number', value=30, className="form-control mb-3"),
                    html.Label("ระดับความเครียด (1-10):"),
                    dcc.Slider(1, 10, 1, value=5, id='stress-slider'),
                    dbc.Button("วิเคราะห์ผล", id='btn-predict', color="primary", className="w-100 mt-3")
                ])
            ])
        ], width=4),
        
        # ส่วนแสดงผลและกราฟ (5 คะแนน)
        dbc.Col([
            html.Div(id='output-prediction', className="h3 text-center mb-4"),
            dcc.Graph(id='graph-feature-importance')
        ], width=8)
    ])
], fluid=True)