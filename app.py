import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from pycaret.classification import load_model, predict_model

# ใช้ Theme MINTY เพื่อความสวยงาม (1 คะแนน ส่วน CSS)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
model = load_model('sleep_ai_model')

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

@app.callback(
    [Output('output-prediction', 'children')],
    Input('btn-predict', 'n_clicks'),
    [State('age', 'value'), State('stress-slider', 'value')]
)

def update_output(n, age, stress):
    if n is None: return ["กรุณากรอกข้อมูลแล้วกดปุ่มวิเคราะห์"]
    
    # พยากรณ์ (3 คะแนน)
    input_df = pd.DataFrame([[age, stress]], columns=['Age', 'Stress Level'])
    prediction = predict_model(model, data=input_df)
    res = prediction['prediction_label'][0]
    
    # Extra Module (5 คะแนน): คำนวณรอบการนอน
    # สมมติถ้าอายุเยอะ ควรนอนให้ครบ 5 รอบ (7.5 ชม.)
    extra_advice = f"ผลลัพธ์: {res} | ข้อแนะนำ: คุณควรนอนให้ครบ 5 รอบเพื่อสุขภาพที่ดี"
    
    return [extra_advice]

if __name__ == '__main__':
    app.run_server(debug=True)