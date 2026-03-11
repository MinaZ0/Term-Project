import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from autogluon.tabular import TabularPredictor
import plotly.express as px

# โหลด Model
predictor = TabularPredictor.load("ag_heart_model")

app = dash.Dash(__name__)

app.layout = html.Div(className='container', children=[
    html.Div(className='header', children=[
        html.H1("Heart Disease AI Diagnostic Tool"),
        html.P("กรอกข้อมูลสุขภาพเพื่อวิเคราะห์ความเสี่ยงโรคหัวใจด้วยระบบ AI")
    ]),

    html.Div(className='content-grid', children=[
        # Left Column: Input Grouping
        html.Div(className='input-panel', children=[
            
            # Group 1: ข้อมูลประชากร
            html.Div(className='group-box', children=[
                html.H3("1. Demographic Info"),
                html.Label("อายุ (Age)"),
                dcc.Slider(id='age', min=20, max=85, step=1, value=50, 
                           marks={20:'20', 40:'40', 60:'60', 85:'85'}),
                html.Label("เพศ (Sex)"),
                dcc.Dropdown(id='sex', options=[{'label':'ชาย','value':1},{'label':'หญิง','value':0}], value=1),
            ]),

            # Group 2: ผลการตรวจร่างกาย
            html.Div(className='group-box', children=[
                html.H3("2. Clinical Measurements"),
                html.Div(className='input-row', children=[
                    html.Div([html.Label("ความดัน (Trestbps)"), dcc.Input(id='trestbps', type='number', value=120)]),
                    html.Div([html.Label("คอเลสเตอรอล (Chol)"), dcc.Input(id='chol', type='number', value=200)]),
                ]),
                html.Label("อาการเจ็บหน้าอก (Chest Pain Type)"),
                dcc.Dropdown(id='cp', options=[
                    {'label':'Typical Angina','value':1}, {'label':'Atypical','value':2},
                    {'label':'Non-anginal','value':3}, {'label':'Asymptomatic','value':4}], value=3),
            ]),

            # Group 3: ผลตรวจไฟฟ้าหัวใจ
            html.Div(className='group-box', children=[
                html.H3("3. Electrocardiographic Results"),
                html.Label("อัตราการเต้นหัวใจสูงสุด (Thalach)"),
                dcc.Input(id='thalach', type='number', value=150, style={'width':'100%'}),
                html.Label("ST Depression (Oldpeak)"),
                dcc.Slider(id='oldpeak', min=0, max=6, step=0.1, value=1.0),
            ]),

            html.Button('วิเคราะห์ผลความเสี่ยง', id='predict-btn', n_clicks=0, className='btn-main')
        ]),

        # Right Column: Results & Visualization
        html.Div(className='result-panel', children=[
            html.Div(id='result-output'),
            dcc.Graph(id='importance-graph')
        ])
    ])
])

@app.callback(
    [Output('result-output', 'children'),
     Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('age', 'value'), State('sex', 'value'), State('cp', 'value'),
     State('trestbps', 'value'), State('chol', 'value'), State('thalach', 'value'), 
     State('oldpeak', 'value')]
)
def predict_heart_disease(n, age, sex, cp, trestbps, chol, thalach, oldpeak):
    # เตรียมข้อมูล (Default ค่าที่เหลือ)
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, 0, 0, thalach, 0, oldpeak, 1, 0, 3]], 
                             columns=predictor.feature_metadata_in.get_features())
    
    # พยากรณ์
    prob = predictor.predict_proba(input_df).iloc[0, 1]
    
    # สร้างข้อความแสดงผล
    risk_color = "#e74c3c" if prob > 0.5 else "#2ecc71"
    risk_text = "มีความเสี่ยงสูง" if prob > 0.5 else "ความเสี่ยงต่ำ"
    
    res_html = html.Div(className='card', style={'borderTop': f'10px solid {risk_color}'}, children=[
        html.H2(risk_text, style={'color': risk_color}),
        html.H1(f"{prob*100:.1f}%", style={'fontSize': '50px'}),
        html.P("โอกาสในการพบโรคหัวใจจากการประมวลผล")
    ])

    # กราฟ Feature Importance (จุดเสริมคะแนน)
    importance = predictor.feature_importance(input_df.iloc[:1]) # จำลองจาก input
    fig = px.bar(importance, orientation='h', title="ปัจจัยที่ส่งผลต่อสุขภาพของคุณ",
                 labels={'value': 'ระดับความสำคัญ', 'index': 'ตัวแปร'})
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return res_html, fig

if __name__ == '__main__':
    app.run(debug=True)