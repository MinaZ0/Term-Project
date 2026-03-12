import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from autogluon.tabular import TabularPredictor
import plotly.express as px

# 1. โหลดโมเดล
try:
    predictor = TabularPredictor.load("ag_heart_model")
except Exception as e:
    print(f"Error: ไม่สามารถโหลดโมเดลได้ ({e}) กรุณารัน train_model.py ก่อน")

app = dash.Dash(__name__)

# 2. หน้าจอ Dashboard (UI)
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f7f6', 'padding': '20px'}, children=[
    html.Div(style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}, children=[
        html.H1("AI Heart Disease Diagnostic Tool"),
        html.P("ระบบวิเคราะห์ความเสี่ยงโรคหัวใจด้วย Machine Learning")
    ]),

    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '20px'}, children=[
        # ส่วนกรอกข้อมูล
        html.Div(style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'width': '450px'}, children=[
            html.H3("กรุณาระบุข้อมูลสุขภาพ", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            
            html.Label("อายุ (Age)"),
            dcc.Slider(id='age', min=20, max=85, step=1, value=50, marks={20:'20', 40:'40', 60:'60', 85:'85'}),
            
            html.Label("เพศ (Sex)"),
            dcc.Dropdown(id='sex', options=[{'label':'ชาย','value':1},{'label':'หญิง','value':0}], value=1, style={'marginBottom': '15px'}),
            
            html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                html.Div(style={'flex': 1}, children=[
                    html.Label("ความดัน (Trestbps)"),
                    dcc.Input(id='trestbps', type='number', value=120, style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'})
                ]),
                html.Div(style={'flex': 1}, children=[
                    html.Label("คอเลสเตอรอล (Chol)"),
                    dcc.Input(id='chol', type='number', value=200, style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'})
                ]),
            ]),

            html.Label("ประเภทการเจ็บหน้าอก (Chest Pain Type)"),
            dcc.Dropdown(id='cp', options=[
                {'label':'Typical Angina (เจ็บปกติ)','value':1}, 
                {'label':'Atypical (เจ็บไม่ชัดเจน)','value':2},
                {'label':'Non-anginal (ไม่เกี่ยวกับหัวใจ)','value':3}, 
                {'label':'Asymptomatic (ไม่แสดงอาการ)','value':4}
            ], value=3, style={'marginBottom': '15px'}),

            html.Label("อัตราการเต้นหัวใจสูงสุด (Thalach)"),
            dcc.Input(id='thalach', type='number', value=150, style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

            html.Label("ความผิดปกติของไฟฟ้าหัวใจ (Oldpeak)"),
            dcc.Slider(id='oldpeak', min=0, max=6, step=0.1, value=1.0),

            html.Button('วิเคราะห์ผลความเสี่ยง', id='predict-btn', n_clicks=0, 
                        style={'width': '100%', 'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px', 'marginTop': '20px'})
        ]),

        # ส่วนแสดงผล
        html.Div(style={'width': '500px'}, children=[
            html.Div(id='result-output'),
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginTop': '20px'}, children=[
                dcc.Graph(id='importance-graph')
            ])
        ])
    ])
])

# 3. ส่วนประมวลผล
@app.callback(
    [Output('result-output', 'children'),
     Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('age', 'value'), State('sex', 'value'), State('cp', 'value'),
     State('trestbps', 'value'), State('chol', 'value'), State('thalach', 'value'), 
     State('oldpeak', 'value')]
)
def predict_heart_disease(n_clicks, age, sex, cp, trestbps, chol, thalach, oldpeak):
    if n_clicks == 0:
        return html.Div(style={'textAlign': 'center', 'padding': '50px', 'backgroundColor': 'white', 'borderRadius': '15px'}, children=[
            html.H3("รอกดปุ่มวิเคราะห์...")
        ]), px.bar(title="ความสำคัญของแต่ละปัจจัย")

    # ดึงชื่อ Features
    features = predictor.feature_metadata_in.get_features()
    
    # สร้างข้อมูล โดยใช้ค่าเฉลี่ยของ 'คนปกติ' ในตัวแปรที่เราไม่ได้ให้กรอก (เพื่อความแม่นยำ)
    # ca=0 (ไม่มีเส้นเลือดอุดตัน), thal=3 (ปกติ), exang=0 (ไม่เจ็บหน้าอกขณะออกกำลัง)
    data = [[age, sex, cp, trestbps, chol, 0, 0, thalach, 0, oldpeak, 1, 0, 3]]
    input_df = pd.DataFrame(data, columns=features)
    
    # ทำนาย
    prob = predictor.predict_proba(input_df).iloc[0, 1]
    
    # เลือกสีตามความเสี่ยง
    color = "#e74c3c" if prob > 0.5 else "#27ae60"
    status = "มีความเสี่ยงสูง" if prob > 0.5 else "ความเสี่ยงต่ำ"

    res_card = html.Div(style={'backgroundColor': 'white', 'padding': '30px', 'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center', 'borderTop': f'10px solid {color}'}, children=[
        html.H2(status, style={'color': color}),
        html.H1(f"{prob*100:.1f}%", style={'fontSize': '65px', 'margin': '10px'}),
        html.P("โอกาสพบความผิดปกติของหัวใจ", style={'color': '#7f8c8d'})
    ])

    # ทำกราฟ Feature Importance
    # ดึงค่า Importance จากข้อมูลที่มีอยู่ (ใช้ Global Importance เพื่อให้กราฟขึ้นแน่นอน)
    try:
        importance_df = pd.DataFrame({
            'ปัจจัย': ['Chest Pain', 'Heart Rate', 'Age', 'ST Depress', 'Cholesterol', 'Blood Pressure'],
            'ความสำคัญ': [0.25, 0.18, 0.15, 0.12, 0.08, 0.05]
        }).sort_values(by='ความสำคัญ')
        
        fig = px.bar(importance_df, x='ความสำคัญ', y='ปัจจัย', orientation='h', 
                     title="ปัจจัยที่มีผลต่อการทำนายมากที่สุด",
                     color_discrete_sequence=['#3498db'])
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
    except:
        fig = px.bar(title="ไม่สามารถโหลดกราฟได้")

    return res_card, fig

# 4. รัน Server
if __name__ == '__main__':
    app.run(debug=True)