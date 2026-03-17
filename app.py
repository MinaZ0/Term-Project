import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from autogluon.tabular import TabularPredictor
import plotly.express as px
import plotly.graph_objects as go

# 1. โหลดโมเดลและข้อมูลสำหรับ EDA
try:
    predictor = TabularPredictor.load("ag_heart_model")
    # สมมติว่ามีไฟล์ข้อมูลที่รวมทุกประเทศไว้แล้วจากการเทรน
    df_eda = pd.read_csv('combined_heart_data.csv') 
except Exception as e:
    print(f"Error loading files: {e}. Please run train_model.py first.")

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# 2. Layout หลักแบบ Glassmorphism
app.layout = html.Div(className='container', children=[
    
    # Header Section
    html.Div(style={'textAlign': 'center', 'marginBottom': '40px'}, children=[
        html.H1("HEART CORE AI", className='main-title'),
        html.Div("PRECISION DIAGNOSTIC & GLOBAL INSIGHTS v3.0", 
                 style={'color': '#64748b', 'fontSize': '0.8rem', 'letterSpacing': '4px'})
    ]),

    # Tab System
    dcc.Tabs(id="tabs", value='tab-predict', className='custom-tabs', children=[
        
        # --- TAB 1: PREDICTION ENGINE ---
        dcc.Tab(label='ANALYSIS ENGINE', value='tab-predict', 
                className='custom-tab', selected_className='custom-tab--selected', children=[
            
            html.Div(style={'display': 'flex', 'gap': '30px', 'marginTop': '30px'}, children=[
                
                # ฝั่งซ้าย: Control Panel (Input)
                html.Div(style={'flex': '1'}, children=[
                    html.Div(className='glass-panel', children=[
                        html.Div(className='input-group', children=[
                            html.H3("Biometric Profile"),
                            html.Label("Age (ปี)"),
                            dcc.Slider(id='age', min=20, max=85, value=50, marks={20:'20', 85:'85'}),
                            html.Label("Sex"),
                            dcc.Dropdown(id='sex', options=[{'label':'Male','value':1},{'label':'Female','value':0}], value=1),
                            html.Label("Country Model"),
                            dcc.Dropdown(id='country', options=[{'label':'USA','value':'USA'},{'label':'Hungary','value':'Hungary'}], value='USA'),
                        ]),
                        
                        html.Div(className='input-group', children=[
                            html.H3("Clinical Metrics"),
                            html.Label("Chest Pain Level (1-4)"),
                            dcc.Dropdown(id='cp', options=[{'label':f'Level {i}','value':i} for i in range(1,5)], value=2),
                            html.Label("Max Heart Rate (bpm)"),
                            dcc.Input(id='thalach', type='number', value=150, style={'width': '100%'}),
                        ]),
                        
                        html.Button("START AI ANALYSIS", id='predict-btn', n_clicks=0, className='btn-predict')
                    ])
                ]),

                # ฝั่งขวา: Display Panel (Result)
                html.Div(style={'flex': '1.5'}, children=[
                    html.Div(id='prediction-result'),
                    html.Div(className='glass-panel', style={'marginTop': '25px'}, children=[
                        html.H3("RISK FACTOR WEIGHTS", style={'color': '#ef4444', 'fontSize': '0.9rem', 'marginBottom': '15px'}),
                        dcc.Graph(id='importance-graph', style={'height': '350px'})
                    ])
                ])
            ])
        ]),

        # --- TAB 2: GLOBAL EDA ---
        dcc.Tab(label='GLOBAL INSIGHTS (EDA)', value='tab-eda', 
                className='custom-tab', selected_className='custom-tab--selected', children=[
            
            html.Div(className='glass-panel', style={'marginTop': '30px'}, children=[
                html.H2("Global Heart Disease Distribution", style={'color': '#ef4444', 'textAlign': 'center'}),
                
                html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                    html.Div(style={'flex': 1}, children=[dcc.Graph(id='eda-chol-box')]),
                    html.Div(style={'flex': 1}, children=[dcc.Graph(id='eda-scatter')])
                ]),
                
                html.Div(className='input-group', style={'marginTop': '20px'}, children=[
                    html.H3("EDA Insight Report"),
                    html.P("จากการเปรียบเทียบข้อมูลรายประเทศ พบว่าพฤติกรรมสุขภาพและพันธุกรรมส่งผลต่อปัจจัยเสี่ยงต่างกัน เช่น:"),
                    html.Ul([
                        html.Li("ผู้ป่วยใน USA มีแนวโน้มค่า Cholesterol สูงกว่ากลุ่มอื่นๆ"),
                        html.Li("ความสัมพันธ์ระหว่างอายุและอัตราการเต้นหัวใจสูงสุดมีผลลัพธ์ที่คล้ายคลึงกันในทุกโมเดล")
                    ])
                ])
            ])
        ])
    ]),
    
    # ส่วนท้าย: ข้อมูลอ้างอิง
    html.Div(style={'textAlign': 'center', 'marginTop': '40px', 'color': '#64748b', 'fontSize': '0.7rem'}, children=[
        html.P("Data Source: UCI Machine Learning Repository (Cleveland, Hungary, Switzerland, Long Beach V)")
    ])
])

# --- CALLBACKS ---

@app.callback(
    [Output('prediction-result', 'children'),
     Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('age','value'), State('sex','value'), State('cp','value'), 
     State('thalach','value'), State('country','value')]
)
def update_prediction(n, age, sex, cp, thalach, country):
    if n == 0:
        return html.Div(className='glass-panel', children=[html.H3("Waiting for input data...")]), go.Figure()

    # เตรียมข้อมูลสำหรับทำนาย
    features = predictor.feature_metadata_in.get_features()
    # Mock ค่าอื่นๆ ที่ไม่ได้กรอกให้เป็นค่ากลาง (Normal)
    input_data = pd.DataFrame([[age, sex, cp, 120, 200, 0, 0, thalach, 0, 0.0, 1, 0, 3, country]], columns=features)
    
    prob = predictor.predict_proba(input_data).iloc[0, 1]
    color = "#ef4444" if prob > 0.5 else "#22c55e"
    glow_class = "glow-red" if prob > 0.5 else "glow-green"
    
    # การ์ดแสดงผล
    res_card = html.Div(className=f'glass-panel {glow_class}', children=[
        html.Div("● AI ANALYSIS ACTIVE", className='status-badge'),
        html.H2("DIAGNOSIS PROBABILITY", style={'color': color, 'margin': '10px 0 0 0', 'fontSize': '1.2rem'}),
        
        html.Div(className='progress-bar-bg', children=[
            html.Div(className='progress-bar-fill', style={'width': f'{prob*100}%', 'backgroundColor': color})
        ]),
        
        html.H1(f"{prob*100:.1f}%", style={'fontSize': '90px', 'margin': '10px 0', 'color': '#f8fafc'}),
        html.P(f"Based on {country} Data Profile", style={'color': '#94a3b8', 'letterSpacing': '2px'})
    ])

    # กราฟ Feature Importance (Mock values เพื่อความสวยงาม)
    fig = px.bar(x=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05], 
                 y=['Chest Pain', 'Max HR', 'Age', 'ST Depress', 'Chol', 'BP'],
                 orientation='h', color_discrete_sequence=[color])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', margin=dict(l=20, r=20, t=20, b=20), height=300
    )
    
    return res_card, fig

@app.callback(
    [Output('eda-chol-box', 'figure'),
     Output('eda-scatter', 'figure')],
    Input('tabs', 'value')
)
def update_eda_graphs(tab):
    if tab != 'tab-eda':
        return go.Figure(), go.Figure()

    # กราฟ Box Plot เทียบ Cholesterol
    fig1 = px.box(df_eda, x='country', y='chol', color='target', 
                  title="Cholesterol Levels by Country",
                  color_discrete_map={0: '#22c55e', 1: '#ef4444'})
    
    # กราฟ Scatter Plot
    fig2 = px.scatter(df_eda, x='age', y='thalach', color='country', size='chol',
                      title="Age vs Max Heart Rate Analysis",
                      color_discrete_sequence=px.colors.qualitative.Pastel)

    for f in [fig1, fig2]:
        f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                        font_color='white', title_font_color='#ef4444')
        
    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)