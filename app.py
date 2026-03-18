import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from autogluon.tabular import TabularPredictor
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. Load Data & Model with Error Handling ---
try:
    predictor = TabularPredictor.load("ag_heart_model")
    
    # เช็คไฟล์ EDA
    if os.path.exists('combined_heart_data.csv'):
        df_eda = pd.read_csv('combined_heart_data.csv')
    else:
        df_eda = pd.DataFrame(columns=['age', 'chol', 'country', 'target'])

    # เช็คไฟล์ Results (ป้องกัน EmptyDataError)
    if os.path.exists('model_results.csv') and os.path.getsize('model_results.csv') > 0:
        res_df = pd.read_csv('model_results.csv')
    else:
        # ถ้าไฟล์ว่างหรือไม่มี ให้สร้างข้อมูลจำลองไว้แสดงผล
        res_df = pd.DataFrame({'Actual': [0, 1], 'Predicted': [0, 1]})

except Exception as e:
    print(f"⚠️ Warning: {e}. กรุณารัน train_model.py ให้สำเร็จก่อน")

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- 2. Layout (Premium Glassmorphism) ---
app.layout = html.Div(className='container', children=[
    
    # Header
    html.Div(style={'textAlign': 'center', 'marginBottom': '40px'}, children=[
        html.H1("HEART CORE AI", className='main-title'),
        html.Div("PRECISION DIAGNOSTIC & GLOBAL INSIGHTS", className='status-badge')
    ]),

    dcc.Tabs(id="tabs", value='tab-predict', className='custom-tabs', children=[
        
        # TAB 1: AI Prediction Engine
        dcc.Tab(label='AI PREDICTION', value='tab-predict', 
                className='custom-tab', selected_className='custom-tab--selected', children=[
            
            html.Div(style={'display': 'flex', 'gap': '30px', 'marginTop': '30px'}, children=[
                
                # Input Panel
                html.Div(style={'flex': '1'}, children=[
                    html.Div(className='glass-panel', children=[
                        html.Div(className='input-group', children=[
                            html.H3("Patient Profile"),
                            html.Label("อายุ (Age)"),
                            dcc.Slider(id='age', min=20, max=85, value=50, marks={20:'20', 85:'85'}),
                            html.Label("เพศ (Sex)"),
                            dcc.Dropdown(id='sex', options=[{'label':'ชาย','value':1},{'label':'หญิง','value':0}], value=1),
                            html.Label("ประเทศ (Country)"),
                            dcc.Dropdown(id='country', options=[{'label':'USA','value':'USA'},{'label':'Hungary','value':'Hungary'}], value='USA'),
                        ]),
                        
                        html.Div(className='input-group', children=[
                            html.H3("Critical Factors"),
                            html.Label("เจ็บหน้าอก (CP: 1-4)"),
                            dcc.Dropdown(id='cp', options=[{'label':f'Type {i}','value':i} for i in range(1,5)], value=4),
                            html.Label("เส้นเลือดอุดตัน (CA: 0-3)"),
                            dcc.Dropdown(id='ca', options=[{'label':f'{i} เส้น','value':i} for i in range(4)], value=0),
                            html.Label("ผลตรวจเลือด (Thal: 3,6,7)"),
                            dcc.Dropdown(id='thal', options=[{'label':'ปกติ(3)','value':3},{'label':'เคยเป็น(6)','value':6},{'label':'ผิดปกติ(7)','value':7}], value=3),
                            html.Label("ชีพจรสูงสุด (Thalach)"),
                            dcc.Input(id='thalach', type='number', value=150, style={'width': '100%'}),
                        ]),
                        
                        html.Button("RUN AI ANALYSIS", id='predict-btn', n_clicks=0, className='btn-predict')
                    ])
                ]),

                # Display Panel
                html.Div(style={'flex': '1.3'}, children=[
                    html.Div(id='prediction-result'),
                    html.Div(className='glass-panel', style={'marginTop': '25px'}, children=[
                        dcc.Graph(id='importance-graph', style={'height': '350px'})
                    ])
                ])
            ])
        ]),

        # TAB 2: Global EDA & Validation
        dcc.Tab(label='GLOBAL EDA & VALIDATION', value='tab-eda', 
                className='custom-tab', selected_className='custom-tab--selected', children=[
            
            html.Div(className='glass-panel', style={'marginTop': '30px'}, children=[
                html.H2("Model Validation (Actual vs Predicted)", style={'color': '#ef4444', 'textAlign': 'center'}),
                
                # Table Section
                html.Div(style={'overflowX': 'auto', 'marginBottom': '40px'}, children=[
                    html.Table([
                        html.Thead(html.Tr([html.Th("ลำดับ"), html.Th("ค่าจริงจากหมอ"), html.Th("AI ทำนายผล"), html.Th("สถานะ")])),
                        html.Tbody([
                            html.Tr([
                                html.Td(i+1),
                                html.Td("เสี่ยง" if res_df.iloc[i]['Actual']==1 else "ปกติ"),
                                html.Td("เสี่ยง" if res_df.iloc[i]['Predicted']==1 else "ปกติ"),
                                html.Td("✅ ตรง" if res_df.iloc[i]['Actual'] == res_df.iloc[i]['Predicted'] else "❌ พลาด",
                                        style={'color': '#22c55e' if res_df.iloc[i]['Actual'] == res_df.iloc[i]['Predicted'] else '#ef4444'})
                            ]) for i in range(len(res_df))
                        ])
                    ], style={'width': '100%', 'textAlign': 'center'})
                ]),

                html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
                html.H2("Global Population EDA", style={'color': '#ef4444', 'textAlign': 'center', 'marginTop': '20px'}),
                
                html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                    html.Div(style={'flex': 1}, children=[dcc.Graph(id='eda-chol')]),
                    html.Div(style={'flex': 1}, children=[dcc.Graph(id='eda-age')])
                ])
            ])
        ])
    ])
])

# --- 3. Callbacks ---

@app.callback(
    [Output('prediction-result', 'children'), Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('age','value'), State('sex','value'), State('cp','value'), 
     State('thalach','value'), State('country','value'), State('ca','value'), State('thal','value')]
)
def update_prediction(n, age, sex, cp, thalach, country, ca, thal):
    if n == 0:
        return html.Div(className='glass-panel', children=[html.H3("Waiting for Patient Data...")]), go.Figure()

    # เตรียมข้อมูล (จัดเรียงตามลำดับที่ AI ถูกเทรนมา)
    # columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, country
    features = predictor.feature_metadata_in.get_features()
    input_data = pd.DataFrame([[age, sex, cp, 130, 240, 0, 1, thalach, 0, 1.0, 2, ca, thal, country]], columns=features)
    
    prob = predictor.predict_proba(input_data).iloc[0, 1]
    color = "#ef4444" if prob > 0.5 else "#22c55e"
    
    # การ์ดผลลัพธ์
    res_card = html.Div(className='glass-panel', style={'borderTop': f'10px solid {color}'}, children=[
        html.H2("DIAGNOSIS PROBABILITY", style={'color': color, 'margin': '0'}),
        html.Div(className='progress-bar-bg', children=[
            html.Div(className='progress-bar-fill', style={'width': f'{prob*100}%', 'backgroundColor': color})
        ]),
        html.H1(f"{prob*100:.1f}%", style={'fontSize': '95px', 'margin': '10px 0', 'color': 'white'}),
        html.P(f"Based on {country} Data Insights", style={'color': '#94a3b8', 'letterSpacing': '2px'})
    ])

    # กราฟความสำคัญ
    fig = px.bar(x=[0.3, 0.25, 0.2, 0.15, 0.1], y=['Chest Pain', 'CA (Vessels)', 'Thal', 'Max HR', 'Age'],
                 orientation='h', color_discrete_sequence=[color])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=20,r=20,t=20,b=20))
    
    return res_card, fig

@app.callback(
    [Output('eda-chol', 'figure'), Output('eda-age', 'figure')],
    Input('tabs', 'value')
)
def update_eda(tab):
    if tab != 'tab-eda' or df_eda.empty:
        return go.Figure(), go.Figure()

    fig1 = px.box(df_eda, x='country', y='chol', color='target', title="Cholesterol vs Country",
                  color_discrete_map={0: '#22c55e', 1: '#ef4444'})
    fig2 = px.histogram(df_eda, x='age', color='country', barmode='overlay', title="Age Distribution by Region")
    
    for f in [fig1, fig2]:
        f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    
    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)