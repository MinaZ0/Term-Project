import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
from pycaret.classification import load_model, predict_model
import plotly.graph_objects as go
import plotly.express as px

# โหลดโมเดลและข้อมูลที่คลีนแล้ว
model = load_model('models/heart_model')
df = pd.read_csv('data/heart_cleaned.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    html.H1("Heart Disease Prediction Dashboard", className="text-center my-4"),
    
    dbc.Row([
        # ส่วนที่ 1: Input Parameters (3 คะแนน)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Patient Data Input"),
                dbc.CardBody([
                    html.Label("Age"),
                    dcc.Slider(id='age-slider', min=20, max=80, value=50, marks={i: str(i) for i in range(20, 81, 10)}),
                    html.Label("Sex (1=Male, 0=Female)", className="mt-2"),
                    dcc.Dropdown(id='in-sex', options=[{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}], value=1),
                    html.Label("Cholesterol", className="mt-2"),
                    dbc.Input(id='in-chol', type='number', value=240),
                    dbc.Button("Predict Now", id='btn-predict', color="danger", className="mt-4 w-100")
                ])
            ])
        ], width=4),

        # ส่วนที่ 2: Prediction Graph (5 คะแนน)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Analysis Result"),
                dbc.CardBody([
                    dcc.Graph(id='gauge-plot'),
                    html.H2(id='output-text', className="text-center")
                ])
            ])
        ], width=8)
    ]),

    # ส่วนที่ 3: Custom Module - Data Insights (5 คะแนน)
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.H3("Custom Module: Dataset Trends"),
            html.P("Compare patient's cholesterol with overall dataset distribution:"),
            dcc.Graph(id='trend-plot')
        ])
    ], className="mt-4 mb-5")
], fluid=True)

# Callback สำหรับประมวลผลทั้งหมด
# ตรงนี้ต้องใช้ชื่อ id ให้ตรงกับที่แก้ข้างบน
@app.callback(
    [Output('gauge-plot', 'figure'), Output('output-text', 'children'), Output('trend-plot', 'figure')],
    Input('btn-predict', 'n_clicks'),
    [State('age-slider', 'value'), 
     State('in-sex', 'value'), 
     State('in-chol', 'value')]
)
def update_dashboard(n, age, sex, chol):
    # สร้างข้อมูลจำลองให้ครบ 13 คอลัมน์ตามที่โมเดลต้องการ
    data_dict = {'age':[age], 'sex':[sex], 'cp':[3], 'trestbps':[130], 'chol':[chol], 
                 'fbs':[0], 'restecg':[1], 'thalach':[150], 'exang':[0], 
                 'oldpeak':[1.0], 'slope':[1], 'ca':[0], 'thal':[2]}
    input_df = pd.DataFrame(data_dict)
    
    # พยากรณ์
    pred = predict_model(model, data=input_df)
    score = pred['prediction_score'][0] * 100
    label = "High Risk" if pred['prediction_label'][0] == 1 else "Low Risk"

    # กราฟที่ 1: Gauge (Prediction Graph)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={'text': "Risk Probability %"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if label=="High Risk" else "green"}}
    ))

    # กราฟที่ 2: Histogram (Custom Module)
    fig_trend = px.histogram(df, x="chol", color="target", marginal="box", 
                             title="Cholesterol Distribution in Dataset")
    fig_trend.add_vline(x=chol, line_dash="dash", line_color="black", annotation_text="Patient's Level")

    return fig_gauge, f"Conclusion: {label}", fig_trend

if __name__ == '__main__':
    app.run_server(debug=True)