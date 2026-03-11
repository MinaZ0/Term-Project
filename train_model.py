import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. Load Data
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('processed.cleveland.data', names=columns, na_values='?')

# 2. Preprocessing
# จัดการค่าว่างด้วยค่ามัธยฐาน
df = df.fillna(df.median())
# ปรับ Target: 0 = ปกติ, 1-4 = มีความเสี่ยง (แปลงเป็น Binary Classification)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 3. Train Model ด้วย AutoGluon
print("Starting training with AutoGluon...")
predictor = TabularPredictor(label='target', path='ag_heart_model').fit(
    train_data=df, 
    presets='best_quality', # เน้นแม่นยำที่สุด
    time_limit=120          # รัน 2 นาที (ปรับเพิ่มได้ถ้ามีเวลา)
)

print("Training Complete! folder 'ag_heart_model' created.")