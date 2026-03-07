import pandas as pd
import numpy as np
from pycaret.classification import *

# 1. จัดการข้อมูล (Data Cleaning)
# ระบุชื่อคอลัมน์ตาม UCI Heart Disease Dataset
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# โหลดข้อมูล (สมมติว่าไฟล์อยู่ในโฟลเดอร์ data)
df = pd.read_csv('data/processed.cleveland.data', names=cols, na_values='?')

# Clean: ลบแถวที่มีค่าว่าง (2 คะแนน)
df = df.dropna()

# ปรับ Target: ให้ 0=ปกติ, 1=เสี่ยงโรค (จากเดิม 1-4 ให้เป็น 1)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
df.to_csv('data/heart_cleaned.csv', index=False)

# 2. เทรนโมเดลด้วย PyCaret (3 คะแนน)
s = setup(data=df, target='target', session_id=123, verbose=False)
best_model = compare_models() # ค้นหาโมเดลที่ดีที่สุด
final_model = finalize_model(best_model)

# บันทึกโมเดลไว้ในโฟลเดอร์ models
save_model(final_model, 'models/heart_model')
print("--- Training Complete: Model saved in models/heart_model.pkl ---")