from pycaret.classification import *
import pandas as pd

# โหลดข้อมูลที่คลีนแล้ว
df = pd.read_csv('cleaned_sleep_data.csv')

# 1. Setup PyCaret (3 คะแนน)
s = setup(data=df, target='Sleep Disorder', session_id=123, normalize=True)

# 2. Compare and Create Model
best_model = compare_models()

# 3. บันทึกโมเดลไว้ใช้ใน Dash
save_model(best_model, 'sleep_ai_model')

# 4. สร้างกราฟไว้โชว์ (5 คะแนน)
plot_model(best_model, plot='feature', save=True)
print("✅ Model Training Complete: Saved sleep_ai_model.pkl")