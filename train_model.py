import pandas as pd
from autogluon.tabular import TabularPredictor
import os

def prepare_and_train():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # 1. โหลดข้อมูล
    if not os.path.exists('processed.cleveland.data'):
        print("❌ ไม่พบไฟล์ processed.cleveland.data")
        return

    df = pd.read_csv('processed.cleveland.data', names=columns, na_values='?')
    df = df.fillna(df.median())
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df['country'] = 'USA'

    # 2. จำลองข้อมูลเปรียบเทียบ
    df_compare = df.sample(frac=0.8, random_state=42).copy()
    df_compare['country'] = 'Hungary'
    
    df_combined = pd.concat([df, df_compare], ignore_index=True)
    df_combined.to_csv('combined_heart_data.csv', index=False)

    # 3. เทรนโมเดล
    predictor = TabularPredictor(label='target', path='ag_heart_model').fit(
        df_combined, presets='best_quality'
    )
    
    # 4. สร้างข้อมูลผลลัพธ์ (แก้ปัญหาไฟล์ว่าง)
    test_sample = df_combined.sample(n=10, random_state=7)
    preds = predictor.predict(test_sample.drop(columns=['target']))
    
    results = pd.DataFrame({
        'Actual': test_sample['target'].values, 
        'Predicted': preds.values
    })
    
    # บันทึกไฟล์และเช็คความเรียบร้อย
    results.to_csv('model_results.csv', index=False)
    print(f"✅ สร้างไฟล์ model_results.csv สำเร็จ (มี {len(results)} แถว)")
    print("🚀 ทุกอย่างพร้อมแล้ว! รัน python app.py ได้เลย")

if __name__ == '__main__':
    prepare_and_train()