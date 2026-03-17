import pandas as pd
from autogluon.tabular import TabularPredictor

def load_and_prep(url, country_name):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    # โหลดข้อมูล (สมมติว่าคุณมีไฟล์ประเทศอื่นๆ ในเครื่อง หรือใช้ลิงก์ URL)
    df = pd.read_csv(url, names=columns, na_values='?')
    df = df.fillna(df.median())
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df['country'] = country_name # เพิ่มคอลัมน์ระบุประเทศ
    return df

# ในที่นี้จะจำลองการแบ่งข้อมูล Cleveland เป็น 2 ประเทศเพื่อให้เห็นภาพการเปรียบเทียบ
df_all = load_and_prep('processed.cleveland.data', 'USA')
# จำลองข้อมูลอีกชุดเป็น Hungary (ในงานจริงคุณสามารถหาไฟล์ hungarian.data มาเพิ่มได้)
df_hungary = df_all.sample(frac=0.5).copy()
df_hungary['country'] = 'Hungary'
df_hungary['chol'] = df_hungary['chol'] * 0.8 # สมมติตัวแปรต่างกัน

df_final = pd.concat([df_all, df_hungary])
df_final.to_csv('combined_heart_data.csv', index=False)

# เทรนโมเดล
predictor = TabularPredictor(label='target', path='ag_heart_model').fit(df_final, presets='best_quality')
print("✅ Training Complete with Multi-Country Data!")