import pandas as pd

def clean_data(input_path):
    # อ่านข้อมูล (ระบุชื่อไฟล์ของคุณ)
    df = pd.read_csv(input_path)
    
    # 1. จัดการ Missing Values (2 คะแนน)
    # ตัวอย่าง: ถ้า Sleep Disorder ว่าง ให้เติมว่า 'None'
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    # 2. ทำ Feature Engineering (จัดการข้อมูล Blood Pressure)
    if 'Blood Pressure' in df.columns:
        df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df.drop('Blood Pressure', axis=1, inplace=True)
    
    # 3. ลบข้อมูลที่ไม่จำเป็น
    cols_to_drop = ['Person ID']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df.to_csv('cleaned_sleep_data.csv', index=False)
    print("✅ Data Cleaning Complete: Saved as cleaned_sleep_data.csv")

if __name__ == "__main__":
    clean_data('sleep_health_data.csv') # เปลี่ยนชื่อไฟล์ตามที่คุณมี