import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# โหลดข้อมูล
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('processed.cleveland.data', names=columns, na_values='?')
df = df.fillna(df.median())

# สร้าง EDA: เปรียบเทียบความสัมพันธ์ของปัจจัยตามเพศ (แทนประเทศ)
plt.figure(figsize=(12, 6))
sns.boxplot(x='sex', y='thalach', hue='target', data=df, palette='Reds')
plt.title('Max Heart Rate vs Sex grouped by Heart Disease')
plt.savefig('assets/eda_plot.png') # บันทึกรูปไปโชว์ใน Dash
plt.show()