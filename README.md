❤️ Heart Disease AI Diagnostic Tool

ระบบวิเคราะห์ความเสี่ยงโรคหัวใจด้วย Machine Learning และ Dashboard แบบ Interactive
พัฒนาโดยใช้ AutoML เพื่อเลือกโมเดลที่ดีที่สุด และแสดงผลผ่าน Web Dashboard

🔍 Overview

โปรเจกต์นี้มีวัตถุประสงค์เพื่อพยากรณ์ความเสี่ยงของโรคหัวใจจากข้อมูลสุขภาพของผู้ป่วย
โดยใช้เทคนิค Machine Learning + AutoML (AutoGluon)
และแสดงผลผ่าน Dashboard ที่ใช้งานง่าย

🚀 Features
	•	🔹 วิเคราะห์ความเสี่ยงโรคหัวใจแบบ Real-time
	•	🔹 ใช้ AutoML เพื่อเลือกโมเดลอัตโนมัติ
	•	🔹 ใช้ Ensemble เพิ่มความแม่นยำ
	•	🔹 รองรับการเลือก Country เพื่อปรับผลลัพธ์
	•	🔹 แสดงผลเป็น % ความเสี่ยง
	•	🔹 มีกราฟ Feature Importance และ EDA

🧠 Machine Learning

โปรเจกต์นี้ใช้ AutoML (AutoGluon) เพื่อทดลองหลายโมเดล เช่น:
	•	Logistic Regression
	•	Decision Tree
	•	Random Forest
	•	Neural Network

และใช้เทคนิค Ensemble เพื่อรวมผลลัพธ์จากหลายโมเดล
ทำให้ได้ผลลัพธ์ที่แม่นยำและเสถียรมากขึ้น

📊 Important Features

โมเดลให้ความสำคัญกับตัวแปรหลัก ได้แก่:
	•	CP (Chest Pain) – ประเภทอาการเจ็บหน้าอก
	•	Thalach (Max Heart Rate) – อัตราการเต้นหัวใจสูงสุด
	•	Oldpeak (ST Depression) – ความผิดปกติของคลื่นหัวใจ
  
🗂 Dataset
	•	ใช้ชุดข้อมูล Heart Disease Dataset
	•	จำนวนข้อมูล: 303 แถว
	•	จำนวนตัวแปร: 14 คอลัมน์
	•	ข้อมูลประกอบด้วย:
	•	Age, Sex
	•	Blood Pressure
	•	Cholesterol
	•	Heart Rate
	•	Target (โรคหัวใจ)

🖥️ Dashboard

พัฒนาโดยใช้:
	•	Python
	•	Dash
	•	Plotly

ความสามารถ:
	•	กรอกข้อมูลสุขภาพ
	•	วิเคราะห์ผลทันที
•	แสดงผลเป็นเปอร์เซ็นต์
	•	แสดงกราฟช่วยวิเคราะห์

🛠️ Technologies Used
	•	Python
	•	Pandas
	•	Plotly
	•	Dash
	•	AutoGluon (AutoML)

⚙️ Installation
  git clone https://github.com/MinaZ0/Term-Project.git
  cd Term-Project
  pip install -r requirements.txt
  python app.py

▶️ Usage
1. Train Model
   python train_model.py
2. Run Dashboard
   python app.py

📌 Project Structure
Term-Project/
│
├── app.py                     # ไฟล์หลักสำหรับรัน Dashboard
├── train_model.py             # ใช้สำหรับเทรนโมเดล Machine Learning
├── eda_report.py              # วิเคราะห์ข้อมูล (EDA)
│
├── combined_heart_data.csv    # Dataset หลัก
├── model_results.csv          # ผลลัพธ์ของโมเดล
│
├── processed.cleveland.data   # ข้อมูลจาก Cleveland
├── processed.hungarian.data   # ข้อมูลจาก Hungary
├── processed.switzerland.data # ข้อมูลจาก Switzerland
│
├── ag_heart_model/            # โฟลเดอร์เก็บโมเดลที่เทรนแล้ว
├── assets/                    # CSS / รูปภาพ สำหรับ Dashboard
│
├── requirements.txt           # รายการไลบรารีที่ใช้
├── .gitignore                 # ไฟล์ที่ไม่ต้องการอัปโหลดขึ้น Git
│
└── README.md                

โครงสร้างโปรเจกต์ถูกออกแบบให้แยกส่วนของ
- การวิเคราะห์ข้อมูล (EDA)
- การพัฒนาโมเดล (Training)
- การแสดงผล (Dashboard)
เพื่อให้สามารถพัฒนาและปรับปรุงแต่ละส่วนได้อย่างอิสระ

📈 Future Improvements
	•	เพิ่ม Dataset ให้หลากหลายมากขึ้น
	•	เพิ่มความแม่นยำของโมเดล
	•	รองรับข้อมูลแบบ Real-world
	•	เพิ่มระบบแนะนำสุขภาพ

👨‍💻 Authors
	•	แพรวา แก้วเจริญ     รหัสนักศึกษา 6810110242 รับผิดชอบในส่วน data and cleansing data
  • เอกนฤน ฟองสุวรรณ   รหัสนักศึกษา 6810110433 รับผิดชอบในส่วนของ model dashboard
  • อนันตพงศ์ แสนเดช   รหัสนักศึกษา 6810110746 รับผิกชอบในส่วนของ รายงาน,present

