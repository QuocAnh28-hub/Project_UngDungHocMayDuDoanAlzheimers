import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Chẩn đoán Alzheimer - Machine Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and scaler
@st.cache_resource
def load_resources():
    rf = joblib.load('models/best_rf.joblib')
    knn = joblib.load('models/best_knn.joblib')
    svc = joblib.load('models/best_svc.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/features.json', 'r') as f:
        features = json.load(f)
    with open('models/final_metrics.json', 'r') as f:
        metrics = json.load(f)
    return rf, knn, svc, scaler, features, metrics

rf_model, knn_model, svc_model, scaler, feature_names, metrics_data = load_resources()

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 20px;
        border-radius: 10px;
        color: #991b1b;
    }
    .risk-low {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 20px;
        border-radius: 10px;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🧠 HỆ THỐNG DỰ ĐOÁN NGUY CƠ BỆNH ALZHEIMER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ứng dụng Học máy (Random Forest, KNN, SVM) hỗ trợ sàng lọc và phát hiện sớm bệnh Alzheimer</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=70)
st.sidebar.title("⚙️ Cấu hình hệ thống")

model_choice = st.sidebar.selectbox(
    "Lựa chọn mô hình dự đoán:",
    ["RandomForestClassifier (Khuyên dùng - 94.2%)", "Support Vector Machine (85.4%)", "KNeighborsClassifier (69.5%)"]
)

if "RandomForest" in model_choice:
    current_model = rf_model
    model_name_tag = "RandomForestClassifier"
elif "Support Vector" in model_choice:
    current_model = svc_model
    model_name_tag = "Support Vector Classification"
else:
    current_model = knn_model
    model_name_tag = "KNeighborsClassifier"

# Sample Profiles in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Mẫu bệnh nhân thử nghiệm")
sample_choice = st.sidebar.radio(
    "Nạp hồ sơ mẫu nhanh:",
    ["Tự nhập thủ công", "Mẫu 1: Nguy cơ cao (Suy giảm nhận thức nặng)", "Mẫu 2: Nguy cơ thấp (Người cao tuổi khỏe mạnh)"]
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Dự đoán nguy cơ cá nhân", "📊 Khám phá dữ liệu (EDA)", "📈 Hiệu năng & So sánh mô hình"])

with tab1:
    st.subheader("📝 Nhập thông tin y tế của bệnh nhân")
    st.info("💡 Vui lòng nhập đầy đủ các chỉ số nhân khẩu học, lối sống, tiền sử bệnh và các bài kiểm tra nhận thức dưới đây để nhận đánh giá chẩn đoán tức thì.")

    # Preset values
    if sample_choice == "Mẫu 1: Nguy cơ cao (Suy giảm nhận thức nặng)":
        def_age, def_gender, def_eth, def_edu = 82, 1, 0, 1
        def_bmi, def_smoke, def_alc, def_act, def_diet, def_sleep = 27.5, 1, 8.0, 1.5, 4.0, 5.0
        def_fam, def_cardio, def_diab, def_dep, def_head, def_hyp = 1, 1, 1, 1, 0, 1
        def_sys, def_dia, def_chol, def_ldl, def_hdl, def_trig = 155, 95, 260.0, 160.0, 35.0, 280.0
        def_mmse, def_func, def_adl, def_mem, def_beh = 12.5, 3.2, 2.8, 1, 1
        def_conf, def_diso, def_pers, def_diff, def_forg = 1, 1, 1, 1, 1
    elif sample_choice == "Mẫu 2: Nguy cơ thấp (Người cao tuổi khỏe mạnh)":
        def_age, def_gender, def_eth, def_edu = 68, 0, 0, 3
        def_bmi, def_smoke, def_alc, def_act, def_diet, def_sleep = 22.8, 0, 1.0, 6.5, 8.5, 8.0
        def_fam, def_cardio, def_diab, def_dep, def_head, def_hyp = 0, 0, 0, 0, 0, 0
        def_sys, def_dia, def_chol, def_ldl, def_hdl, def_trig = 118, 76, 175.0, 95.0, 60.0, 110.0
        def_mmse, def_func, def_adl, def_mem, def_beh = 28.5, 8.8, 9.2, 0, 0
        def_conf, def_diso, def_pers, def_diff, def_forg = 0, 0, 0, 0, 0
    else:
        def_age, def_gender, def_eth, def_edu = 72, 0, 0, 2
        def_bmi, def_smoke, def_alc, def_act, def_diet, def_sleep = 24.5, 0, 2.0, 4.0, 6.0, 7.0
        def_fam, def_cardio, def_diab, def_dep, def_head, def_hyp = 0, 0, 0, 0, 0, 0
        def_sys, def_dia, def_chol, def_ldl, def_hdl, def_trig = 130, 80, 200.0, 110.0, 50.0, 150.0
        def_mmse, def_func, def_adl, def_mem, def_beh = 21.0, 6.0, 6.5, 0, 0
        def_conf, def_diso, def_pers, def_diff, def_forg = 0, 0, 0, 0, 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 1. Nhân khẩu học & Lối sống")
        age = st.slider("Tuổi (Age):", 60, 90, def_age)
        gender = st.selectbox("Giới tính (Gender):", [0, 1], index=def_gender, format_func=lambda x: "Nam" if x==0 else "Nữ")
        ethnicity = st.selectbox("Chủng tộc (Ethnicity):", [0, 1, 2, 3], index=def_eth,
                                 format_func=lambda x: {0:"Caucasian", 1:"African American", 2:"Asian", 3:"Other"}[x])
        education = st.selectbox("Trình độ học vấn (Education Level):", [0, 1, 2, 3], index=def_edu,
                                 format_func=lambda x: {0:"Không có bằng cấp", 1:"Trung học", 2:"Đại học / Cử nhân", 3:"Sau đại học"}[x])
        bmi = st.number_input("Chỉ số khối cơ thể (BMI):", 15.0, 40.0, float(def_bmi), step=0.1)
        smoking = st.selectbox("Tình trạng hút thuốc (Smoking):", [0, 1], index=def_smoke, format_func=lambda x: "Không hút" if x==0 else "Có hút thuốc")
        alcohol = st.slider("Mức tiêu thụ rượu bia hàng tuần (đơn vị):", 0.0, 20.0, float(def_alc), step=0.5)
        physical = st.slider("Hoạt động thể chất (giờ/tuần):", 0.0, 10.0, float(def_act), step=0.5)
        diet = st.slider("Chất lượng chế độ ăn uống (Diet Quality 0-10):", 0.0, 10.0, float(def_diet), step=0.5)
        sleep = st.slider("Chất lượng giấc ngủ (Sleep Quality 4-10):", 4.0, 10.0, float(def_sleep), step=0.5)

        st.markdown("##### 2. Tiền sử bệnh lý")
        c1, c2 = st.columns(2)
        with c1:
            fam_hist = st.checkbox("Tiền sử gia đình có người mắc Alzheimer", value=bool(def_fam))
            cardio = st.checkbox("Bệnh tim mạch", value=bool(def_cardio))
            diabetes = st.checkbox("Bệnh tiểu đường", value=bool(def_diab))
        with c2:
            depression = st.checkbox("Trầm cảm", value=bool(def_dep))
            head_injury = st.checkbox("Tiền sử chấn thương sọ não", value=bool(def_head))
            hypertension = st.checkbox("Tăng huyết áp", value=bool(def_hyp))

    with col2:
        st.markdown("##### 3. Chỉ số đo lường lâm sàng")
        b1, b2 = st.columns(2)
        with b1:
            systolic = st.number_input("Huyết áp tâm thu (Systolic BP, mmHg):", 90, 180, def_sys)
            chol_total = st.number_input("Cholesterol toàn phần (mg/dL):", 150.0, 300.0, float(def_chol), step=1.0)
            chol_hdl = st.number_input("Cholesterol HDL (mg/dL):", 20.0, 100.0, float(def_hdl), step=1.0)
        with b2:
            diastolic = st.number_input("Huyết áp tâm trương (Diastolic BP, mmHg):", 60, 120, def_dia)
            chol_ldl = st.number_input("Cholesterol LDL (mg/dL):", 50.0, 200.0, float(def_ldl), step=1.0)
            triglycerides = st.number_input("Triglycerides (mg/dL):", 50.0, 400.0, float(def_trig), step=1.0)

        st.markdown("##### 4. Đánh giá nhận thức & Hoạt động chức năng (Rất quan trọng)")
        mmse = st.slider("Điểm MMSE (Mini-Mental State Examination, 0 - 30):", 0.0, 30.0, float(def_mmse), step=0.5,
                         help="Điểm càng thấp thể hiện suy giảm trí tuệ và nhận thức càng nghiêm trọng.")
        func_assess = st.slider("Đánh giá chức năng (Functional Assessment, 0 - 10):", 0.0, 10.0, float(def_func), step=0.1)
        adl = st.slider("Điểm sinh hoạt thường ngày (ADL - Activities of Daily Living, 0 - 10):", 0.0, 10.0, float(def_adl), step=0.1)

        st.markdown("##### 5. Các biểu hiện và triệu chứng thần kinh")
        s1, s2 = st.columns(2)
        with s1:
            memory_comp = st.checkbox("Phàn nàn về suy giảm trí nhớ", value=bool(def_mem))
            behav_prob = st.checkbox("Vấn đề bất thường về hành vi", value=bool(def_beh))
            confusion = st.checkbox("Tình trạng lú lẫn", value=bool(def_conf))
        with s2:
            disorient = st.checkbox("Mất định hướng không gian/thời gian", value=bool(def_diso))
            personality = st.checkbox("Thay đổi tính cách rõ rệt", value=bool(def_pers))
            diff_tasks = st.checkbox("Khó khăn hoàn thành việc quen thuộc", value=bool(def_diff))
            forget = st.checkbox("Hay quên nghiêm trọng", value=bool(def_forg))

    st.markdown("---")
    predict_btn = st.button("🚀 TIẾN HÀNH DỰ ĐOÁN NGUY CƠ ALZHEIMER", use_container_width=True, type="primary")

    if predict_btn:
        # Prepare input dict
        input_data = {
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'EducationLevel': education,
            'BMI': bmi,
            'Smoking': smoking,
            'AlcoholConsumption': alcohol,
            'PhysicalActivity': physical,
            'DietQuality': diet,
            'SleepQuality': sleep,
            'FamilyHistoryAlzheimers': int(fam_hist),
            'CardiovascularDisease': int(cardio),
            'Diabetes': int(diabetes),
            'Depression': int(depression),
            'HeadInjury': int(head_injury),
            'Hypertension': int(hypertension),
            'SystolicBP': systolic,
            'DiastolicBP': diastolic,
            'CholesterolTotal': chol_total,
            'CholesterolLDL': chol_ldl,
            'CholesterolHDL': chol_hdl,
            'CholesterolTriglycerides': triglycerides,
            'MMSE': mmse,
            'FunctionalAssessment': func_assess,
            'MemoryComplaints': int(memory_comp),
            'BehavioralProblems': int(behav_prob),
            'ADL': adl,
            'Confusion': int(confusion),
            'Disorientation': int(disorient),
            'PersonalityChanges': int(personality),
            'DifficultyCompletingTasks': int(diff_tasks),
            'Forgetfulness': int(forget)
        }

        # Convert to DataFrame with correct columns order
        input_df = pd.DataFrame([input_data])[feature_names]
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = current_model.predict(input_scaled)[0]
        probability = current_model.predict_proba(input_scaled)[0][1]

        st.markdown("### 📊 KẾT QUẢ PHÂN TÍCH CHẨN ĐOÁN LÂM SÀNG")
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ NGUY CƠ CAO MẮC BỆNH ALZHEIMER</h3>
                    <p style="font-size: 1.15rem;">Hệ thống dự đoán bệnh nhân có các dấu hiệu điển hình của bệnh Alzheimer với xác suất <b>{probability*100:.2f}%</b> (Sử dụng mô hình {model_name_tag}).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>🛡️ NGUY CƠ THẤP / CHƯA PHÁT HIỆN DẤU HIỆU BỆNH ALZHEIMER</h3>
                    <p style="font-size: 1.15rem;">Hệ thống dự đoán khả năng mắc bệnh Alzheimer ở mức an toàn với xác suất bệnh <b>{probability*100:.2f}%</b> (Sử dụng mô hình {model_name_tag}).</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Thước đo xác suất mắc bệnh:**")
            st.progress(float(probability))

            st.markdown("#### 💡 Khuyến nghị y khoa:")
            if prediction == 1:
                st.error("""
                - **Khám chuyên sâu**: Cần đưa bệnh nhân đến chuyên khoa Thần kinh tại các bệnh viện tuyến đầu để chụp MRI/PET sọ não và xét nghiệm sinh hóa dịch não tủy.
                - **Hỗ trợ sinh hoạt**: Điểm ADL và Functional Assessment thấp cho thấy bệnh nhân cần người thân hỗ trợ trong các sinh hoạt cá nhân thường nhật.
                - **Theo dõi triệu chứng**: Ghi chép nhật ký hành vi, kiểm soát nguy cơ té ngã và mất định hướng khi đi lạc.
                """)
            else:
                st.success("""
                - **Duy trì lối sống lành mạnh**: Tiếp tục duy trì chế độ ăn giàu rau xanh, cá béo (Mediterranean diet) và hoạt động thể chất đều đặn.
                - **Rèn luyện nhận thức**: Đọc sách, chơi cờ, học ngoại ngữ hoặc các hoạt động kích thích trí não để tăng cường dự trữ nhận thức.
                - **Kiểm tra định kỳ**: Khám sức khỏe tổng quát và làm bài test MMSE hàng năm cho người trên 65 tuổi.
                """)

        with res_col2:
            st.markdown("#### 🔍 So sánh chỉ số nhận thức then chốt")
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
            metrics_compare = {
                'MMSE (max 30)': (mmse / 30.0) * 100,
                'ADL (max 10)': (adl / 10.0) * 100,
                'Func Assess (max 10)': (func_assess / 10.0) * 100,
                'Chất lượng ngủ (max 10)': (sleep / 10.0) * 100,
                'Chế độ ăn (max 10)': (diet / 10.0) * 100
            }
            colors = ['#dc2626' if v < 50 else '#16a34a' for v in metrics_compare.values()]
            ax.barh(list(metrics_compare.keys()), list(metrics_compare.values()), color=colors, height=0.55)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Tỷ lệ phần trăm so với thang chuẩn (%)')
            ax.axvline(50, color='gray', linestyle='--', alpha=0.7, label='Ngưỡng cảnh báo 50%')
            ax.legend(loc='lower right')
            for i, v in enumerate(metrics_compare.values()):
                ax.text(v + 2, i, f"{v:.1f}%", va='center', fontsize=9, weight='bold')
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader("📊 Khám phá và Trực quan hóa Dữ liệu (EDA)")
    eda_df = pd.read_csv('alzheimers_disease_data.csv')
    st.write(f"Tổng số bản ghi: **{len(eda_df):,}** bệnh nhân | Số đặc trưng: **{eda_df.shape[1]}** thuộc tính")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Phân phối tỷ lệ mắc bệnh Alzheimer (Diagnosis)")
        st.image('images/fig3_19_result.png', use_container_width=True)
    with c2:
        st.markdown("##### Ma trận tương quan giữa các đặc trưng lâm sàng")
        st.image('images/fig3_14_result.png', use_container_width=True)

    st.markdown("---")
    st.markdown("##### Tương quan giữa Điểm MMSE, Hoạt động sinh hoạt ADL và Tuổi theo Chẩn đoán")
    st.image('images/fig3_25_result.png', use_container_width=True)

    st.markdown("---")
    st.markdown("##### Xếp hạng mức độ tương quan của từng đặc trưng với biến mục tiêu")
    st.image('images/fig3_23_result.png', use_container_width=True)

with tab3:
    st.subheader("📈 Đánh giá và So sánh Hiệu năng Mô hình")
    baseline_df = pd.read_csv('baseline_comparison.csv')
    st.markdown("##### 1. Bảng so sánh các thuật toán phân loại học máy")
    st.dataframe(
        baseline_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}'
        }).background_gradient(subset=['F1-Score', 'Accuracy'], cmap='Blues'),
        use_container_width=True
    )

    st.markdown("##### 2. Biểu đồ so sánh trực quan hiệu năng")
    st.image('images/fig3_39_result.png', use_container_width=True)

    st.markdown("---")
    st.markdown("##### 3. Chi tiết kết quả 3 mô hình chính sau khi Tối ưu siêu tham số (GridSearchCV)")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("**1. RandomForestClassifier (Tốt nhất)**")
        st.image('images/fig3_44_result.png', use_container_width=True)
        st.caption("Accuracy: 94.19% | F1-Score: 91.53% | AUC: 0.9380")
    with m_col2:
        st.markdown("**2. Support Vector Machine (SVC)**")
        st.image('images/fig3_48_result.png', use_container_width=True)
        st.caption("Accuracy: 85.35% | F1-Score: 79.61% | AUC: 0.8972")
    with m_col3:
        st.markdown("**3. KNeighborsClassifier (KNN)**")
        st.image('images/fig3_46_result.png', use_container_width=True)
        st.caption("Accuracy: 69.53% | F1-Score: 61.81% | AUC: 0.7837")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Trường Đại học Công nghệ Kỹ thuật Hưng Yên - 2026</p>", unsafe_allow_html=True)
