# Ứng Dụng Học Máy Dự Đoán Nguy Cơ Mắc Bệnh Alzheimer
**MACHINE LEARNING APPLICATIONS FOR PREDICTING ALZHEIMER'S DISEASE**

- **Môn học**: Học máy cơ bản
- **Sinh viên thực hiện**: 
  1. Đỗ Hữu Quốc Ánh
  2. Đỗ Tiến Đạt
- **Lớp**: 12523W.4
- **Giáo viên hướng dẫn**: TS. Hoàng Quốc Việt
- **Trường**: Đại học Công nghệ Kỹ thuật Hưng Yên - 2026

---

## 📌 Cấu trúc dự án

```text
Project_2/
│
├── BTL.docx                        # Báo cáo Bài tập lớn hoàn chỉnh
├── Alzheimer_Disease_Prediction.ipynb  # File Notebook xử lý dữ liệu và huấn luyện mô hình hoàn chỉnh
├── alzheimers_disease_data.csv     # Bộ dữ liệu bệnh Alzheimer (2,149 mẫu, 35 thuộc tính)
├── app.py                          # Ứng dụng Web chẩn đoán lâm sàng tương tác bằng Streamlit
├── baseline_comparison.csv         # Bảng tổng hợp số liệu so sánh các mô hình học máy
├── README.md                       # Tài liệu hướng dẫn sử dụng và thông tin dự án
│
├── models/                         # Trọng số mô hình đã lưu
│   ├── best_rf.joblib              # Mô hình Random Forest tối ưu (Accuracy 94.19%, F1 91.53%)
│   ├── best_svc.joblib             # Mô hình Support Vector Machine tối ưu (Accuracy 85.35%)
│   ├── best_knn.joblib             # Mô hình K-Nearest Neighbors tối ưu (Accuracy 69.53%)
│   ├── scaler.joblib               # Chuẩn hóa Z-Score StandardScaler
│   ├── features.json               # Danh sách 32 đặc trưng đầu vào
│   └── final_metrics.json          # Báo cáo chi tiết ma trận nhầm lẫn và độ đo đánh giá
│
├── images/                         # Toàn bộ hình ảnh minh họa cho Báo cáo BTL.docx
├── Tài liệu code mẫu/              # Các bài thực hành tham khảo
└── Tài liệu lý thuyết/             # Slide bài giảng Học máy cơ bản (TS. Hoàng Quốc Việt)
```

---

## 🚀 Hướng dẫn chạy ứng dụng Web chẩn đoán (Streamlit)

Chạy lệnh sau trong PowerShell hoặc Terminal tại thư mục dự án:

```powershell
python -m streamlit run app.py
```

Sau khi chạy, mở trình duyệt tại địa chỉ: **`http://localhost:8501`**

### Các tính năng chính của ứng dụng:
1. **Dự đoán nguy cơ cá nhân**:
   - Nhập 32 chỉ số thuộc 6 nhóm: Nhân khẩu học, Lối sống, Tiền sử bệnh, Chỉ số lâm sàng, Đánh giá nhận thức MMSE/ADL và Triệu chứng lâm sàng.
   - Hỗ trợ nạp nhanh mẫu bệnh nhân nguy cơ cao hoặc người khỏe mạnh để thử nghiệm tức thì.
   - Hiển thị chẩn đoán, thanh đo xác suất nguy cơ, đồ thị so sánh chỉ số với ngưỡng an toàn, và các khuyến nghị can thiệp y khoa.
2. **Khám phá dữ liệu (EDA)**: Trực quan hóa ma trận tương quan, phân phối tỷ lệ mắc bệnh, biểu đồ đa biến theo độ tuổi và điểm nhận thức.
3. **Hiệu năng & So sánh mô hình**: Bảng đối sánh và biểu đồ ma trận nhầm lẫn, đường cong ROC giữa RandomForest, SVC và KNN.

---

## 📊 Kết quả thực nghiệm chính

| Mô hình thuật toán | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier** | **94.19%** | **94.41%** | **88.82%** | **91.53%** | **0.9380** |
| **Support Vector Machine (SVC)** | 85.35% | 78.34% | 80.92% | 79.61% | 0.8972 |
| **KNeighborsClassifier (KNN)** | 69.53% | 55.50% | 69.74% | 61.81% | 0.7837 |

- **Top các yếu tố nguy cơ hàng đầu**:
  1. `FunctionalAssessment` (Đánh giá chức năng)
  2. `ADL` (Hoạt động sinh hoạt thường ngày)
  3. `MMSE` (Điểm kiểm tra trạng thái tâm thần)
  4. `MemoryComplaints` (Phàn nàn về suy giảm trí nhớ)
  5. `BehavioralProblems` (Rối loạn hành vi)
