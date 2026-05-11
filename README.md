# Financial Fraud Detection using Graph Mining on PaySim Dataset

## 1. Giới thiệu

Dự án tập trung nghiên cứu và xây dựng hệ thống phát hiện gian lận tài chính (Fraud Detection) dựa trên phương pháp khai phá đồ thị (Graph Mining) sử dụng bộ dữ liệu PaySim.

Thay vì chỉ xem từng giao dịch độc lập như các mô hình Machine Learning truyền thống, dự án biểu diễn hệ thống giao dịch dưới dạng đồ thị:
- Node: tài khoản
- Edge: giao dịch tài chính

Qua đó mô hình có thể khai thác:
- quan hệ giữa các tài khoản
- hành vi giao dịch bất thường
- fraud rings
- money laundering paths
- mule accounts

---

# 2. Dataset

Dataset sử dụng:
- PaySim Mobile Money Transactions Dataset

Nguồn:
https://www.kaggle.com/datasets/ealaxi/paysim1

Dataset mô phỏng hệ thống giao dịch mobile money thực tế với:
- hơn 6 triệu giao dịch
- nhiều loại transaction
- nhãn fraud

---

# 3. Mục tiêu dự án

Mục tiêu của dự án:
- Phân tích dữ liệu giao dịch tài chính
- Thực hiện EDA chuyên sâu
- Xây dựng đồ thị giao dịch
- Trích xuất graph features
- Phát hiện tài khoản gian lận
- Nghiên cứu Graph-based Fraud Detection
- So sánh Machine Learning và Graph Mining

---

# 4. Công nghệ sử dụng

## Ngôn ngữ
- Python

## Thư viện chính

| Library | Mục đích |
|---|---|
| pandas | xử lý dữ liệu |
| numpy | tính toán |
| matplotlib | visualization |
| seaborn | visualization |
| networkx | graph construction |
| scikit-learn | machine learning |
| PyTorch Geometric | graph neural network |
| jupyter | notebook |

---

# 5. Cấu trúc dự án

graph-fraud-detection/
│
├── data/
│   └── PS_20174392719_1491204439457_log.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_graph_construction.ipynb
│   ├── 04_graph_features.ipynb
│   └── 05_fraud_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── graph_builder.py
│   ├── feature_engineering.py
│   └── model.py
│
├── results/
│
├── requirements.txt
│
└── README.md

---

# 6. Exploratory Data Analysis (EDA)

Trong giai đoạn EDA, dự án thực hiện:
- phân tích phân bố giao dịch
- phân tích class imbalance
- transaction type analysis
- balance analysis
- sender behavior analysis
- temporal analysis
- fraud pattern analysis

Các insight quan trọng:
- dữ liệu mất cân bằng nghiêm trọng
- fraud tập trung chủ yếu ở:
  - TRANSFER
  - CASH_OUT
- tồn tại transaction burst behavior
- xuất hiện nhiều account draining patterns

---

# 7. Graph Construction

Dữ liệu được chuyển đổi sang đồ thị có hướng:

- Node:
  - sender account
  - receiver account

- Edge:
  - transaction

- Edge features:
  - amount
  - transaction type
  - fraud label

Ví dụ:

```text
A → B
B → C
C → D
```

---

# 8. Graph Features

Các đặc trưng đồ thị được sử dụng:
- Degree Centrality
- Betweenness Centrality
- PageRank
- Community Detection
- Transaction Frequency
- Temporal Features

---

# 9. Fraud Detection Pipeline

Raw Dataset
      ↓
EDA
      ↓
Preprocessing
      ↓
Graph Construction
      ↓
Graph Feature Engineering
      ↓
Machine Learning / GNN
      ↓
Fraud Classification


---

# 10. Machine Learning Models

Các mô hình được sử dụng:
- Logistic Regression
- Random Forest
- XGBoost
- Graph Neural Networks (GCN/GAT)

---

# 11. Hướng phát triển

Các hướng mở rộng:
- Temporal Graph Networks
- Dynamic Fraud Detection
- Node Embedding (Node2Vec)
- Graph AutoEncoder
- Explainable AI for Fraud Detection

---

# 12. Cài đặt

## Clone project

```bash
git clone <repository-url>
```

---

## Tạo virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Cài dependencies

```bash
pip install -r requirements.txt
```

---

# 13. Chạy notebook

```bash
jupyter notebook
```

---

# 14. Kết quả mong đợi

Hệ thống có khả năng:
- Phát hiện giao dịch bất thường
- Xác định tài khoản đáng ngờ
- Tìm fraud communities
- Phân tích transaction network
- Hỗ trợ điều tra fraud

---

# 15. Kiến thức áp dụng

Dự án áp dụng:
- Data Mining
- Graph Mining
- Machine Learning
- Graph Neural Networks
- Fraud Analytics
- Network Analysis

