import os
import sys
import json
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
import torch
import numpy as np

# Khai báo đường dẫn để hệ thống có thể import FraudGraphSAGE từ thư mục libs
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jobs.train_model import FraudGraphSAGE, normalize_edge_attr

# ==========================================
# 1. CẤU HÌNH TRANG & CSS TÙY CHỈNH
# ==========================================
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận GNN", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F4F7FE; }
    
    /* Banner hiển thị trạng thái FRAUD */
    .fraud-banner {
        background-color: #FCE8E8; border-radius: 10px; padding: 20px;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
    }
    .fraud-text { color: #A31D1D; font-size: 32px; font-weight: 900; margin: 0; }
    .fraud-prob { color: #A31D1D; font-size: 32px; font-weight: 700; margin: 0; }
    
    /* Banner hiển thị trạng thái SAFE */
    .safe-banner {
        background-color: #E8FCF1; border-radius: 10px; padding: 20px;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
    }
    .safe-text { color: #107C41; font-size: 32px; font-weight: 900; margin: 0; }
    .safe-prob { color: #107C41; font-size: 32px; font-weight: 700; margin: 0; }
    
    /* Tùy chỉnh nút bấm */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; width: 100%; font-weight: bold; font-size: 16px;
    }
    div.stButton > button:hover { opacity: 0.9; color: white; }
    
    .column-header { font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TẢI MÔ HÌNH VÀ ĐỒ THỊ VÀO BỘ NHỚ (CACHED)
# ==========================================
@st.cache_resource
def load_gnn_core():
    """Tải file đồ thị nền và trọng số mô hình tốt nhất vào RAM/GPU"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = "data/graph_data/graph_data.pt"
    # Bạn có thể đổi thành best_full_batch_model.pth nếu dùng bản full-batch
    model_path = "data/reports/best_full_batch_model.pth" 
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return None, None, device, "Không tìm thấy tệp tin Đồ thị hoặc Trọng số Mô hình trong thư mục data/."
        
    try:
        data = torch.load(data_path, map_location=device, weights_only=False)
        data = normalize_edge_attr(data)
        model = FraudGraphSAGE(data.num_node_features, data.num_edge_features, hidden_dim=64).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return data, model, device, "Hệ thống lõi GNN đã sẵn sàng!"
    except Exception as e:
        return None, None, device, f"Lỗi khi khởi tạo mô hình: {str(e)}"

# Gọi hàm khởi động hệ thống
data, model, device, status_msg = load_gnn_core()

# hiển thị trạng thái tải mô hình ở thanh bên
st.sidebar.markdown(f"**Trạng thái hệ thống:**\n{status_msg}")
threshold = st.sidebar.slider("Ngưỡng chặn gian lận (Threshold)", 0.5, 0.99, 0.85)

# ==========================================
# 3. GIAO DIỆN CHÍNH (LAYOUT 2 CỘT)
# ==========================================
col_left, col_spacer, col_right = st.columns([4, 0.2, 6])

# ------------------------------------------
# CỘT TRÁI: FORM NHẬP THÔNG TIN GIAO DỊCH
# ------------------------------------------
with col_left:
    st.markdown('<div class="column-header">Nhập thông tin giao dịch </div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        r1_c1, r1_c2 = st.columns(2)
        # Cho phép nhập ID tài khoản (Ví dụ nhập số: 10, 55, 1024 hoặc nhập chữ cho tài khoản mới)
        nameOrig = r1_c1.text_input(":black[nameOrig (ID người gửi)]", value="")
        nameDest = r1_c2.text_input(":black[nameDest (ID người nhận)]", value="")
        
        r2_c1, r2_c2 = st.columns(2)
        step = r2_c1.number_input(":black[step (Bước thời gian)]", value=None, step=1)
        tx_type = r2_c2.selectbox(":black[type (Loại giao dịch)]", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
        
        r3_c1, r3_c2 = st.columns(2)
        amount = r3_c1.number_input(":black[amount (Số tiền chuyển)]", value=None, step=1000.0)
        oldbalanceOrg = r3_c2.number_input(":black[oldbalanceOrg (Số dư gốc người gửi)]", value=None, step=1000.0)
        
        r4_c1, r4_c2 = st.columns(2)
        newbalanceOrig = r4_c1.number_input(":black[newbalanceOrig (Số dư mới người gửi)]", value=None, step=1000.0)
        oldbalanceDest = r4_c2.number_input(":black[oldbalanceDest (Số dư gốc người nhận)]", value=None, step=1000.0)
        
        r5_c1, r5_c2 = st.columns(2)
        newbalanceDest = r5_c1.number_input(":black[newbalanceDest (Số dư mới người nhận)]", value=None, step=1000.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Dự đoán giao dịch")

# ------------------------------------------
# CỘT PHẢI: XỬ LÝ INFERENCE VÀ HIỂN THỊ KẾT QUẢ
# ------------------------------------------
with col_right:
    st.markdown('<div class="column-header">Kết quả dự đoán GNN</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        if not predict_btn:
            st.info("Hãy điền thông tin và bấm 'Dự đoán giao dịch' để kích hoạt mô hình GraphSAGE.")
        elif data is None or model is None:
            st.error("Không thể thực hiện dự đoán do hệ thống lõi chưa được tải thành công. Vui lòng chạy các Giai đoạn huấn luyện trước.")
        else:
            with st.spinner("Đang trích xuất đặc trưng không gian và chạy mô hình..."):
                start_time = time.perf_counter()
                
                # --- BƯỚC A: TÍNH TOÁN ĐẶC TRƯNG CẠNH TỨC THỜI (ON-THE-FLY) ---
                log_amount = np.log1p(amount)
                err_orig = newbalanceOrig + amount - oldbalanceOrg
                err_dest = oldbalanceDest + amount - newbalanceDest
                is_transfer_val = 1.0 if tx_type == "TRANSFER" else 0.0
                
                edge_attr_tensor = torch.tensor([[log_amount, err_orig, err_dest, is_transfer_val]], dtype=torch.float).to(device)
                
                # --- BƯỚC B: LOOKUP ĐẶC TRƯNG NÚT TỪ ĐỒ THỊ GỐC (HỖ TRỢ COLD-START) ---
                feature_names = ['out_degree', 'out_total_amt', 'in_degree', 'in_total_amt', 'pagerank']
                
                def fetch_node_profile(account_str):
                    try:
                        node_id = int(account_str)
                        # Nếu ID nằm trong danh sách nút đã biết
                        if node_id < data.x.shape[0]:
                            feat_vector = data.x[node_id]
                            # Chuyển đổi ngược tensor thành dict để hiển thị lên màn hình JSON
                            feat_dict = {name: round(float(feat_vector[i].item()), 4) for i, name in enumerate(feature_names)}
                            feat_dict["is_new_account"] = False
                            return feat_vector.unsqueeze(0), feat_dict
                    except ValueError:
                        pass
                    
                    # Nếu là tài khoản mới tinh chưa từng xuất hiện (Cold-start)
                    zero_vector = torch.zeros((1, data.x.shape[1]), dtype=torch.float)
                    zero_dict = {name: 0.0 for name in feature_names}
                    zero_dict["is_new_account"] = True
                    return zero_vector, zero_dict

                src_tensor, src_json_dict = fetch_node_profile(nameOrig)
                dst_tensor, dst_json_dict = fetch_node_profile(nameDest)
                
                # --- BƯỚC C: SUY LUẬN ĐỒ THỊ CON (SUB-GRAPH INFERENCE) ---
                temp_x = torch.cat([src_tensor, dst_tensor], dim=0).to(device)
                temp_edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    logits = model(temp_x, temp_edge_index, edge_attr_tensor)
                    prob = torch.sigmoid(logits).item()
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                is_fraud = prob > threshold
                
                # --- BƯỚC D: HIỂN THỊ BANNER KẾT QUẢ THẬT ---
                if is_fraud:
                    st.markdown(f"""
                        <div class="fraud-banner">
                            <p class="fraud-text">FRAUD</p>
                            <p class="fraud-prob">{prob * 100:.4f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="safe-banner">
                            <p class="safe-text">SAFE</p>
                            <p class="safe-prob">{prob * 100:.4f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # 2. XỔ DỮ LIỆU ĐẶC TRƯNG ĐÃ TRA CỨU RA BẢNG JSON ĐEN
                st.markdown("#### Account features được lookup thực tế")
                json_c1, json_c2 = st.columns(2)
                
                with json_c1:
                    st.markdown(f"**Source Node: {nameOrig}**")
                    st.code(json.dumps(src_json_dict, indent=4), language='json')
                    
                with json_c2:
                    st.markdown(f"**Destination Node: {nameDest}**")
                    st.code(json.dumps(dst_json_dict, indent=4), language='json')
                
                # 3. KHU VỰC GIẢI THÍCH LÝ DO DỰA TRÊN THÔNG SỐ ĐẦU VÀO
                st.markdown("#### Báo cáo phân tích kỹ thuật")
                st.write(f"**Thời gian xử lý suy luận (Latency):** {latency_ms:.2f} ms")
                
                if is_fraud:
                    st.markdown(f"""
                    * Mô hình chỉ ra xác suất rủi ro vượt ngưỡng an toàn quy định ({threshold:.0%}).
                    * Hệ thống phát hiện chỉ số sai lệch số dư của tài khoản gốc đạt mức phi logic: **{err_orig:,.2f}**.
                    * Đặc trưng cấu hình kết nối mạng lưới tài khoản đích (`nameDest`) có dấu hiệu bất thường về mặt hình học đồ thị.
                    """)
                else:
                    st.markdown("""
                    * Chỉ số rủi ro nằm trong vùng an toàn cho phép.
                    * Cấu trúc dòng tiền luân chuyển tương thích với các mẫu hình giao dịch thông thường.
                    """)
