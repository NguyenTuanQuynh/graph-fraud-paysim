
import numpy as np
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
# Import mô hình từ thư mục libs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jobs.train_model import FraudGraphSAGE

# ==========================================
# 1. KHỞI TẠO ỨNG DỤNG VÀ STATE
# ==========================================
app = FastAPI(
    title="PaySim GNN Anti-Fraud API",
    description="Inference Pipeline thời gian thực cho hệ thống phát hiện rửa tiền.",
    version="1.0.0"
)

# Biến toàn cục lưu trữ Model và Data
system_state = {}

@app.on_event("startup")
async def load_model_and_graph():
    """Tải Đồ thị và Mô hình vào RAM/VRAM ngay khi server khởi động"""
    print("Đang khởi động Inference Engine...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = "data/graph_data/graph_data.pt"
    model_path = "data/reports/best_full_batch_model.pth"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        raise RuntimeError("Không tìm thấy Graph Data hoặc Model Weights!")
        
    data = torch.load(data_path, map_location=device, weights_only=False)
    model = FraudGraphSAGE(data.num_node_features, data.num_edge_features, 64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    system_state['data'] = data
    system_state['model'] = model
    system_state['device'] = device
    system_state['num_features'] = data.x.shape[1]
    system_state['total_nodes'] = data.x.shape[0]
    
    print("Inference Engine đã sẵn sàng nhận Request!")

# ==========================================
# 2. ĐỊNH NGHĨA PAYLOAD (DATA SCHEMA)
# ==========================================
class TransactionPayload(BaseModel):
    tx_id: str
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

# ==========================================
# 3. ENDPOINT XỬ LÝ GIAO DỊCH
# ==========================================
@app.post("/api/v1/predict")
async def predict_fraud(tx: TransactionPayload):
    start_time = time.perf_counter()
    
    data = system_state['data']
    model = system_state['model']
    device = system_state['device']
    
    try:
        # --- BƯỚC 1: FEATURE ENGINEERING ON-THE-FLY ---
        # Tính toán đặc trưng Cạnh (Giao dịch)
        log_amount = np.log1p(tx.amount)
        err_orig = tx.newbalanceOrig + tx.amount - tx.oldbalanceOrg
        err_dest = tx.oldbalanceDest + tx.amount - tx.newbalanceDest
        is_transfer = 1.0 if tx.type == 'TRANSFER' else 0.0
        
        edge_attr = torch.tensor([[log_amount, err_orig, err_dest, is_transfer]], dtype=torch.float).to(device)
        
        # --- BƯỚC 2: TRUY XUẤT ĐẶC TRƯNG NÚT ---
        # Hàm tra cứu ID: Trong thực tế, bạn sẽ map string ID (tx.nameOrig) sang Integer ID thông qua Redis/Database.
        # Ở demo này, ta giả lập việc parse Integer ID từ string (VD: account '105' -> id 105). 
        # Nếu lỗi (User lạ hoắc), gán vector Zero.
        def get_node_feature(account_id_str):
            try:
                # Giả lập tra cứu ID
                node_id = int(account_id_str)
                if node_id < system_state['total_nodes']:
                    return data.x[node_id].unsqueeze(0)
            except ValueError:
                pass
            # Trả về Vector 0 cho người dùng chưa từng tồn tại (Khởi động lạnh)
            return torch.zeros((1, system_state['num_features']), dtype=torch.float)

        src_feat = get_node_feature(tx.nameOrig).to(device)
        dst_feat = get_node_feature(tx.nameDest).to(device)
        
        # --- BƯỚC 3: ĐÓNG GÓI SUB-GRAPH VÀ DỰ ĐOÁN ---
        temp_x = torch.cat([src_feat, dst_feat], dim=0)
        temp_edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(device) # Cạnh nối từ Nút 0 (src) -> Nút 1 (dst)
        
        with torch.no_grad():
            logits = model(temp_x, temp_edge_index, edge_attr)
            prob = torch.sigmoid(logits).item()
            
        # --- BƯỚC 4: QUYẾT ĐỊNH VÀ TRẢ KẾT QUẢ ---
        latency_ms = (time.perf_counter() - start_time) * 1000
        is_fraud = prob > 0.85 # Ngưỡng chặn
        
        return {
            "transaction_id": tx.tx_id,
            "fraud_probability": round(prob, 4),
            "action": "BLOCK" if is_fraud else "ALLOW",
            "latency_ms": round(latency_ms, 2),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chạy server khi test cục bộ (Optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_pipeline:app", host="0.0.0.0", port=8000, reload=True)