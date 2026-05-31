import time
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

# Khai báo đường dẫn để import model từ thư mục libs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jobs.train_model import FraudGraphSAGE, normalize_edge_attr

def load_system(data_path, model_path):
    print("Đang khởi động Hệ thống AML Core...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tải Đồ thị
    data = torch.load(data_path, map_location=device, weights_only=False)
    data = normalize_edge_attr(data)
    
    # Khởi tạo và tải Trọng số mô hình
    model = FraudGraphSAGE(data.num_node_features, data.num_edge_features, 64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() 
    
    return data, model, device

def run_realtime_inference(data_path, model_path, max_transactions=1000):
    print("===" * 15)
    print("BẮT ĐẦU CHẠY LUỒNG STREAMING THỜI GIAN THỰC")
    print("===" * 15)
    
    data, model, device = load_system(data_path, model_path)
    
    # Lấy danh sách các giao dịch thuộc tập Test (Tương lai)
    test_indices = data.test_mask.nonzero(as_tuple=False).view(-1)
    print(f"Nguồn dữ liệu đã sẵn sàng. Có {len(test_indices):,} giao dịch đang chờ.")
    print("\nNhấn Ctrl+C để dừng hệ thống...\n")
    print(f"{'TX_ID':<10} | {'SENDER':<10} | {'RECEIVER':<10} | {'PREDICTION':<15} | {'ACTUAL':<10} | {'LATENCY'}")
    print("-" * 75)
    
    alerts = 0
    total_time = 0.0
    processed = 0
    
    try:
        with torch.no_grad():
            for idx in test_indices:
                if processed >= max_transactions:
                    break
                    
                start_time = time.perf_counter()
                
                # Trích xuất Nút gửi và Nút nhận
                src_node = data.edge_index[0, idx]
                dst_node = data.edge_index[1, idx]
                
                # Lấy nhãn thực tế (Đã vá lỗi: dùng edge_label thay vì y)
                actual_label = data.edge_label[idx]
                
                # Tạo Sub-graph siêu nhỏ (Chỉ gồm 2 nút và 1 cạnh hiện tại) để suy luận siêu tốc
                temp_x = torch.cat([data.x[src_node].unsqueeze(0), data.x[dst_node].unsqueeze(0)], dim=0)
                temp_edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(device)
                temp_edge_attr = data.edge_attr[idx].unsqueeze(0).to(device)
                
                # Chạy mô hình dự đoán
                logits = model(temp_x, temp_edge_index, temp_edge_attr)
                prob = torch.sigmoid(logits).item()
                prediction = 1 if prob > 0.5 else 0
                
                latency = (time.perf_counter() - start_time) * 1000 
                total_time += latency
                processed += 1
                
                # Hiển thị
                pred_text = "FRAUD" if prediction == 1 else "VALID"
                actual_text = "FRAUD" if actual_label.item() == 1 else "VALID"
                
                # Chỉ in ra màn hình các giao dịch bị chặn hoặc mỗi 50 giao dịch hợp lệ
                if prediction == 1 or processed % 50 == 0:
                    print(f"TX_{idx.item():<7} | N_{src_node.item():<8} | N_{dst_node.item():<8} | {pred_text:<15} | {actual_text:<10} | {latency:.2f} ms")
                    
                    if prediction == 1:
                        alerts += 1
                        time.sleep(0.1) 
                
    except KeyboardInterrupt:
        print("\nHệ thống bị người dùng dừng đột ngột.")
        
    print("\nBÁO CÁO HIỆU NĂNG STREAMING:")
    print(f"   - Số giao dịch đã quét: {processed:,}")
    print(f"   - Số cảnh báo Gian lận (Chặn): {alerts:,}")
    print(f"   - Tốc độ xử lý trung bình: {total_time/processed:.2f} mili-giây/giao dịch")

if __name__ == "__main__":
    # Đảm bảo đường dẫn này trỏ đúng vào model bạn đã huấn luyện thành công ở GĐ4
    DATA_FILE = "data/graph_data/graph_data.pt"
    MODEL_FILE = "data/reports/best_full_batch_model.pth" 
    
    if os.path.exists(DATA_FILE) and os.path.exists(MODEL_FILE):
        run_realtime_inference(DATA_FILE, MODEL_FILE, max_transactions=2000)
    else:
        print("Lỗi: Thiếu dữ liệu đồ thị hoặc file Trọng số. Vui lòng chạy Giai đoạn 3 và 4 trước.")
