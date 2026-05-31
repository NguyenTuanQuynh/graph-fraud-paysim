import torch
import pandas as pd
import os
from torch_geometric.data import Data

def run_stage_3(edge_path, node_path, output_path):
    print("===" * 15)
    print("🚀 BẮT ĐẦU GIAI ĐOẠN 3: ĐÓNG GÓI TENSOR ĐỒ THỊ (BẢN VÁ LỖI HOÀN CHỈNH)")
    print("===" * 15)
    
    # Đọc dữ liệu đặc trưng dạng bảng từ Giai đoạn 2
    df_edges = pd.read_csv(edge_path)
    df_nodes = pd.read_csv(node_path)
    
    # 1. Xử lý Đặc trưng Nút (Node Features - x)
    print("🧠 Đang chuyển đổi Ma trận Nút (x)...")
    # Loại bỏ cột index 'node_id', chỉ giữ lại các cột giá trị số học đã chuẩn hóa
    node_features = df_nodes.drop(columns=['node_id']).values
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 2. Xử lý Cấu trúc Đồ thị (edge_index) - ĐÃ SỬA LỖI CRASH BỘ NHỚ
    print("🔗 Đang xây dựng Ma trận Kết nối (edge_index)...")
    edge_index_np = df_edges[['orig_id', 'dest_id']].values.transpose()
    # Thêm .contiguous() để ép các phần tử xếp liên tục trong RAM/VRAM, chống lỗi khi đưa vào GPU
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).contiguous()
    
    # 3. Xử lý Đặc trưng Cạnh (Edge Attributes - edge_attr)
    print("📦 Đang chuyển đổi Đặc trưng Giao dịch (edge_attr)...")
    edge_attr_cols = ['log_amount', 'errorBalanceOrig', 'errorBalanceDest', 'is_transfer']
    edge_attr = torch.tensor(df_edges[edge_attr_cols].values, dtype=torch.float)
    
    # 4. Xử lý Nhãn Phân Loại (Labels) - ĐÃ SỬA LỖI XUNG ĐỘT PHÂN LỚP
    print("🎯 Đang trích xuất Nhãn phân loại cạnh (edge_label)...")
    # Đổi tên từ 'y' thành 'edge_label' giúp PyG hiểu đây là bài toán phân loại CẠNH chứ không phải NÚT
    edge_label = torch.tensor(df_edges['isFraud'].values, dtype=torch.float)
    
    # 5. Tích hợp Trục Thời Gian - ĐÃ SỬA LỖI RÒ RỈ DỮ LIỆU TƯƠNG LAI (DATA LEAKAGE)
    print("⏱️ Đang đồng bộ hóa trục thời gian giao dịch (edge_time)...")
    # Lưu lại cột step của từng cạnh dưới dạng Tensor để Loader ở GĐ 4 chặn thông tin từ tương lai
    edge_time = torch.tensor(df_edges['step'].values, dtype=torch.long)
    
    # 6. Tạo các mặt nạ phân chia dữ liệu (Temporal Masks)
    print("✂️ Đang thiết lập Bức tường thời gian chia tập dữ liệu...")
    steps = df_edges['step'].values
    train_mask = torch.tensor(steps <= 480, dtype=torch.bool)
    val_mask = torch.tensor((steps > 480) & (steps <= 600), dtype=torch.bool)
    test_mask = torch.tensor(steps > 600, dtype=torch.bool)
    
    # 7. Đóng gói tất cả cấu trúc vào đối tượng Data của PyTorch Geometric
    print("📦 Đang đóng gói dữ liệu thành đối tượng PyG Data...")
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        edge_label=edge_label,
        edge_time=edge_time
    )
    
    # Gán các mặt nạ phân hoạch thời gian vào đồ thị
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Khai báo tường minh số lượng Nút để hệ thống không phải tự suy đoán (Tránh Warnings)
    data.num_nodes = len(df_nodes)
    
    # 8. Lưu trữ dữ liệu dạng Tensor xuống ổ cứng (.pt)
    print(f"💾 Đang lưu PyG Data hoàn chỉnh vào: {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    
    # --- Báo cáo kiểm tra chất lượng Tensor độc lập ---
    print("\n📊 BÁO CÁO KIỂM TRA CHẤT LƯỢNG ĐỒ THỊ:")
    print(f"   - Số lượng Nút (Tài khoản): {data.num_nodes:,}")
    print(f"   - Số lượng Cạnh (Giao dịch): {data.num_edges:,}")
    print(f"   - Kiểm tra bộ nhớ edge_index liên tục: {data.edge_index.is_contiguous()}")
    print(f"   - Cấu trúc nhãn cạnh (edge_label) chuẩn hóa: {list(data.edge_label.shape)}")
    print(f"   - Đồng bộ thời gian cạnh (edge_time): {list(data.edge_time.shape)}")
    print(f"   - Số lượng giao dịch tập Huấn luyện (Train): {data.train_mask.sum().item():,}")
    print(f"   - Số lượng giao dịch tập Kiểm định (Val): {data.val_mask.sum().item():,}")
    print(f"   - Số lượng giao dịch tập Đánh giá (Test): {data.test_mask.sum().item():,}\n")
    print("🎉 HOÀN THÀNH GIAI ĐOẠN 3 HOÀN HẢO VÀ SẴN SÀNG CHO HỌC SÂU!")

run_stage_3(
    edge_path="data/features/edge_features.csv",
    node_path="data/features/node_features.csv",
    output_path="data/graph_data/graph_data.pt")
