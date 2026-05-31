import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
import os

def extract_edge_features(df):
    """
    Trích xuất và biến đổi các đặc trưng cho mỗi Giao dịch (Cạnh)
    """
    print("🔪 Đang trích xuất Đặc trưng Cạnh (Edge Features)...")
    
    # 1. Logic Kinh tế: Lỗi số dư (Balance Errors)
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 2. Chuẩn hóa số tiền: Hàm Logarit giúp thu hẹp khoảng cách giữa tiền lẻ và tiền tỷ
    # Dùng log1p (log(1+x)) để tránh lỗi log(0)
    df['log_amount'] = np.log1p(df['amount'])
    
    # 3. Mã hóa loại giao dịch (CASH_OUT = 0, TRANSFER = 1)
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    
    # Lọc ra các cột cần thiết cho Cạnh
    edge_cols = ['orig_id', 'dest_id', 'step', 'log_amount', 'errorBalanceOrig', 'errorBalanceDest', 'is_transfer', 'isFraud']
    df_edges = df[edge_cols].copy()
    
    return df_edges

def extract_node_features(df):
    """
    Trích xuất các đặc trưng cấu trúc và hành vi cho mỗi Tài khoản (Nút)
    """
    print("🕸️ Đang trích xuất Đặc trưng Nút (Node Features)... (Có thể mất 1-2 phút)")
    
    # Xác định tổng số Node (tài khoản)
    num_nodes = max(df['orig_id'].max(), df['dest_id'].max()) + 1
    node_df = pd.DataFrame({'node_id': range(num_nodes)})
    
    # --- 1. Đặc trưng Hành vi (Dùng Pandas) ---
    # Tài khoản gửi (Out)
    out_stats = df.groupby('orig_id').agg(
        out_degree=('dest_id', 'count'),
        out_total_amt=('amount', 'sum')
    ).reset_index().rename(columns={'orig_id': 'node_id'})
    
    # Tài khoản nhận (In)
    in_stats = df.groupby('dest_id').agg(
        in_degree=('orig_id', 'count'),
        in_total_amt=('amount', 'sum')
    ).reset_index().rename(columns={'dest_id': 'node_id'})
    
    # Gộp lại
    node_df = node_df.merge(out_stats, on='node_id', how='left')
    node_df = node_df.merge(in_stats, on='node_id', how='left').fillna(0)
    
    # --- 2. Đặc trưng Cấu trúc (Dùng NetworkX) ---
    print("   -> Đang xây dựng đồ thị NetworkX để tính PageRank...")
    G = nx.from_pandas_edgelist(df, source='orig_id', target='dest_id', create_using=nx.DiGraph())
    
    print("   -> Đang chạy thuật toán PageRank...")
    pagerank_dict = nx.pagerank(G, alpha=0.85)
    node_df['pagerank'] = node_df['node_id'].map(pagerank_dict).fillna(0)
    
    # --- 3. Chuẩn hóa dữ liệu (Standard Scaling) ---
    print("⚖️ Đang chuẩn hóa (Scale) các đặc trưng Nút...")
    features_to_scale = ['out_degree', 'out_total_amt', 'in_degree', 'in_total_amt', 'pagerank']
    
    scaler = StandardScaler()
    node_df[features_to_scale] = scaler.fit_transform(node_df[features_to_scale])
    
    return node_df

def run_stage_2(input_path, edge_out_path, node_out_path):
    print("===" * 15)
    print("🚀 BẮT ĐẦU GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG")
    print("===" * 15)
    
    df = pd.read_csv(input_path)
    
    # Trích xuất
    df_edges = extract_edge_features(df)
    df_nodes = extract_node_features(df)
    
    # Lưu kết quả
    print("💾 Đang lưu kết quả vào thư mục data/features/...")
    os.makedirs(os.path.dirname(edge_out_path), exist_ok=True)
    
    df_edges.to_csv(edge_out_path, index=False)
    df_nodes.to_csv(node_out_path, index=False)
    
    print(f"✅ Đã lưu Đặc trưng Cạnh ({len(df_edges):,} dòng)")
    print(f"✅ Đã lưu Đặc trưng Nút ({len(df_nodes):,} dòng)")
    print("🎉 HOÀN THÀNH GIAI ĐOẠN 2!")

# --- Khối chạy thử ---
if __name__ == "__main__":
    INPUT_FILE = "data/silver/paysim_filtered.csv"
    EDGE_FILE = "data/features/edge_features.csv"
    NODE_FILE = "data/features/node_features.csv"
    
    if os.path.exists(INPUT_FILE):
        run_stage_2(INPUT_FILE, EDGE_FILE, NODE_FILE)
    else:
        print(f"❌ Lỗi: Không tìm thấy file {INPUT_FILE}. Vui lòng chạy Giai đoạn 1 trước!")