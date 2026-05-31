import pandas as pd
import os

def load_and_filter_data(file_path):
    """
    Đọc file CSV và chỉ giữ lại các giao dịch có rủi ro cao (TRANSFER, CASH_OUT).
    Trong PaySim, gian lận KHÔNG bao giờ xảy ra ở các loại PAYMENT, DEBIT, CASH_IN.
    Việc lọc này giúp giảm hơn 50% dung lượng dữ liệu vô ích.
    """
    print(f"🔄 Đang đọc dữ liệu từ: {file_path}...")
    df = pd.read_csv(file_path)
    
    print("✂️ Đang lọc các giao dịch TRANSFER và CASH_OUT...")
    # Dùng .copy() để tránh cảnh báo SettingWithCopyWarning của Pandas sau này
    df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    return df_filtered

def map_account_ids(df):
    """
    Mạng GNN yêu cầu ID của Node phải là số nguyên liên tục (0, 1, 2...).
    Hàm này ánh xạ các chuỗi (VD: 'C12345') thành số nguyên.
    """
    print("🧮 Đang ánh xạ ID tài khoản sang số nguyên...")
    
    # Gom tất cả tài khoản gửi và nhận lại để tạo danh sách ID duy nhất
    all_accounts = pd.concat([df['nameOrig'], df['nameDest']]).unique()
    
    # Tạo dictionary để map: {'C123...': 0, 'C456...': 1, ...}
    account_map = {name: i for i, name in enumerate(all_accounts)}
    
    # Áp dụng map vào DataFrame để tạo 2 cột ID mới
    df['orig_id'] = df['nameOrig'].map(account_map)
    df['dest_id'] = df['nameDest'].map(account_map)
    
    print(f"✅ Đã tìm thấy {len(all_accounts):,} tài khoản duy nhất (Nodes).")
    print(f"✅ Đã giữ lại {len(df):,} giao dịch (Edges).")
    
    return df, account_map

def run_stage_1(raw_path, output_path):
    """
    Hàm thực thi chính cho Giai đoạn 1.
    """
    print("===" * 15)
    print("🚀 BẮT ĐẦU GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU")
    print("===" * 15)
    
    # Bước 1 & 2
    df_filtered = load_and_filter_data(raw_path)
    df_mapped, _ = map_account_ids(df_filtered)
    
    # Bước 3: Lưu trữ dữ liệu Silver
    print(f"💾 Đang lưu dữ liệu đã xử lý vào: {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_mapped.to_csv(output_path, index=False)
    
    print("🎉 HOÀN THÀNH GIAI ĐOẠN 1!")
    return df_mapped

# --- Khối chạy thử (chỉ chạy khi bạn gọi trực tiếp file này) ---
if __name__ == "__main__":
    # Lưu ý: Đường dẫn này giả định bạn đang chạy lệnh từ thư mục gốc của dự án
    RAW_FILE = "data/raw/paysim.csv"
    SILVER_FILE = "data/silver/paysim_filtered.csv"
    
    # Nếu file raw chưa có, code sẽ báo lỗi ngay. Đảm bảo bạn đã để file paysim.csv đúng chỗ.
    if os.path.exists(RAW_FILE):
        run_stage_1(RAW_FILE, SILVER_FILE)
    else:
        print(f"❌ Lỗi: Không tìm thấy file {RAW_FILE}. Hãy kiểm tra lại thư mục data/raw/!")