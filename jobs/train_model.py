import os
import sys

# On Windows, PyTorch/PyG and scikit-learn can load separate OpenMP runtimes.
# Set this before importing those libraries so the training script can start.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# ==========================================
# 1. ĐỊNH NGHĨA MÔ HÌNH VÀ HÀM MẤT MÁT
# ==========================================
class FraudGraphSAGE(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        in_dim = (hidden_dim * 2) + edge_in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col], edge_attr], dim=1)
        return self.mlp(edge_feat).squeeze()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = (self.alpha * targets) + ((1 - self.alpha) * (1 - targets))
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def normalize_edge_attr(data):
    train_edge_attr = data.edge_attr[data.train_mask]
    mean = train_edge_attr.mean(dim=0, keepdim=True)
    std = train_edge_attr.std(dim=0, keepdim=True).clamp_min(1e-6)
    data.edge_attr = (data.edge_attr - mean) / std
    return data

# ==========================================
# 2. HÀM ĐÁNH GIÁ (FULL BATCH)
# ==========================================
def evaluate_full_batch(model, data, mask, edge_index, edge_attr, threshold=0.5):
    model.eval()
    with torch.no_grad():
        # Truyền toàn bộ đồ thị vào mô hình
        logits = model(data.x, edge_index, edge_attr)
        
        # Chỉ lấy kết quả tại các vị trí mask tương ứng (Train/Val/Test)
        probs = torch.sigmoid(logits[mask])
        preds = (probs > threshold).float()
        
        y_true = data.edge_label[mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_probs = probs.cpu().numpy()
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr = precision_score(y_true, y_pred, zero_division=0)
        re = recall_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_probs)
        
    return f1, pr, re, pr_auc

def find_best_threshold(model, data, mask, edge_index, edge_attr):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, edge_index, edge_attr)[mask]).cpu().numpy()
        y_true = data.edge_label[mask].cpu().numpy()

    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in [i / 100 for i in range(1, 100)]:
        y_pred = (probs > threshold).astype("float32")
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": threshold,
                "f1": f1,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }

    return best

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
def train_full_batch(data_path, epochs=100, patience=15):
    print("===" * 15)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN GRAPH-SAGE (FULL-BATCH)")
    print("===" * 15)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Thiết bị xử lý: {device}")
    if device.type == 'cpu':
        print("⚠️ CẢNH BÁO: Đang chạy Full-batch trên CPU. Quá trình này sẽ tốn rất nhiều RAM và thời gian!")

    print("📥 Đang tải tensor đồ thị...")
    data = torch.load(data_path, map_location=device, weights_only=False)
    data = normalize_edge_attr(data)
    
    # --- XỬ LÝ CHỐNG RÒ RỈ DỮ LIỆU TƯƠNG LAI ---
    # Khi train, chúng ta TẠM THỜI ẩn các cạnh thuộc Val và Test đi
    # Chỉ cho phép mô hình Message Passing trên các cạnh Train
    train_edge_index = data.edge_index[:, data.train_mask]
    train_edge_attr = data.edge_attr[data.train_mask]
    
    model = FraudGraphSAGE(data.num_node_features, data.num_edge_features, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    best_val_f1 = -1.0
    best_threshold = 0.5
    epochs_no_improve = 0
    best_model_saved = False
    os.makedirs("data/reports", exist_ok=True)
    best_model_path = "data/reports/best_full_batch_model.pth"
    
    print("\n⏳ Bắt đầu tiến trình tối ưu hóa trọng số...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. Quét qua mô hình (CHỈ DÙNG CẤU TRÚC ĐỒ THỊ TRAIN)
        # Lưu ý: Kết quả logits xuất ra lúc này chỉ có độ dài bằng với số lượng cạnh Train
        logits = model(data.x, train_edge_index, train_edge_attr)
        
        # 2. Tính Loss trực tiếp trên toàn bộ kết quả vừa xuất ra
        loss = criterion(logits, data.edge_label[data.train_mask])
        
        # 3. Cập nhật Gradient
        loss.backward()
        optimizer.step()
        
        # 4. Đánh giá mỗi 5 epoch để tiết kiệm thời gian tính toán
        if epoch % 5 == 0:
            # Lúc đánh giá Train, dùng đồ thị Train
            train_f1, _, _, _ = evaluate_full_batch(model, data, data.train_mask, data.edge_index, data.edge_attr)
            # Lúc đánh giá Val, bắt buộc phải dùng đồ thị gốc (Full edge_index) để mô hình có đường đi
            threshold_metrics = find_best_threshold(model, data, data.val_mask, data.edge_index, data.edge_attr)
            val_f1, val_pr, val_re, val_prauc = evaluate_full_batch(
                model,
                data,
                data.val_mask,
                data.edge_index,
                data.edge_attr,
                threshold=threshold_metrics["threshold"],
            )
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_re:.4f} | Val PR-AUC: {val_prauc:.4f} | Threshold: {threshold_metrics['threshold']:.2f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold = threshold_metrics["threshold"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                best_model_saved = True
            else:
                epochs_no_improve += 5
                if epochs_no_improve >= patience:
                    print(f"🛑 Đã kích hoạt Dừng sớm tại Epoch {epoch}.")
                    break

    print("\n📊 BÁO CÁO HIỆU NĂNG CUỐI CÙNG (TEST SET):")
    if not best_model_saved:
        torch.save(model.state_dict(), best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    # Chạy trên tập Test bằng đồ thị đầy đủ
    test_f1, test_pr, test_re, test_prauc = evaluate_full_batch(
        model,
        data,
        data.test_mask,
        data.edge_index,
        data.edge_attr,
        threshold=best_threshold,
    )
    print(f"   - Threshold: {best_threshold:.2f}")
    print(f"   - Precision: {test_pr:.4f}")
    print(f"   - Recall:    {test_re:.4f}")
    print(f"   - F1-Score:  {test_f1:.4f}")
    print(f"   - PR-AUC:    {test_prauc:.4f}\n")

if __name__ == "__main__":
    GRAPH_DATA_PATH = "data/graph_data/graph_data.pt"
    train_full_batch(data_path=GRAPH_DATA_PATH, epochs=100, patience=20)
