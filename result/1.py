# plot_gap_from_npy.py
import os
import numpy as np
import matplotlib.pyplot as plt

adv_dir = "cifar10_adv_per"   # 修改成你实际存放 90 个 npy 的目录
num_labels = 10                 # CIFAR-10

def concentration_scores(array):
    N = array.shape[0]
    scores = np.empty(N, dtype=np.float32)
    for i in range(N):
        img = np.sum(np.abs(array[i]), axis=-1)  # (H, W)
        flat = img.flatten()
        if flat.size < 2:
            scores[i] = 0.0
            continue
        flat_sorted = np.sort(flat)
        top2_sum = np.sum(flat_sorted[-2:])
        total_sum = np.sum(img) + 1e-12
        scores[i] = top2_sum / total_sum
    return scores


def compute_gap_from_dir(adv_dir, num_labels=10):
    s = np.zeros(num_labels, dtype=np.float32)
    coll = np.zeros((num_labels, num_labels - 1), dtype=np.float32)

    for tgt in range(num_labels):           # 目标类
        src_labels = list(range(num_labels))
        src_labels.remove(tgt)              # 所有源类

        for idx, src in enumerate(src_labels):
            path = os.path.join(adv_dir, f"data_{src}_{tgt}.npy")
            if not os.path.exists(path):
                continue

            arr = np.load(path)             # (N,H,W,C)
            arr = np.abs(arr)
            scores = concentration_scores(arr)
            v = float(np.max(scores))
            coll[tgt, idx] = v
            s[tgt] += v

    return s, coll


def main():
    s, coll = compute_gap_from_dir(adv_dir, num_labels)
    labels = np.arange(num_labels)

    # -------- 正确的 MAD / 异常指数计算（对齐你 analomy()） --------
    consistency_constant = 1.4826
    median = np.median(s)
    mad_raw = np.median(np.abs(s - median))
    mad = consistency_constant * mad_raw          # 和你 analomy() 一样

    if mad == 0:
        mad = 1e-12

    # 保留符号：谁大谁小一眼能看出
    anomaly_index = (s - median) / mad
    tau = 4.0

    # GAP 空间的阈值带：|s - M| > tau * mad
    delta_gap = tau * mad
    gap_upper = median + delta_gap
    gap_lower = median - delta_gap
    # --------------------------------------------------

    print("GAP 向量 s:", s)
    print("中位数 M:", median)
    print("MAD_raw:", mad_raw)
    print("MAD (scaled):", mad)
    print("异常指数 A_i:", anomaly_index)
    print("阈值 tau:", tau)
    print("GAP 阈值带: [", gap_lower, ",", gap_upper, "]")

    plt.figure(figsize=(8, 4))

    infected_label = 0
    bar_colors = ["C3" if c == infected_label else "C0"
                  for c in labels]

    plt.bar(labels, s, color=bar_colors)
    plt.xlabel("Class label")
    plt.ylabel("GAP R(y_t)")
    plt.xticks(labels)

    # 中位数 + 阈值区间
    plt.axhline(median, color='gray', linestyle='--', linewidth=1.0, label='median')
    plt.axhline(gap_upper, color='orange', linestyle='--', linewidth=1.0, label='MAD band')
    plt.axhline(gap_lower, color='orange', linestyle='--', linewidth=1.0)

    # 每个类上标注异常指数（保留正负号）
    for i, v in enumerate(s):
        plt.text(i, v, f"{anomaly_index[i]:.1f}", ha='center', va='bottom', fontsize=8)

    plt.title("GAP distribution (infected class = 0, tau = 4)")
    plt.legend()
    plt.tight_layout()

    out_path = "gap_model.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存 GAP 图到: {out_path}")


if __name__ == "__main__":
    main()