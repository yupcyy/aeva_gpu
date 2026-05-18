# visualize_perturbations_from_npy.py
import os
import numpy as np
import matplotlib.pyplot as plt

adv_dir = "cifar10_adv_per_0"   # 修改为你的 npy 目录
num_labels = 10


def concentration_scores(array):
    """
    array: (N, H, W, C) 的扰动，返回 shape=(N,) 的集中度评分
    """
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


def scan_all_pairs(adv_dir, num_labels=10):
    """
    返回:
      s: shape=(num_labels,) 的 GAP 向量，对应每个目标类
      V: shape=(num_labels, num_labels) 的矩阵，V[src, tgt] 是该方向的极值集中度
    """
    s = np.zeros(num_labels, dtype=np.float32)
    V = np.zeros((num_labels, num_labels), dtype=np.float32)

    for tgt in range(num_labels):
        src_labels = list(range(num_labels))
        src_labels.remove(tgt)

        for src in src_labels:
            path = os.path.join(adv_dir, f"data_{src}_{tgt}.npy")
            if not os.path.exists(path):
                continue
            arr = np.load(path)       # (N,H,W,C)
            arr = np.abs(arr)
            scores = concentration_scores(arr)
            v = float(np.max(scores))
            V[src, tgt] = v
            s[tgt] += v

    return s, V


def main():
    s, V = scan_all_pairs(adv_dir, num_labels)

    print("GAP 向量 s:", s)
    infected_target = 0

    # 为 infected_target=0，找出使 V[src,0] 最大的源类
    src_for_inf = int(np.argmax(V[:, infected_target]))
    print(f"[infected] target={infected_target}, chosen src={src_for_inf}, "
          f"V[src,tgt]={V[src_for_inf, infected_target]:.4f}")

    # 找一个“最正常”的目标类：s 最小且 != infected_target
    benign_candidates = [t for t in range(num_labels)
                         if t != infected_target and s[t] > 0]
    if not benign_candidates:
        raise RuntimeError("没有找到合适的 benign target（可能数据不完整）")

    benign_target = int(min(benign_candidates, key=lambda t: s[t]))
    src_for_benign = int(np.argmax(V[:, benign_target]))
    print(f"[benign]  target={benign_target}, chosen src={src_for_benign}, "
          f"V[src,tgt]={V[src_for_benign, benign_target]:.4f}")

    # 加载对应扰动
    def load_pair(src, tgt):
        path = os.path.join(adv_dir, f"data_{src}_{tgt}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        arr = np.load(path)
        return np.abs(arr)

    pert_inf = load_pair(src_for_inf, infected_target)
    pert_ben = load_pair(src_for_benign, benign_target)

    # 在这两个方向各自挑一张“最集中的样本”
    scores_inf = concentration_scores(pert_inf)
    idx_inf = int(np.argmax(scores_inf))
    mu_inf = pert_inf[idx_inf]
    heat_inf = np.mean(mu_inf, axis=-1)

    scores_ben = concentration_scores(pert_ben)
    idx_ben = int(np.argmax(scores_ben))
    mu_ben = pert_ben[idx_ben]
    heat_ben = np.mean(mu_ben, axis=-1)

    print(f"[infected] sample idx={idx_inf}, score={scores_inf[idx_inf]:.4f}")
    print(f"[benign  ] sample idx={idx_ben}, score={scores_ben[idx_ben]:.4f}")

    # 可视化
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(heat_inf, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Infected target={infected_target}\n"
              f"src={src_for_inf}, score={scores_inf[idx_inf]:.3f}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(heat_ben, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Benign target={benign_target}\n"
              f"src={src_for_benign}, score={scores_ben[idx_ben]:.3f}")
    plt.axis('off')

    plt.suptitle("Adversarial perturbation heatmaps (|mu|, channel-avg)")
    plt.tight_layout()

    # --------- 关键：保存图像 ---------
    out_path = "perturbations_heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存扰动热力图到: {out_path}")


if __name__ == "__main__":
    main()