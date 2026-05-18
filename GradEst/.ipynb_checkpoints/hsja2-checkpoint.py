# hsja.py
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------
# GPU版 Decision Function
# -------------------------------------------------------------
@tf.function
def decision_function_tf(model, images, original_label, target_label=-1,
                         clip_min=0.0, clip_max=1.0):
    """
    images: tf.Tensor, shape (N,H,W,C) 或 (H,W,C)
    返回: 布尔mask (N,), 表示是否落在目标分类侧
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    if tf.rank(images) == 3:
        images = tf.expand_dims(images, 0)
    images = tf.clip_by_value(images, clip_min, clip_max)
    logits = model(images, training=False)
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)

    orig_tf = tf.cast(original_label, tf.int32)
    tgt_tf = tf.cast(target_label, tf.int32)

    def untargeted_case():
        return tf.not_equal(preds, orig_tf)

    def targeted_case():
        return tf.equal(preds, tgt_tf)

    mask = tf.cond(tgt_tf < 0, untargeted_case, targeted_case)
    return mask


# -------------------------------------------------------------
# 距离函数
# -------------------------------------------------------------
def compute_distance(x_ori, x_pert, constraint='l2'):
    if constraint == 'l2':
        return np.linalg.norm((x_ori - x_pert).reshape(-1))
    else:
        return np.max(np.abs(x_ori - x_pert))


# -------------------------------------------------------------
# 初始化：寻找初始扰动样本
# -------------------------------------------------------------
def initialize(sample_np, model, original_label, target_image=None,
               target_label=-1, clip_min=0.0, clip_max=1.0,
               max_tries=10000, batch_init=32):
    """
    找到一个初始误分类样本。
    """
    if target_image is not None:
        return target_image.astype(np.float32)

    sample_np = sample_np.astype(np.float32)
    tries = 0
    while tries < max_tries:
        noises = np.random.uniform(clip_min, clip_max, size=(batch_init,) + sample_np.shape).astype(np.float32)
        mask = decision_function_tf(model, noises, original_label, target_label, clip_min, clip_max)
        mask_np = mask.numpy()
        success_idx = np.where(mask_np)[0]
        if success_idx.size > 0:
            rn = noises[success_idx[0]]
            # 二分精调靠近边界
            low, high = 0.0, 1.0
            while high - low > 1e-3:
                mid = (low + high) / 2.0
                blended = (1 - mid) * sample_np + mid * rn
                m = decision_function_tf(model, blended[None, ...], original_label, target_label, clip_min, clip_max).numpy()[0]
                if m:
                    high = mid
                else:
                    low = mid
            init = (1 - high) * sample_np + high * rn
            return init.astype(np.float32)
        tries += batch_init
    raise RuntimeError("initialize() failed — increase max_tries or provide target_image")


# -------------------------------------------------------------
# 近似梯度估计（GPU并行）
# -------------------------------------------------------------
def approximate_gradient_tf(model, sample, num_evals, delta,
                            original_label, target_label=-1,
                            clip_min=0.0, clip_max=1.0,
                            constraint='l2', eval_batch_size=256):
    if isinstance(sample, np.ndarray):
        sample_np = sample.astype(np.float32)
    else:
        sample_np = np.array(sample, dtype=np.float32)
    if sample_np.ndim == 3:
        sample_np = sample_np[None, ...]

    sample_tf = tf.convert_to_tensor(sample_np, dtype=tf.float32)
    delta_tf = tf.cast(delta, tf.float32)
    clip_min_tf = tf.cast(clip_min, tf.float32)
    clip_max_tf = tf.cast(clip_max, tf.float32)

    H, W, C = sample_np.shape[1:]
    total = int(num_evals)
    acc = np.zeros((H, W, C), dtype=np.float32)

    processed = 0
    while processed < total:
        bs = min(eval_batch_size, total - processed)
        if constraint == 'l2':
            rv_tf = tf.random.normal((bs, H, W, C), dtype=tf.float32)
            rv_tf = rv_tf / (tf.sqrt(tf.reduce_sum(tf.square(rv_tf), axis=(1, 2, 3), keepdims=True)) + 1e-12)
        else:
            rv_tf = tf.random.uniform((bs, H, W, C), -1.0, 1.0, dtype=tf.float32)
            rv_tf = rv_tf / (tf.reduce_max(tf.abs(rv_tf), axis=(1, 2, 3), keepdims=True) + 1e-12)

        sample_bs = tf.repeat(sample_tf, repeats=[bs], axis=0)
        perturbed_tf = tf.clip_by_value(sample_bs + delta_tf * rv_tf, clip_min_tf, clip_max_tf)
        mask_tf = decision_function_tf(model, perturbed_tf, original_label, target_label, clip_min, clip_max)
        mask_np = mask_tf.numpy().astype(np.float32)
        fval = (2.0 * mask_np.reshape(bs, 1, 1, 1) - 1.0)
        grad_chunk = np.mean(fval * rv_tf.numpy(), axis=0)
        acc += grad_chunk * (bs / total)
        processed += bs

    norm = np.linalg.norm(acc.reshape(-1))
    if norm > 0:
        acc = acc / (norm + 1e-12)
    return tf.convert_to_tensor(acc[None, ...], dtype=tf.float32)


# -------------------------------------------------------------
# 二分搜索（GPU并行+shape_invariants）
# -------------------------------------------------------------
@tf.function
def binary_search_batch_parallel_tf(sample_np, perturbed_candidates_np, model,
                                    original_label, target_label=-1,
                                    constraint='l2', tol=1e-5,
                                    clip_min=0.0, clip_max=1.0):
    sample_tf = tf.convert_to_tensor(sample_np[None, ...], dtype=tf.float32)
    candidates_tf = tf.convert_to_tensor(perturbed_candidates_np, dtype=tf.float32)
    N = tf.shape(candidates_tf)[0]

    low = tf.zeros((N,), dtype=tf.float32)
    high = tf.ones((N,), dtype=tf.float32)

    def cond(low, high):
        return tf.reduce_any(high - low > tol)

    def body(low, high):
        mid = (high + low) / 2.0
        mid_exp = tf.reshape(mid, (-1, 1, 1, 1))
        blended = (1.0 - mid_exp) * sample_tf + mid_exp * candidates_tf
        blended = tf.clip_by_value(blended, clip_min, clip_max)
        mask = decision_function_tf(model, blended, original_label, target_label, clip_min, clip_max)
        highs_new = tf.where(mask, mid, high)
        lows_new = tf.where(mask, low, mid)
        return lows_new, highs_new

    low_final, high_final = tf.while_loop(
        cond, body, [low, high],
        parallel_iterations=1,
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])]
    )

    high_exp = tf.reshape(high_final, (-1, 1, 1, 1))
    perturbed_on_boundary = (1.0 - high_exp) * sample_tf + high_exp * candidates_tf
    perturbed_on_boundary = tf.clip_by_value(perturbed_on_boundary, clip_min, clip_max)
    return perturbed_on_boundary, high_final


# -------------------------------------------------------------
# 包装函数（单样本）
# -------------------------------------------------------------
def binary_search_batch(sample_np, perturbed_candidate_np, model,
                        original_label, target_label=-1,
                        constraint='l2', tol=1e-5,
                        clip_min=0.0, clip_max=1.0):
    cand_np = np.expand_dims(perturbed_candidate_np.astype(np.float32), axis=0)
    perturbed_tf, dist_tf = binary_search_batch_parallel_tf(
        sample_np=sample_np,
        perturbed_candidates_np=cand_np,
        model=model,
        original_label=original_label,
        target_label=target_label,
        constraint=constraint,
        tol=tol,
        clip_min=clip_min,
        clip_max=clip_max
    )
    perturbed_np = perturbed_tf.numpy()[0]
    dist = np.linalg.norm((perturbed_np - sample_np).reshape(-1))
    return perturbed_np, float(dist)


# -------------------------------------------------------------
# HSJA主函数
# -------------------------------------------------------------
def hsja(model, sample,
         clip_max=1.0, clip_min=0.0,
         constraint='l2',
         num_iterations=50,
         gamma=1.0,
         target_label=None,
         target_image=None,
         stepsize_search='geometric_progression',
         max_num_evals=1e4,
         init_num_evals=100,
         eval_batch_size=256,
         verbose=True,
         original_label=None):
    
    sample_np = np.array(sample, dtype=np.float32)
    if original_label is None:
        preds = model.predict(np.expand_dims(sample_np, 0))
        original_label = int(np.argmax(preds, axis=1)[0])
    if target_label is None:
        target_label = -1

    perturbed = initialize(sample_np, model, original_label, target_image, target_label, clip_min, clip_max)
    perturbed, dist = binary_search_batch(sample_np, perturbed, model, original_label, target_label,
                                          constraint, tol=1e-5, clip_min=clip_min, clip_max=clip_max)

    print(f"[hsja] init dist={dist:.6e}")

    for j in range(int(num_iterations)):

        print(f"[hsja] iter {j+1}/{num_iterations}, dist={dist:.6e}")

        delta = 0.1 * (clip_max - clip_min) if j == 0 else np.sqrt(np.prod(sample_np.shape)) * gamma * dist
        num_evals = int(min(init_num_evals * np.sqrt(j + 1), max_num_evals))

        grad_tf = approximate_gradient_tf(model, perturbed, num_evals, delta,
                                          original_label, target_label,
                                          clip_min, clip_max,
                                          constraint, eval_batch_size)
        grad = grad_tf.numpy()[0]
        update = np.sign(grad) if constraint == 'linf' else grad

        epsilon = dist / np.sqrt(j + 1 + 1e-12)
        candidate = np.clip(perturbed + epsilon * update, clip_min, clip_max)
        if not decision_function_tf(model, candidate[None, ...], original_label, target_label, clip_min, clip_max).numpy()[0]:
            for _ in range(8):
                epsilon /= 2
                candidate = np.clip(perturbed + epsilon * update, clip_min, clip_max)
                if decision_function_tf(model, candidate[None, ...], original_label, target_label, clip_min, clip_max).numpy()[0]:
                    break

        perturbed, dist = binary_search_batch(sample_np, candidate, model, original_label, target_label,
                                              constraint, tol=1e-5, clip_min=clip_min, clip_max=clip_max)


    return dist, perturbed.astype(np.float32)