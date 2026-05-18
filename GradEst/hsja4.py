# GradEst/hsja.py
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------
# Decision function (TF / GPU friendly)
# -------------------------------------------------------------
@tf.function
def decision_function_tf(model, images, original_label, target_label=-1, clip_min=0.0, clip_max=1.0):
    """
    Batched decision function returning boolean mask (N,) on desired side.
    images: tf.Tensor (N,H,W,C) or (H,W,C)
    original_label: scalar int or tf.Tensor scalar
    target_label: scalar int (<0 means untargeted)
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

    mask = tf.cond(tf.less(tgt_tf, tf.constant(0, dtype=tf.int32)),
                   untargeted_case,
                   targeted_case)
    return tf.cast(mask, tf.bool)  # shape (N,)


# -------------------------------------------------------------
# Distance helper (numpy)
# -------------------------------------------------------------
def compute_distance(x_ori, x_pert, constraint='l2'):
    if constraint == 'l2':
        return np.linalg.norm((x_ori - x_pert).reshape(-1))
    else:
        return np.max(np.abs(x_ori - x_pert))


# -------------------------------------------------------------
# Initialization (random noise until misclassified + local binary search)
# -------------------------------------------------------------
def initialize(sample_np, model, original_label, target_image=None, target_label=-1,
               clip_min=0.0, clip_max=1.0, max_tries=10000, batch_init=32):
    """
    Try batch_init random noises per try; when find misclassified noise, binary-search
    between original sample and that noise to get close to boundary.
    """
    if target_image is not None:
        return target_image.astype(np.float32)

    sample_np = sample_np.astype(np.float32)
    tries = 0
    while tries < max_tries:
        noises = np.random.uniform(clip_min, clip_max, size=(batch_init,) + sample_np.shape).astype(np.float32)
        mask = decision_function_tf(model, noises, original_label, target_label, clip_min, clip_max)
        mask_np = mask.numpy()
        idx = np.where(mask_np)[0]
        if idx.size > 0:
            rn = noises[idx[0]]
            low, high = 0.0, 1.0
            # binary refine to boundary (use python loop, calls TF decision)
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
# Approximate gradient (TF internal computations, chunked to limit memory)
# -------------------------------------------------------------
import tensorflow as tf

@tf.function
def approximate_gradient_tf(model, sample, num_evals, delta,
                            original_label, target_label=-1,
                            clip_min=0.0, clip_max=1.0,
                            constraint='l2', eval_batch_size=256):
    """
    Fully TF-only gradient estimation for HopSkipJump attack.
    With baseline subtraction (variance reduction).
    ---------------------------------------------------------
    Args:
        model: TF/Keras 模型
        sample: (1,H,W,C) float32 原始输入
        num_evals: 采样数量
        delta: float32 小扰动尺度
        original_label: int
        target_label: int 或 -1 (非定向攻击)
        clip_min, clip_max: 输入范围
        constraint: 'l2' 或 'linf'
        eval_batch_size: 每批 Monte Carlo 样本
    Returns:
        grad_est: (1,H,W,C) TF tensor — 单位化的近似梯度方向
    """
    # 保证输入为 Tensor
    sample = tf.convert_to_tensor(sample, dtype=tf.float32)
    if sample.shape.rank == 3:
        sample = tf.expand_dims(sample, 0)  # (1,H,W,C)

    H, W, C = sample.shape[1], sample.shape[2], sample.shape[3]
    total = tf.cast(num_evals, tf.int32)
    delta = tf.cast(delta, tf.float32)

    # 累积梯度
    grad_acc = tf.zeros((H, W, C), dtype=tf.float32)

    # 遍历所有采样批次
    processed = tf.constant(0)
    while processed < total:
        bs = tf.minimum(eval_batch_size, total - processed)

        # --- 采样随机方向 ---
        if constraint == 'l2':
            rv = tf.random.normal((bs, H, W, C), dtype=tf.float32)
            rv = rv / (tf.sqrt(tf.reduce_sum(tf.square(rv), axis=(1,2,3), keepdims=True)) + 1e-12)
        else:
            rv = tf.random.uniform((bs, H, W, C), -1.0, 1.0, dtype=tf.float32)
            rv = rv / (tf.reduce_max(tf.abs(rv), axis=(1,2,3), keepdims=True) + 1e-12)

        # --- 添加扰动并裁剪 ---
        sample_bs = tf.repeat(sample, repeats=bs, axis=0)
        perturbed = tf.clip_by_value(sample_bs + delta * rv, clip_min, clip_max)

        # --- 决策函数 ---
        mask = decision_function_tf(model, perturbed,
                                    original_label, target_label,
                                    clip_min, clip_max)  # (bs,)

        mask = tf.cast(mask, tf.float32)  # {0,1}
        fvals = 2.0 * mask - 1.0  # {−1,+1}
        fvals -= tf.reduce_mean(fvals)  # baseline subtraction
        fvals = tf.reshape(fvals, (bs, 1, 1, 1))

        # --- 梯度累积 ---
        grad_batch = tf.reduce_mean(fvals * rv, axis=0)
        grad_acc += grad_batch * tf.cast(bs, tf.float32) / tf.cast(total, tf.float32)

        processed += bs

    # --- 归一化 ---
    norm = tf.norm(tf.reshape(grad_acc, (-1,)))
    grad_acc = tf.cond(norm > 0, lambda: grad_acc / (norm + 1e-12), lambda: grad_acc)

    return tf.expand_dims(grad_acc, 0)  # shape=(1,H,W,C)


# -------------------------------------------------------------
# Binary search parallel TF implementation (returns tensors)
# -------------------------------------------------------------
@tf.function
def binary_search_batch_parallel_tf(sample_np, perturbed_candidates_np, model,
                                    original_label, target_label=-1,
                                    constraint='l2', tol=1e-5,
                                    clip_min=0.0, clip_max=1.0):
    """
    sample_np: (H,W,C) numpy or tensor-compatible
    perturbed_candidates_np: tf.Tensor or numpy array (N,H,W,C)
    Returns:
      perturbed_on_boundary_tf: tf.Tensor (N,H,W,C)
      highs: tf.Tensor (N,) final high coefficients
    Implementation fully in TF while_loop to avoid repeated python<->gpu syncs.
    """
    sample_tf = tf.convert_to_tensor(sample_np[None, ...], dtype=tf.float32)  # (1,H,W,C)
    candidates_tf = tf.convert_to_tensor(perturbed_candidates_np, dtype=tf.float32)  # (N,H,W,C)
    N = tf.shape(candidates_tf)[0]

    low = tf.zeros((N,), dtype=tf.float32)
    high = tf.ones((N,), dtype=tf.float32)

    # while loop condition: any(high - low > tol)
    def cond(low, high):
        return tf.reduce_any(high - low > tol)

    # body: compute mids, query model in batch, update lows/highs
    def body(low, high):
        mid = (high + low) / 2.0  # (N,)
        mid_exp = tf.reshape(mid, (-1, 1, 1, 1))  # (N,1,1,1)
        blended = (1.0 - mid_exp) * sample_tf + mid_exp * candidates_tf  # (N,H,W,C)
        blended = tf.clip_by_value(blended, clip_min, clip_max)
        mask = decision_function_tf(model, blended, original_label, target_label, clip_min, clip_max)  # (N,)
        lows_new = tf.where(mask, low, mid)   # if mask True (on target side) keep low, else low=mid
        highs_new = tf.where(mask, mid, high) # if mask True -> high=mid else keep high
        return lows_new, highs_new

    # shape_invariants allow N to be dynamic
    low_final, high_final = tf.while_loop(cond, body, [low, high],
                                          parallel_iterations=1,
                                          shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])])

    high_exp = tf.reshape(high_final, (-1, 1, 1, 1))
    perturbed_on_boundary = (1.0 - high_exp) * sample_tf + high_exp * candidates_tf
    perturbed_on_boundary = tf.clip_by_value(perturbed_on_boundary, clip_min, clip_max)
    return perturbed_on_boundary, high_final  # both tensors


# -------------------------------------------------------------
# Wrapper for single-candidate binary search (keeps API)
# -------------------------------------------------------------
def binary_search_batch(sample_np, perturbed_candidate_np, model,
                        original_label, target_label=-1,
                        constraint='l2', tol=1e-5,
                        clip_min=0.0, clip_max=1.0):
    cand_np = np.expand_dims(np.array(perturbed_candidate_np, dtype=np.float32), axis=0)  # (1,H,W,C)
    perturbed_tf, highs = binary_search_batch_parallel_tf(sample_np, cand_np, model,
                                                          original_label, target_label,
                                                          constraint, tol, clip_min, clip_max)
    perturbed_np = perturbed_tf.numpy()[0]
    if constraint == 'l2':
        dist = np.linalg.norm((perturbed_np - sample_np).reshape(-1))
    else:
        dist = np.max(np.abs(perturbed_np - sample_np))
    return perturbed_np.astype(np.float32), float(dist)


# -------------------------------------------------------------
# geometric progression search in TF (find epsilon via halving until success)
# -------------------------------------------------------------
@tf.function
def geometric_progression_for_stepsize_tf(x_tf, update_tf, dist_tf, cur_iter_tf,
                                          model, original_label, target_label=-1,
                                          clip_min=0.0, clip_max=1.0, max_trials=50):
    """
    x_tf: (1,H,W,C)
    update_tf: (1,H,W,C)
    dist_tf: scalar float32
    cur_iter_tf: scalar int32
    returns: epsilon scalar tensor
    """
    # initial epsilon = dist / sqrt(cur_iter)
    eps = dist_tf / tf.sqrt(tf.cast(cur_iter_tf, tf.float32) + 1e-12)
    i0 = tf.constant(0, dtype=tf.int32)

    def cond(eps, i):
        candidate = tf.clip_by_value(x_tf + eps * update_tf, clip_min, clip_max)
        mask = decision_function_tf(model, candidate, original_label, target_label, clip_min, clip_max)
        success = tf.reduce_any(mask)
        return tf.logical_and(tf.less(i, max_trials), tf.logical_not(success))

    def body(eps, i):
        return eps / 2.0, i + 1

    eps_final, _ = tf.while_loop(cond, body, [eps, i0],
                                 shape_invariants=[tf.TensorShape([]), tf.TensorShape([])],
                                 parallel_iterations=1)
    return eps_final


# -------------------------------------------------------------
# HSJA main function (API kept)
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
    """
    Run HSJA attack. Returns (dist (float), perturbed (np.array H,W,C))
    """
    # normalize inputs
    sample_np = np.array(sample, dtype=np.float32)
    if original_label is None:
        preds = model.predict(np.expand_dims(sample_np, 0))
        original_label = int(np.argmax(preds, axis=1)[0])
    if target_label is None:
        target_label = -1
    else:
        target_label = int(target_label)

    # initialization
    perturbed = initialize(sample_np, model, original_label, target_image, target_label, clip_min, clip_max)
    perturbed, dist = binary_search_batch(sample_np, perturbed, model, original_label, target_label,
                                          constraint, tol=1e-5, clip_min=clip_min, clip_max=clip_max)

    print(f"[hsja] init dist={dist:.6e}")

    # main loop
    for j in range(int(num_iterations)):
        print(f"[hsja] iter {j+1}/{num_iterations}, dist={dist:.6e}")

        # delta schedule (original HSJA)
        if j == 0:
            delta = 0.1 * (clip_max - clip_min)
        else:
            d = int(np.prod(sample_np.shape))
            delta = np.sqrt(d) * gamma * dist

        # num evals schedule
        num_evals = int(init_num_evals * np.sqrt(j + 1))
        num_evals = min(num_evals, int(max_num_evals))

        # approximate gradient (TF chunked)
        grad_tf = approximate_gradient_tf(model, perturbed, num_evals, delta,
                                          original_label, target_label,
                                          clip_min, clip_max, constraint,
                                          eval_batch_size)
        # update direction
        if constraint == 'linf':
            update_tf = tf.sign(grad_tf)
        else:
            update_tf = grad_tf

        # make x and update tensors with stable shapes
        x_tf = tf.convert_to_tensor(perturbed[None, ...], dtype=tf.float32)        # (1,H,W,C)
        # ensure update_tf is same shape (1,H,W,C)
        if tf.rank(update_tf) == 3:
            update_tf = tf.expand_dims(update_tf, 0)

        # geometric progression search in TF (keeps behavior of original HSJA)
        if stepsize_search == 'geometric_progression':
            dist_tf = tf.convert_to_tensor(float(dist), dtype=tf.float32)
            cur_iter_tf = tf.convert_to_tensor(int(j + 1), dtype=tf.int32)
            eps_tf = geometric_progression_for_stepsize_tf(x_tf, update_tf, dist_tf, cur_iter_tf,
                                                            model, original_label, target_label,
                                                            clip_min, clip_max, max_trials=50)
            epsilon = float(eps_tf.numpy())
            # compute candidate on CPU (small overhead) then project to boundary via TF batch BS
            # Here we convert update_tf to numpy once:
            update_np = update_tf.numpy()[0]
            candidate = np.clip(perturbed + epsilon * update_np, clip_min, clip_max)
            perturbed, dist = binary_search_batch(sample_np, candidate, model, original_label, target_label,
                                                  constraint, tol=1e-5, clip_min=clip_min, clip_max=clip_max)
        else:
            # safety fallback - should not happen since user requested geometric only

            print("[hsja] Unsupported stepsize_search; skipping update")
            continue

    return float(dist), perturbed.astype(np.float32)