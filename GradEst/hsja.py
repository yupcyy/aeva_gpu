# GradEst/hsja.py
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------
# 1. TF decision function
# -------------------------------------------------------------
@tf.function
def decision_function_tf(model, images, original_label, target_label=-1,
                         clip_min=0.0, clip_max=1.0):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    if tf.rank(images) == 3:
        images = tf.expand_dims(images, 0)
    images = tf.clip_by_value(images, clip_min, clip_max)
    logits = model(images, training=False)
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)

    orig_tf = tf.cast(original_label, tf.int32)
    tgt_tf = tf.cast(target_label, tf.int32)

    def untargeted():
        return tf.not_equal(preds, orig_tf)

    def targeted():
        return tf.equal(preds, tgt_tf)

    mask = tf.cond(tgt_tf < 0, untargeted, targeted)
    return tf.cast(mask, tf.bool)


# -------------------------------------------------------------
# 2. distance helper
# -------------------------------------------------------------
def compute_distance(x_ori, x_pert, constraint='l2'):
    if constraint == 'l2':
        return np.linalg.norm((x_ori - x_pert).reshape(-1))
    else:
        return np.max(np.abs(x_ori - x_pert))


# -------------------------------------------------------------
# 3. initialization (same logic as original)
# -------------------------------------------------------------
def initialize(sample_np, model, original_label, target_image=None,
               target_label=-1, clip_min=0.0, clip_max=1.0,
               max_tries=10000, batch_init=32):
    if target_image is not None:
        return target_image.astype(np.float32)

    sample_np = sample_np.astype(np.float32)
    tries = 0
    while tries < max_tries:
        noises = np.random.uniform(
            clip_min, clip_max,
            size=(batch_init,) + sample_np.shape
        ).astype(np.float32)
        mask = decision_function_tf(
            model, noises,
            original_label, target_label,
            clip_min, clip_max
        )
        mask_np = mask.numpy()
        idx = np.where(mask_np)[0]
        if idx.size > 0:
            rn = noises[idx[0]]
            # binary refine toward boundary
            low, high = 0.0, 1.0
            while high - low > 1e-3:
                mid = (low + high) / 2.0
                blended = (1 - mid) * sample_np + mid * rn
                m = decision_function_tf(
                    model,
                    blended[None, ...],
                    original_label, target_label,
                    clip_min, clip_max
                ).numpy()[0]
                if m:
                    high = mid
                else:
                    low = mid
            return ((1 - high) * sample_np + high * rn).astype(np.float32)
        tries += batch_init

    raise RuntimeError("initialize() failed — try larger max_tries or give target image.")


# -------------------------------------------------------------
# 4. TF-only approximate gradient (with baseline subtraction)
#    固定输入类型，减少 retracing
# -------------------------------------------------------------
@tf.function(reduce_retracing=True)
def approximate_gradient_tf(model,
                            sample,           # (1,H,W,C)
                            num_evals,        # scalar int32 tensor
                            delta,            # scalar float32 tensor
                            original_label,   # scalar int32 tensor
                            target_label,     # scalar int32 tensor
                            clip_min, clip_max,
                            constraint='l2',
                            eval_batch_size=256):
    # sample shape
    if tf.rank(sample) == 3:
        sample = tf.expand_dims(sample, 0)

    H = tf.shape(sample)[1]
    W = tf.shape(sample)[2]
    C = tf.shape(sample)[3]

    total = tf.cast(num_evals, tf.int32)
    grad_acc = tf.zeros((H, W, C), dtype=tf.float32)

    processed = tf.constant(0, dtype=tf.int32)
    while processed < total:
        bs = tf.minimum(tf.constant(eval_batch_size, tf.int32), total - processed)

        if constraint == 'l2':
            rv = tf.random.normal((bs, H, W, C), dtype=tf.float32)
            rv = rv / (tf.sqrt(tf.reduce_sum(tf.square(rv), axis=(1, 2, 3), keepdims=True)) + 1e-12)
        else:
            rv = tf.random.uniform((bs, H, W, C), -1.0, 1.0, dtype=tf.float32)
            rv = rv / (tf.reduce_max(tf.abs(rv), axis=(1, 2, 3), keepdims=True) + 1e-12)

        sample_bs = tf.repeat(sample, repeats=bs, axis=0)
        perturbed = tf.clip_by_value(sample_bs + delta * rv, clip_min, clip_max)

        mask = decision_function_tf(
            model, perturbed,
            original_label, target_label,
            clip_min, clip_max
        )
        fvals = tf.cast(mask, tf.float32) * 2.0 - 1.0
        fvals = fvals - tf.reduce_mean(fvals)     # baseline
        fvals = tf.reshape(fvals, (bs, 1, 1, 1))

        grad_batch = tf.reduce_mean(fvals * rv, axis=0)
        grad_acc = grad_acc + grad_batch * tf.cast(bs, tf.float32) / tf.cast(total, tf.float32)

        processed += bs

    flat = tf.reshape(grad_acc, (-1,))
    norm = tf.norm(flat)
    grad_acc = tf.cond(norm > 0,
                       lambda: grad_acc / (norm + 1e-12),
                       lambda: grad_acc)

    return tf.expand_dims(grad_acc, 0)   # (1,H,W,C)


# -------------------------------------------------------------
# 5. TF binary search (use theta like original)
# -------------------------------------------------------------
@tf.function
def binary_search_batch_parallel_tf(sample_np, perturbed_candidates_np, model,
                                    original_label, target_label,
                                    theta,
                                    constraint='l2',
                                    clip_min=0.0, clip_max=1.0):
    sample_tf = tf.convert_to_tensor(sample_np[None, ...], dtype=tf.float32)
    cands_tf = tf.convert_to_tensor(perturbed_candidates_np, dtype=tf.float32)
    N = tf.shape(cands_tf)[0]

    low = tf.zeros((N,), dtype=tf.float32)
    high = tf.ones((N,), dtype=tf.float32)

    def cond(l, h):
        return tf.reduce_any((h - l) / theta > 1.0)

    def body(l, h):
        mid = (h + l) / 2.0
        mid_exp = tf.reshape(mid, (-1, 1, 1, 1))
        blended = (1.0 - mid_exp) * sample_tf + mid_exp * cands_tf
        blended = tf.clip_by_value(blended, clip_min, clip_max)
        mask = decision_function_tf(
            model, blended,
            original_label, target_label,
            clip_min, clip_max
        )
        h_new = tf.where(mask, mid, h)
        l_new = tf.where(mask, l, mid)
        return l_new, h_new

    low_f, high_f = tf.while_loop(
        cond, body, [low, high],
        parallel_iterations=1,
        shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])]
    )

    high_exp = tf.reshape(high_f, (-1, 1, 1, 1))
    proj = (1.0 - high_exp) * sample_tf + high_exp * cands_tf
    proj = tf.clip_by_value(proj, clip_min, clip_max)
    return proj, high_f


def binary_search_batch(sample_np, perturbed_candidate_np, model,
                        original_label, target_label, theta,
                        constraint='l2',
                        clip_min=0.0, clip_max=1.0):
    cand = np.expand_dims(perturbed_candidate_np.astype(np.float32), 0)
    proj_tf, _ = binary_search_batch_parallel_tf(
        sample_np, cand, model,
        tf.convert_to_tensor(original_label, tf.int32),
        tf.convert_to_tensor(target_label, tf.int32),
        tf.convert_to_tensor(theta, tf.float32),
        constraint, clip_min, clip_max
    )
    proj_np = proj_tf.numpy()[0]
    if constraint == 'l2':
        dist = np.linalg.norm((proj_np - sample_np).reshape(-1))
    else:
        dist = np.max(np.abs(proj_np - sample_np))
    return proj_np.astype(np.float32), float(dist)


# -------------------------------------------------------------
# 6. geometric progression
# -------------------------------------------------------------
@tf.function
def geometric_progression_for_stepsize_tf(x_tf, update_tf, dist_tf, cur_iter_tf,
                                          model, original_label, target_label,
                                          clip_min=0.0, clip_max=1.0,
                                          max_trials=50):
    eps = dist_tf / tf.sqrt(tf.cast(cur_iter_tf, tf.float32) + 1e-12)
    i0 = tf.constant(0, dtype=tf.int32)

    def cond(eps, i):
        cand = tf.clip_by_value(x_tf + eps * update_tf, clip_min, clip_max)
        mask = decision_function_tf(
            model, cand,
            original_label, target_label,
            clip_min, clip_max
        )
        ok = tf.reduce_any(mask)
        return tf.logical_and(i < max_trials, tf.logical_not(ok))

    def body(eps, i):
        return eps / 2.0, i + 1

    eps_f, _ = tf.while_loop(
        cond, body, [eps, i0],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([])]
    )
    return eps_f


# -------------------------------------------------------------
# 7. HSJA main
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
        o = model.predict(np.expand_dims(sample_np, 0))
        original_label = int(np.argmax(o, axis=1)[0])

    if target_label is None:
        target_label = -1
    else:
        target_label = int(target_label)

    d = int(np.prod(sample_np.shape))
    if constraint == 'l2':
        theta = gamma / (np.sqrt(d) * d)
    else:
        theta = gamma / (d ** 2)

    # init
    perturbed = initialize(sample_np, model, original_label,
                           target_image, target_label,
                           clip_min, clip_max)
    perturbed, dist_post_update = binary_search_batch(
        sample_np, perturbed, model,
        original_label, target_label, theta,
        constraint, clip_min, clip_max
    )
    dist = compute_distance(sample_np, perturbed, constraint)

    if verbose:
        print(f"[hsja] init dist={dist:.6e}")

    for j in range(num_iterations):
        if verbose:
            print(f"[hsja] iter {j+1}/{num_iterations}, dist={dist:.6e}")

        # delta — original rule
        if j == 0:
            delta = 0.1 * (clip_max - clip_min)
        else:
            if constraint == 'l2':
                delta = np.sqrt(d) * theta * dist_post_update
            else:
                delta = d * theta * dist_post_update

        # num_evals schedule
        num_evals = int(init_num_evals * np.sqrt(j + 1))
        num_evals = int(min(num_evals, max_num_evals))

        # TF grad (all tensors, to reduce retracing)
        grad_tf = approximate_gradient_tf(
            model,
            tf.convert_to_tensor(perturbed[None, ...], tf.float32),
            tf.convert_to_tensor(num_evals, tf.int32),
            tf.convert_to_tensor(delta, tf.float32),
            tf.convert_to_tensor(original_label, tf.int32),
            tf.convert_to_tensor(target_label, tf.int32),
            tf.convert_to_tensor(clip_min, tf.float32),
            tf.convert_to_tensor(clip_max, tf.float32),
            constraint=constraint,
            eval_batch_size=eval_batch_size
        )

        if constraint == 'linf':
            update_tf = tf.sign(grad_tf)
        else:
            update_tf = grad_tf

        x_tf = tf.convert_to_tensor(perturbed[None, ...], tf.float32)

        if stepsize_search == 'geometric_progression':
            eps_tf = geometric_progression_for_stepsize_tf(
                x_tf, update_tf,
                tf.convert_to_tensor(dist, tf.float32),
                tf.convert_to_tensor(j + 1, tf.int32),
                model,
                tf.convert_to_tensor(original_label, tf.int32),
                tf.convert_to_tensor(target_label, tf.int32),
                clip_min, clip_max, 50
            )
            eps = float(eps_tf.numpy())
        else:
            eps = dist / np.sqrt(j + 1.0)

        update_np = update_tf.numpy()[0]
        candidate = np.clip(perturbed + eps * update_np, clip_min, clip_max)

        perturbed, dist_post_update = binary_search_batch(
            sample_np, candidate, model,
            original_label, target_label, theta,
            constraint, clip_min, clip_max
        )
        dist = compute_distance(sample_np, perturbed, constraint)

    return float(dist), perturbed.astype(np.float32)