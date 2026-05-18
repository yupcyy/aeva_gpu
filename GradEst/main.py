# GradEst/main.py
import numpy as np
from GradEst.hsja import hsja

def attack(model, basic_imgs, target_imgs, target_labels,
           num_iterations=50, gamma=1.0, constraint='l2',
           init_num_evals=100, max_num_evals=30000,
           clip_min=0.0, clip_max=1.0, stepsize_search='geometric_progression',
           eval_batch_size=512, verbose=True):
    """
    Wrapper to attack a batch of samples. Preserves original outputs:
    returns vec (B,) and vec_per (B,H,W,C)
    """
    B = int(basic_imgs.shape[0])
    vec = np.empty((B,), dtype=np.float32)
    vec_per = np.zeros_like(basic_imgs, dtype=np.float32)

    for i in range(B):
        sample = basic_imgs[i].astype(np.float32)
        tgt_img = target_imgs[i].astype(np.float32) if target_imgs is not None else None
        tgt_label = int(target_labels[i])

        # compute original label
        pred = model.predict(np.expand_dims(sample,0))
        original_label = int(np.argmax(pred, axis=1)[0])

        print(f"attacking the {i}th sample... original_label: {original_label}, target_label: {tgt_label}")

        dist, pert = hsja(model, sample,
                         clip_max=clip_max, clip_min=clip_min, constraint=constraint,
                         num_iterations=num_iterations, gamma=gamma,
                         target_label=tgt_label, target_image=tgt_img,
                         stepsize_search=stepsize_search,
                         max_num_evals=max_num_evals, init_num_evals=init_num_evals,
                         eval_batch_size=eval_batch_size, verbose=verbose,
                         original_label=original_label)
        per = pert - sample
        vec[i] = np.max(np.abs(per)) / (dist + 1e-12)
        vec_per[i] = per

    return vec, vec_per