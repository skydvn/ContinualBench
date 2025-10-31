import torch
import numpy as np
import random


def classwise_fair_selection(task, cand_target, sorted_index, num_per_label, n_classes, memory_size, is_shuffle=True):
    """
    Class-wise fair selection for coreset allocation.
    Adapted from the original BCSR implementation.
    """
    # Convert inputs to numpy arrays for consistent processing
    if isinstance(cand_target, torch.Tensor):
        cand_target = cand_target.cpu().numpy()
    if isinstance(sorted_index, torch.Tensor):
        sorted_index = sorted_index.cpu().numpy()
    
    num_examples_per_task = memory_size // task
    num_examples_per_class = num_examples_per_task // n_classes
    num_residuals = num_examples_per_task - num_examples_per_class * n_classes
    residuals = np.sum([(num_examples_per_class - n_c)*(num_examples_per_class > n_c) for n_c in num_per_label])
    num_residuals += residuals
    
    # Get the number of coreset instances per class
    while True:
        n_less_sample_class = np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])
        num_class = (n_classes - n_less_sample_class)
        if (num_residuals // num_class) > 0:
            num_examples_per_class += (num_residuals // num_class)
            num_residuals -= (num_residuals // num_class) * num_class
        else:
            break
    
    # Get best coresets per class
    selected = []
    target_tid = np.floor(max(cand_target)/n_classes)

    for j in range(n_classes):
        position = np.squeeze((cand_target[sorted_index]==j+(target_tid*n_classes)).nonzero())
        if position.size > 1:  # Use .size for numpy arrays
            selected.append(position[:num_examples_per_class])
        elif position.size == 0:  # Use .size for numpy arrays
            continue
        else:
            selected.append([position])
    
    # Fill rest space as best residuals
    if len(selected) > 0:
        selected = np.concatenate(selected)
        unselected = np.array(list(set(np.arange(num_examples_per_task))^set(selected)))
        final_num_residuals = num_examples_per_task - len(selected)
        best_residuals = unselected[:final_num_residuals]
        selected = np.concatenate([selected, best_residuals])
    else:
        # If no samples were selected, just use random selection
        selected = np.random.permutation(num_examples_per_task)

    if is_shuffle:
        np.random.shuffle(selected)

    return sorted_index[selected.astype(int)] 
