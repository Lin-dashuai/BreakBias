"""Role-aware view perturbation (RVP): shared reference, BE and AP views."""
import random
import numpy as np
from scipy.ndimage import rotate
from imgaug import augmenters as iaa
import higra as hg

def hard_filter_component_tree(img, tree_type, rotate_angle, hard_strength='base'):
    """H03 hard reference: original Higra/Gamma pipeline with count-one node removal."""
    assert hard_strength == 'base'
    img_h, img_w, img_d = img.shape
    img_reshape = img.reshape(img_h, img_w * img_d)
    if random.random() > 0.5:
        contrast = iaa.GammaContrast((0.5, 1.5))
        img_reshape = contrast.augment_image(img_reshape)
    img = img_reshape.reshape(img_h, img_w, img_d)
    img = rotate(img, axes=(1, 2), angle=rotate_angle, reshape=False, order=1)
    img = img * 255
    img = img.astype(np.uint8)
    mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    neighbours = hg.mask_2_neighbours(mask)
    graph = hg.get_nd_regular_implicit_graph(img.shape, neighbours)
    if tree_type == 'maxtree':
        tree, altitudes = hg.component_tree_max_tree(graph, img)
        weight = np.ones((tree.num_vertices(),))
        weight[:tree.num_leaves()] = 0
        nbChildren = hg.accumulate_parallel(tree, weight, hg.Accumulators.sum)
        nbSiblings = hg.propagate_parallel(tree, nbChildren)
        nbSiblings[tree.root()] = 0
        deleted_nodes = nbSiblings == 1
        filtered_higra = hg.reconstruct_leaf_data(tree, altitudes, deleted_nodes)
    elif tree_type == 'mintree':
        tree, altitudes = hg.component_tree_min_tree(graph, img)
        weight = np.ones((tree.num_vertices(),))
        weight[:tree.num_leaves()] = 0
        nbChildren = hg.accumulate_parallel(tree, weight, hg.Accumulators.sum)
        nbSiblings = hg.propagate_parallel(tree, nbChildren)
        nbSiblings[tree.root()] = 0
        deleted_nodes = nbSiblings == 1
        filtered_higra = hg.reconstruct_leaf_data(tree, altitudes, deleted_nodes)
    filtered_higra = filtered_higra.astype(np.float32)
    filtered_higra = filtered_higra / 255
    return filtered_higra

def _clip01(image):
    return np.clip(image, 0.0, 1.0).astype(np.float32)

def build_shared_hard_reference(image, tree_type, rotate_angle, hard_strength='base'):
    out = hard_filter_component_tree(image, tree_type, rotate_angle, hard_strength='base')
    if not isinstance(out, np.ndarray):
        return image
    return _clip01(out)

def tree_sensitive_mask(image, hard_view):
    diff = np.abs(hard_view - image)
    if float(diff.max()) < 1e-06:
        return np.zeros_like(image, dtype=bool)
    thr = max(0.01, float(np.percentile(diff, 85)))
    mask = diff > thr
    if mask.mean() < 0.002:
        mask = diff > max(0.003, float(np.percentile(diff, 70)))
    return mask

def anatomy_preserving_altitude_jitter(image, hard_view, jitter=5.0):
    mask = tree_sensitive_mask(image, hard_view)
    if not mask.any():
        return image
    delta = random.choice([-1.0, 1.0]) * random.uniform(0.5 * jitter / 255.0, jitter / 255.0)
    out = image.copy()
    out[mask] = out[mask] + delta
    return _clip01(out)
