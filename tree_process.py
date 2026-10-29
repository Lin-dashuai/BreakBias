"""Higra hard reference used by the H03 input pipeline."""
import random
import numpy as np
from scipy.ndimage import rotate
from imgaug import augmenters as iaa
import higra as hg

def tree_process_3D_higra(img, tree_type, rotate_angle, hard_strength='base'):
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
