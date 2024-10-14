# Region-based CDAM
# Combines a CDAM and a segmentation map, similar to XRAI (https://arxiv.org/abs/1906.02825)
#
# Modified from https://github.com/PAIR-code/saliency/blob/master/saliency/core/xrai.py (Apache License 2.0)

import logging
import numpy as np
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize

_logger = logging.getLogger(__name__)

_FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150


def _normalize_image(im, value_range, resize_shape=None):
    """Normalize an image by resizing it and rescaling its values.

    Args:
        im: Input image.
        value_range: [min_value, max_value]
        resize_shape: New image shape. Defaults to None.

    Returns:
        Resized and rescaled image.
    """
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im - im_min) / (im_max - im_min)
    im = im * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        im = resize(
            im,
            resize_shape,
            order=3,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )
    return im


def _get_segments_felzenszwalb(im, resize_image=True, scale_range=None, dilation_rad=3):
    """Compute image segments based on Felzenszwalb's algorithm.

    Efficient graph-based image segmentation, Felzenszwalb, P.F.
    and Huttenlocher, D.P. International Journal of Computer Vision, 2004

    Args:
      im: Input image.
      resize_image: If True, the image is resized to 224,224 for the segmentation
                    purposes. The resulting segments are rescaled back to match
                    the original image size. It is done for consistency w.r.t.
                    segmentation parameter range. Defaults to True.
      scale_range:  Range of image values to use for segmentation algorithm.
                    Segmentation algorithm is sensitive to the input image
                    values, therefore we need to be consistent with the range
                    for all images. If None is passed, the range is scaled to
                    [-1.0, 1.0]. Defaults to None.
      dilation_rad: Sets how much each segment is dilated to include edges,
                    larger values cause more blobby segments, smaller values
                    get sharper areas. Defaults to 5.
    Returns:
        masks: A list of boolean masks as np.ndarrays if size HxW for im size of
               HxWxC.
    """

    # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
    # parameters for that
    if scale_range is None:
        scale_range = _FELZENSZWALB_IM_VALUE_RANGE
    # Normalize image value range and size
    original_shape = im.shape[:2]
    # TODO (tolgab) This resize is unnecessary with more intelligent param range
    # selection
    if resize_image:
        im = _normalize_image(im, scale_range, _FELZENSZWALB_IM_RESIZE)
    else:
        im = _normalize_image(im, scale_range)
    segs = []
    for scale in _FELZENSZWALB_SCALE_VALUES:
        for sigma in _FELZENSZWALB_SIGMA_VALUES:
            seg = segmentation.felzenszwalb(
                im, scale=scale, sigma=sigma, min_size=_FELZENSZWALB_MIN_SEGMENT_SIZE
            )
            if resize_image:
                seg = resize(
                    seg,
                    original_shape,
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                ).astype(int)
            segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        footprint = disk(dilation_rad)
        masks = [dilation(mask, footprint=footprint) for mask in masks]
    return masks


def _attr_aggregation_max(attr, axis=-1):
    return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
    # Compute the attr density over mask1. If mask2 is specified, compute density
    # for mask1 \ mask2
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _get_diff_mask(add_mask, base_mask):
    return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
    return np.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs):
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks

# adapted from _xrai and _xrai_fast, where attr -> maps.
def regions(
    maps,
    segs,
    gain_fun=_gain_density,
    area_perc_th=1.0,
    min_pixel_diff=50,
    integer_segments=True,
    approximate=False
):
    """Compute explainable regions given attention maps and segments. Explainable regions are
        a set of importance scores (identical dimensions as an input `maps` except the color channel),
        by aggregating and comparing importance scores that fall in each segment. The segmentation map (`segs`)
        needs to be provided. Adapted from xrai and xrai_fast in the saliency package.

    Args:
        maps: attention maps (aka importance scores).
        segs: Input segments as a list of boolean masks. See _get_segments_felzenszwalb.    
        gain_fun: The function that computes region importances from attention maps.
            Defaults to _gain_density.
        area_perc_th: The region map is computed to cover area_perc_th of
            the image. Lower values will run faster, but produce
            uncomputed areas in the image that will be filled to
            satisfy completeness. Defaults to 1.0.
            Not used if apprxomiate is set to True.
        min_pixel_diff: Do not consider masks that have difference less than
            this number compared to the current mask. Set it to 1
            to remove masks that completely overlap with the
            current mask.
        integer_segments: if set to True (default), the segments are returned as an integer array with
            the same dimensions as the input (excluding color channels). The elements
            of the array are set to values from the [1,N] range, where 1 is the most
            important segment and N is the least important segment. If set to False, the segments are returned as a
            boolean array, where the first dimension has size N. The [0, ...] mask is
            the most important and the [N-1, ...] mask is the least important.
        approximate: if set to False (default), do not consider mask overlap during importance ranking,
            significantly speeding up the algorithm for less accurate results.
 
    Returns:
        tuple: region map and list of masks or an integer image with
            area ranks depending on the parameter integer_segments.
    """
    output_importance = -np.inf * np.ones(shape=maps.shape, dtype=float)

    n_masks = len(segs)
    current_mask = np.zeros(maps.shape, dtype=bool)

    masks_trace = []

    if approximate is False:
        remaining_masks = {ind: mask for ind, mask in enumerate(segs)}
        current_area_perc = 0.0

        added_masks_cnt = 1
        # While the mask area is less than area_th and remaining_masks is not empty
        while current_area_perc <= area_perc_th:
            best_gain = -np.inf
            best_key = None
            remove_key_queue = []
            for mask_key in remaining_masks:
                mask = remaining_masks[mask_key]
                # If mask does not add more than min_pixel_diff to current mask, remove
                mask_pixel_diff = _get_diff_cnt(mask, current_mask)
                if mask_pixel_diff < min_pixel_diff:
                    remove_key_queue.append(mask_key)
                    if _logger.isEnabledFor(logging.DEBUG):
                        _logger.debug(
                            "Skipping mask with pixel difference: {:.3g},".format(
                                mask_pixel_diff
                            )
                        )
                    continue
                gain = gain_fun(mask, maps, mask2=current_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_key = mask_key
            for key in remove_key_queue:
                del remaining_masks[key]
            if not remaining_masks:
                break
            added_mask = remaining_masks[best_key]
            mask_diff = _get_diff_mask(added_mask, current_mask)
            masks_trace.append((mask_diff, best_gain))

            current_mask = np.logical_or(current_mask, added_mask)
            current_area_perc = np.mean(current_mask)
            output_importance[mask_diff] = best_gain
            del remaining_masks[best_key]  # delete used key
            if _logger.isEnabledFor(logging.DEBUG):
                current_importance_sum = np.sum(maps[current_mask])
                _logger.debug(
                    "{} of {} masks added,"
                    "importance_sum: {}, area: {:.3g}/{:.3g}, {} remaining masks".format(
                        added_masks_cnt,
                        n_masks,
                        current_importance_sum,
                        current_area_perc,
                        area_perc_th,
                        len(remaining_masks),
                    )
                )
            added_masks_cnt += 1
    else:
        masks_trace = []

        # Sort all masks based on gain, ignore overlaps
        seg_importances = [gain_fun(seg_mask, maps) for seg_mask in segs]
        segs, seg_importances = list(zip(*sorted(zip(segs, seg_importances), key=lambda x: -x[1])))

        for i, added_mask in enumerate(segs):
            mask_diff = _get_diff_mask(added_mask, current_mask)
            # If mask does not add more than min_pixel_diff to current mask, skip
            mask_pixel_diff = _get_diff_cnt(added_mask, current_mask)
            if mask_pixel_diff < min_pixel_diff:
                if _logger.isEnabledFor(logging.DEBUG):
                    _logger.debug(
                        "Skipping mask with pixel difference: {:.3g},".format(
                            mask_pixel_diff
                        )
                    )
                continue
            mask_gain = gain_fun(mask_diff, maps)
            masks_trace.append((mask_diff, mask_gain))
            output_importance[mask_diff] = mask_gain
            current_mask = np.logical_or(current_mask, added_mask)
            if _logger.isEnabledFor(logging.DEBUG):
                current_importance_sum = np.sum(maps[current_mask])
                current_area_perc = np.mean(current_mask)
                _logger.debug(
                    "{} of {} masks processed,"
                    "importance_sum: {}, area: {:.3g}/{:.3g}".format(
                        i + 1,
                        n_masks,
                        current_importance_sum,
                        current_area_perc,
                        area_perc_th,
                    )
                )

    uncomputed_mask = output_importance == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_importance[uncomputed_mask] = gain_fun(uncomputed_mask, maps)
    masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
    if np.any(uncomputed_mask):
        masks_trace.append(uncomputed_mask)
    if integer_segments:
        importance_ranks = np.zeros(shape=maps.shape, dtype=int)
        for i, mask in enumerate(masks_trace):
            importance_ranks[mask] = i + 1
        return output_importance, importance_ranks
    else:
        return output_importance, masks_trace
