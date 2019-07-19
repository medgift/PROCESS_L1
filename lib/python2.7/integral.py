import cv2
import numpy as np
from PIL import Image
from skimage.transform.integral import integral_image, integrate
from random import randint
"""
TO-DO:
* use proper point(x, y) class [over np.arrays?]
"""


def is_white_patch(cur_patch,white_percentage):
    ''' Basic is_white check: checks if the extracted patch is white
        and returns True if so

    input:
    cur_patch, patch to check
    white_percentage, white portion threshold

    output:
    True if percentage of white> white portion threshold
    False otherwise
    '''
    #good buttt slowww
    # half black and half white patches are still kept. not a good thing.
    is_white = True
    total_white = float(cur_patch.shape[0] *cur_patch.shape[1] * cur_patch.shape[2] * 255)
    if (cur_patch.sum()/total_white)>white_percentage:
        return is_white
    else:
        return not is_white

def is_point_on_border(
        point_x, point_y,
        margin_x=0, margin_y=0
):
    """Tell if a point is on the border of a rectangular area.

    Arguments
    +++++++++

    :param int point_x: point's x coordinate

    :param int point_y: point's y coordinate

    Keyword arguments
    +++++++++++++++++

    :param int margin_x: margin on x coordinate

    :param int margin_y: margin on y coordinate


    return: bool
    """
    return (point_x < margin_x) or (point_y < margin_y)


def is_point_within_boundaries(
        point_x, point_y, offset, bound
):
    """Tell if a point is within the boundaries of a rectangular area.

    Arguments
    +++++++++

    :param int point_x: point x coordinate

    :param int point_y: point x coordinate

    :param int offset: added to `point_x` and `point_y`

    :param int bound: boundaries as Numpy-style 2D array shape

    return: bool
    """
    # discard coordinates out of mask's boundary (it
    # shouldn't be necessary in sliding window mode, TBD...). Artifact
    # of random coordinates (!?)
    # [BUG] looks overkill
    patch_bound = np.resize(
        np.array(
            [point_x + offset, point_y + offset]
        ), (2,)
    )
    return (patch_bound[0] <= bound[0]) and (patch_bound[1] <= bound[1])


def get_random_index(min, max):
    """Return a generator (that never stops) over a uniform
    distribution of random indexes in [`min`, `max`].

    """
    while True:
        yield randint(min, max)


def get_linear_index(min, max):
    """Return a generator over a linear index range in [`min`, `max`].

    """
    i = min
    while i < max:
        yield i
        i += 1


def patch_sampling(slide, mask, **opts):
    """Patch sampling on whole slide image by random points over an uniform
    distribution.

    Arguments
    +++++++++

    slide = OpenSlide Object
    mask = mask image ( 0-1 int type nd-array)

    Keyword arguments
    +++++++++++++++++

    :param obj logger: a pymod:logging instance

    slide_level = level of mask

    patch_size = size of patch scala integer n
    n_samples = the number of output patches

    ...plaese complete me!

    return: list of patches (RGB images), list of patch point (starting from left top)
    """
    # def values updated from **opts
    dopts = {
        'area_overlap' : .6,
        'bad_batch_size' : 500,
        'gray_threshold' : 90,
        'logger' : None,
        'margin_width_x' : 250, # as per original code, watch out!
        'margin_width_y' : 50,  # ditto
        'method' : 'random',
        'n_samples' : 100,
        'patch_size' : 224,
        'slide_level' : 5,
        'white_level' : 200,
        'white_threshold' : .3,
        'white_threshold_incr' : .05,
        'white_threshold_max' : .7,
    }
    for dk in dopts:
        # [BUG] if called with a missing key, won't get the deault!?
        try:
            dopts[dk] = opts.pop(dk, None)
        except KeyError as k:
            pass
        # reinject as standard var... This is just because I'm lazy and want
        # to keep the original names ;-)
        exec "{} = dopts[dk]".format(dk)

    if opts:
        # leftovers...
        raise RuntimeError('unexpected options {}'.format(opts))

    # bind to the requested aux functions
    get_index = globals()['get_{}_index'.format(method)]
    logger.debug("Using sampling: {} => {}".format(method, get_index))

    if not callable(get_index):
        logger.error('{}: invalid index sampling method'.format(method))
        return None, None

    logger.debug("kw opts:\n{}.".format(dopts))

    report = {
        'out_of_boundary' : 0,
        'on_border' : 0,
        'black_patches' : 0,
        'white_patches' : 0,
        'gray_patches' : 0,
    }
    patch_list = []
    patch_point = []
    # patch size at given level or resolution
    level_patch_size = int(patch_size / slide.level_downsamples[slide_level])
    # lists of nonzero points in the mask
    x_l, y_l = mask.nonzero()
    x_ln = len(x_l)

    logger.debug('Mask has {} x, {} y nonzero points'.format(x_ln, len(y_l)))

    if x_ln < level_patch_size * 2:
        logger.error("Not enough nonzero mask points for at least 2 patches. Bailing out.")
        return None, None

    # computing the actual level of resolution (dot product)
    x_ws = (np.round(x_l * slide.level_downsamples[slide_level])).astype(int)
    y_ws = (np.round(y_l * slide.level_downsamples[slide_level])).astype(int)
    cnt = 0         # good patch counter
    nt_cnt = 0      # not taken patch counter
    pidx_itr = get_index(0, x_ln - 1)
    while(cnt < n_samples):
        # pick an index...
        p_idx = pidx_itr.next()
        # ...correspondig point in the mask
        level_point_x, level_point_y = x_l[p_idx], y_l[p_idx]
        # [BUG] otsu threshold takes also border, so discard?? mmh, needs
        # double check (risk missing stuff...)
        if is_point_on_border(level_point_x, level_point_y, margin_width_x, margin_width_y):
            logger.debug(
                'Skipping point on mask border: {}x ?< {}, {}y ?< {}'.format(
                    level_point_x, margin_width_x, level_point_y, margin_width_y
                )
            )
            report['on_border'] += 1
            continue

        if not is_point_within_boundaries(
                level_point_x, level_point_y, level_patch_size, mask.shape
        ):
            logger.debug(
                'Skipping point out of mask boundary: {} ?> {}, {} ?> {}'.format(
                    level_point_x, mask.shape[0], level_point_y, mask.shape[1]
                )
            )
            report['out_of_boundary'] += 1
            continue

        # make patch from mask image
        level_patch_mask = mask[
            int(level_point_x) : int(level_point_x + level_patch_size),
            int(level_point_y) : int(level_point_y + level_patch_size)
        ]

        # apply integral
        ii_map = integral_image(level_patch_mask)
        ii_sum = integrate(ii_map, (0, 0), (level_patch_size - 1, level_patch_size - 1))

        # total patch area should covers at least x% of the annotation
        # region
        overlap = float(ii_sum) / (level_patch_size**2)
        if overlap < area_overlap:
            continue

        # square patch (RGB point array in [0, 255])
        patch = slide.read_region(
            (y_ws[p_idx], x_ws[p_idx]), 0, (patch_size, patch_size)
        )
        patch = np.array(patch)

        if np.sum(patch) == 0:
            report['black_patches'] += 1
            logger.debug('Skipping black patch at {}, {}'.format(level_point_x, level_point_y))
            continue

        # check almost white RGB values.
        white_mask = patch[:,:,0:3] > white_level
        # sum over the 3 RGB channels
        if float(np.sum(white_mask)) / (patch_size**2*3) <= white_threshold:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2BGR)
            if np.mean(patch) > gray_threshold:
                # got a good one...
                patch_list.append(patch)
                # ...with its location
                patch_point.append((x_l[p_idx], y_l[p_idx]))
                cnt += 1
            else:
                report['gray_patches'] += 1
                logger.debug('Skipping grey patch at {}, {}'.format(x_l[p_idx], y_l[p_idx]))
        else:
            # bad one: too white
            report['white_patches'] += 1
            nt_cnt += 1

        # too many bad ones in this batch, tweak it. This is somewhat overdoing
        if nt_cnt >= bad_batch_size:
            if white_threshold >= white_threshold_max:
                logger.warning('Max white threshold reached! Bailing out')
                break

            white_threshold += white_threshold_incr
            nt_cnt = 0
            logger.debug(
                'white_threshold += {}, now at {}'.format(
                    white_threshold_incr, white_threshold
                )
            )
    # {end while}

    logger.info(
        'Skipped points: {} on mask boder, {} out of mask boundary'.format(
            report['on_border'], report['out_of_boundary']
        )
    )
    logger.info(
        'Skipped patches: {} black, {} white, {} gray'.format(
            report['black_patches'], report['white_patches'], report['gray_patches']
        )
    )

    return patch_list, patch_point
