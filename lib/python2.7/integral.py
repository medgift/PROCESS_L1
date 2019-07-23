import cv2
import numpy as np
from PIL import Image
from skimage.transform.integral import integral_image, integrate
from random import randint
"""
TO-DO:
* use proper point(x, y) class [over np.arrays?]
* develop a `sampler` class -- too many kludges down there ^_^
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


def get_index__random(min, max):
    """Return a generator (that _never_ stops) over a uniform
    distribution of random indexes in [`min`, `max`].
    """
    while True:
        yield randint(min, max)


def get_index__linear(min, max):
    """Return a generator over a linear index range in [`min`, `max`].
    """
    i = min
    while i < max:
        yield i
        i += 1

def is_batch_over__random(cnt, stop):
    """Tell if random sampling is over given the counter `cnt` and its `stop`
    value.

    :return bool: cnt < stop
    """
    return cnt >= stop


def is_batch_over__linear(cnt, stop):
    """Dummy function as linear sampling ends when it's corresponding index
    generator `get_linear_index` stops. `cnt` and `stop` are ignored.  value.

    :return bool: always False
    """
    return False

def nonzero_range(mask, window=[]):
    """Return a nonzero 2D point array from `mask` in range
    [`window[0]`, `window[1]`).

    :return list, list: as np arrays
    """
    x_l, y_l = mask.nonzero()
    if window:
        x_ln, y_ln = len(x_l), len(y_l)
        x_l = x_l[int(window[0]/100.*x_ln):int(window[1]/100.*x_ln)]
        y_l = y_l[int(window[0]/100.*y_ln):int(window[1]/100.*y_ln)]

    return x_l, y_l


def get_white_threshold__random(
        nt_cnt, bad_batch_size, white_threshold, white_threshold_max, white_threshold_incr
):
    """Too many bad ones in this batch, tweak it"""
    # globals nt_cnt, bad_batch_size, white_threshold, white_threshold_max, white_threshold_incr, logger
    if nt_cnt >= bad_batch_size:
        if white_threshold >= white_threshold_max:
            return None, None

        white_threshold += white_threshold_incr
        nt_cnt = 0
        # logger.debug(
        #     'white_threshold += {}, now at {}'.format(
        #         white_threshold_incr, white_threshold
        #     )
        # )
        return nt_cnt, white_threshold
    return nt_cnt, white_threshold

def get_white_threshold__linear(
        nt_cnt, bad_batch_size, white_threshold, white_threshold_max, white_threshold_incr
):
    """No-op"""
    return nt_cnt, white_threshold


def patch_sampling(slide, mask, **opts):
    """Patch sampling on whole slide image by random points over an uniform
    distribution.

    Arguments
    +++++++++

    slide = OpenSlide Object
    mask = mask image ( 0-1 int type nd-array)


    Keyword arguments
    +++++++++++++++++

    :param int start_idx: start index on the mask's nonzero point
    list. Ignored if `mode` == 'random'

    :param obj logger: a pymod:logging instance

    :param int slide_level: level of mask

    :param int patch_size: size of patch scala integer n

    :param int n_samples: the number of patches to extract (batch size)

    ...plaese complete me!

    :return list:

        patches (RGB images),
        list of patch points (starting from left top),
        last used nonzero mask point's index or None (random sampling only)

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
        'start_idx' : 0,
        'white_level' : 200,
        'white_threshold' : .3,
        'white_threshold_incr' : .05,
        'white_threshold_max' : .7,
        'window' : []
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

    logger.debug("kw opts:\n{}.".format(dopts))

    # bind to aux functions
    bfn = {
        'get_index'             : None,
        'get_white_threshold'   : None,
        'is_batch_over'         : None,
    }
    for n in bfn.keys():
        bfn[n] = globals()['{}__{}'.format(n, method)]
        if not callable(bfn[n]):
            logger.fatal('[BUG] {} => {}: invalid aux function binding'.format(n, bfn[n]))
            return None, None

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
    if window:
        logger.info(
            "Restricting nonzero points range to {}%, {}%".format(window[0], window[1])
        )
    x_l, y_l = nonzero_range(mask, window)
    x_ln, y_ln = len(x_l), len(y_l)

    logger.info('Mask has {} x {} nonzero points'.format(x_ln, len(y_l)))

    if x_ln < level_patch_size * 2:
        logger.error("Not enough nonzero mask points for at least 2 patches. Bailing out.")
        return None, None

    # computing the actual level of resolution (dot product)
    x_ws = (np.round(x_l * slide.level_downsamples[slide_level])).astype(int)
    y_ws = (np.round(y_l * slide.level_downsamples[slide_level])).astype(int)
    cnt = 0         # good patch counter
    nt_cnt = 0      # not taken patch counter
    p_iterator = bfn['get_index'](start_idx, x_ln - 1)
    p_idx = start_idx # just for init purposes
    # while(not bfn['is_batch_over'](cnt, n_samples)):
    while(cnt < n_samples):
        # pick an index...
        try:
            p_idx = p_iterator.next()
        except StopIteration:
            break

        # ...corresponding point in the mask
        level_point_x, level_point_y = x_l[p_idx], y_l[p_idx]
        # [BUG] otsu threshold takes also border, so discard?? mmh, needs
        # double check (risk missing stuff...)
        if is_point_on_border(level_point_x, level_point_y, margin_width_x, margin_width_y):
            # logger.debug(
            #     'Skipping point on mask border: {}x ?< {}, {}y ?< {}'.format(
            #         level_point_x, margin_width_x, level_point_y, margin_width_y
            #     )
            # )
            report['on_border'] += 1
            continue

        if not is_point_within_boundaries(
                level_point_x, level_point_y, level_patch_size, mask.shape
        ):
            # logger.debug(
            #     'Skipping point out of mask boundary: {} ?> {}, {} ?> {}'.format(
            #         level_point_x, mask.shape[0], level_point_y, mask.shape[1]
            #     )
            # )
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
            # logger.debug('Skipping black patch at {}, {}'.format(level_point_x, level_point_y))
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
                # logger.debug('Skipping grey patch at {}, {}'.format(x_l[p_idx], y_l[p_idx]))
        else:
            # bad one: too white
            report['white_patches'] += 1
            nt_cnt += 1

        # possibly get an update
        nt_cnt, white_threshold = bfn['get_white_threshold'](
            nt_cnt, bad_batch_size, white_threshold, white_threshold_max, white_threshold_incr
        )
        if white_threshold == None:
            logger.warning('Max white threshold reached! Bailing out')
            break

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
    logger.info('Extracted {} good patches'.format(len(patch_point)))

    # in 'random' method, only one batch is done, so it doens't make sense to
    # return the last index. Instead signal that we're over with sampling.
    p_idx = None if method == 'random' else p_idx

    return patch_list, patch_point, p_idx
