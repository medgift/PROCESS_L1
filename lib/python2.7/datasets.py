# -*- coding: utf-8 -*-
################################################################################
# For copyright see the `LICENSE` file.
#
# This file is part of PROCESS_UC1.
################################################################################
"""Camelyon17 dataset family driver

Meta
====

:Authors:
    Mara Graziani et al.
    Marco E. Poleggi L<mailto:marco-emilio.poleggi -AT- hesge.ch>
"""
import os, glob, re, errno
from functions import preprocess, get_morp_im
import numpy as np
import matplotlib.pyplot as plt
import cv2
import integral
import openslide
import pprint as pp
import h5py as hd

class Dataset(object):
    """A camelyon17 dataset is structured like:

        |-- centre_<CID>
        |   |-- ...
        |   |-- patient_<PID>_node_<NID>.tif
        |   `-- ...
        | ...
        `-- lesion_annotations
            |-- ...
            |-- patient_<PID>_node_<NID>.xml
            |-- ...

    There is a static mapping between centres and slides:

        CID := 0 => slides for patients PID in [000, 019]
        CID := 1 => slides for patients PID in [020, 039]
        CID := 2 => slides for patients PID in [040, 059]
        CID := 3 => slides for patients PID in [060, 079]
        CID := 4 => slides for patients PID in [080, 099]


    Public attributes
    =================

    All optional. They can be set via :meth:`init`.

    .. py:attribute:: patients
        list(None). List of patients to process. They must be a subset of the
        whole datasets

    .. py:attribute:: logger
        obj(None). A :class:`logging.Logger` object. This is needed!
        **FIX-ME** a dummy one could be used

    ** to be completed **
    """
    name = ''
    patients = []
    slide_source_fld = ''
    xml_source_fld = ''
    results_dir = ''
    h5db_path = ''
    h5db_bname = 'patches'
    # old
    #   h5db_tree_def = '{}/level{}/centre{}/patient{}/node{}'
    h5db_tree_def = '{}/l{}/c{}/p{}/n{}'
    # old
    #   img_fname_prefix = 'level{}_centre{}_patient{}_node{}'
    img_fname_prefix = 'l{}_c{}_p{}_n{}'
    patient_sdir_fmt = 'l{}_c{}_p{}_n{}'
    files_counter = 0
    tum_counter = 0
    nor_counter = 0
    centres = []        # declare which centres to use in [0, 4]
    config = {}
    logger = None
    # filled in at init, mapped from the description above
    dataset = [
        # <CID>-indexed
        # {
        #     'path'      : #... to the centre_<CID>/ dir
        #     'patients'  : {
        #         '<PID>_<NID>': (<WSI_path>, <XML_path>),
        #         ...
        #     },
        # },
        # ...
    ]

    # pseudo private
    _h5db = None
    _centre_ranges = [
        # <CID>-indexed centre to patient static mapping
        #   (<start>, <end>), (...)
        range(0, 19),
        range(20, 39),
        range(40, 59),
        range(60, 79),
        range(80, 99),
    ]
    report = {
        'errors' : 0,
        'warnings' : 0
    }

    def make_batch_dir(self, batch_n):
        """Make a directory for batch results `batch_n`.

        :param  int batch_n: batch number

        :return str: something like

            <self.results_dir>/<batch_n>
        """
        bdir = os.path.join(self.results_dir, str(batch_n))
        try:
            os.makedirs(bdir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                self.logger.fatal('{}: cannot make batch dir: {}'.format(bdir, e))
                return None

        return bdir

    def make_patient_dir(self, info={}):
        """Make a subdirectory for a `patient` in the results directory using
        <self.patient_sdir_fmt> which has placeholders for values in the `info`
        dict.

        :param  dict info: must have keys

                'centre'    : <CID>,
                'patient'   : '<PID>',
                'node'      : '<NID>'

        :return str: something like

            <self.results_dir>/l<slide_level>_c<CID>_p<PID>_n<NID>

        """
        pdir = os.path.join(
           self.results_dir,
            (self.patient_sdir_fmt).format(
                self.config['settings']['slide_level'],
                info['centre'],
                info['patient'],
                info['node']
            )
        )
        try:
            os.makedirs(pdir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                self.logger.error('{}: cannot make patient dir: {}'.format(pdir, e))
                return None

        return pdir


    def get_image_fname_by_info(self, suffix, info={}, batch=''):
        """Get an image file name composed of

             <self.results_dir>/<batch>/<self.img_fname_prefix>_<suffix>.png

        where <self.img_fname_prefix> has placeholders for values in the
        `info` dict.

        :param  str directory: path to the results directory

        :param  str suffix: specific image suffix *without* the extension

        :param  dict info: must have keys

                'centre'    : '<CID>',
                'patient'   : '<PID>',
                'node'      : '<NID>'

        :return str: something like:

             results_dir/<batch>/l<slide_level>_c<CID>_p<PID>_n<NID>_<suffix>.png
        """
        return os.path.join(
            self.results_dir, str(batch),
            (self.img_fname_prefix + '_' + suffix).format(
                self.config['settings']['slide_level'],
                info['centre'],
                info['patient'],
                info['node']
            )
        )

    def get_image_fname(self, patient_dir, suffix, info={}, batch=None):
        """Get an image file name composed of

             <self.results_dir>/<patient_dir>/<suffix>_<batch>.png

        where <self.img_fname_prefix> has placeholders for values in the
        `info` dict.'<batch>' will be formatted to 4 chars and 0-left-padded.

        :param  str patient_dir: path to the results directory (must exist)

        :param  str suffix: specific image suffix *without* the extension

        :return str: if <patient_dir> := 'l<slide_level>_c<CID>_p<PID>_n<NID>',
        something like:

             results_dir/l<slide_level>_c<CID>_p<PID>_n<NID>/<suffix>_<batch>.png

        """

        return os.path.join(
            patient_dir, suffix +
            ("_{:0>4}".format(batch) if not batch == None  else '')
            + '.png',
        )

    def pid2cid(self, pid):
        """Return the 'centre' ID corresponding to a 'patient' ID (static
        mapping).

        :param int pid: patient ID in [0,99]

        :return int: the centre ID, or None if CID is not in `self.centres`
        """
        if pid > 99 or pid < 0:
            raise ValueError('{}: PID is outside range [0, 99]'.format(pid))

        cid = pid/20

        return cid if cid in self.centres else None

    def cid2pid(self, cid):
        """Return the 'patient' ID range corresponding to a 'centre' ID (static
        mapping).

        :param int cid: the centre ID

        :return range(int): patient ID range
        """
        if cid > max(self.centres) or cid < 0:
            raise ValueError('{}: CID is outside range [0, {}]'.format(cid, max(self.centres)))

        return self.centres[cid]


    def get_patients(self, centre):
        """Get a patient dict from dataset representation for `centre` (CID).

        :param int centre: centre ID

        :return dict: patients from dataset representation for `centre` (CID)
        """
        return self.dataset[centre]['patients']


    def get_wsi_path(self, centre, patient):
        """Get the WSI path for a patient of a centre.

        :param int centre: centre ID

        :param str patient: <PID>_<NID> string in the dataset representation

        :return str: the path to the WSI file
        """
        return self.dataset[centre]['patients'][patient][0]

    def get_annotation_path(self, centre, patient):
        """Get the annotation path for a patient of a centre.

        :param int centre: centre ID

        :param str patient: <PID>_<NID> string in the dataset representation

        :return str: the path to the XML annotation file
        """
        return self.dataset[centre]['patients'][patient][1]

    def set_files_counter(self, files_counter):
        self.files_counter = files_counter

    def get_pid(self, patient):
        """Extract the <PID> part from a patient <PID_NID> string.

        :param str patient: <PID>_<NID> string in the dataset representation

        :return str: the <PID>
        """
        return patient.split('_')[0]

    def get_nid(self, patient):
        """Extract the <NID> part from a patient <PID_NID> string.

        :param str patient: <PID>_<NID> string in the dataset representation

        :return str: the <NID>
        """
        return patient.split('_')[1]

    def get_info(self, centre, patient):
        """Build an info dict for a patient of a centre.

        :param int centre: centre ID

        :param str patient: <PID>_<NID> string in the dataset representation

        :return dict: like

            {
                'centre'    : <CID>,  # oh, holy redundancy...
                'patient'   : '<PID>',
                'node'      : '<NID>',
            }
        """
        return {
            'centre'    : centre,
            'patient'   : self.get_pid(patient),
            'node'      : self.get_nid(patient),
        }


    def store(self, info, patch_array, patch_point, tissue_type, batch=None):
        """Store a dataset of patches and locations int our `self._h5db` keyed
        at, respectively (blanks added for readibility):

            <tissue_type>/LevelN/CentreC/PatientP/NodeNo/[<batch>/]<patch_array>
            <tissue_type>/LevelN/CentreC/PatientP/NodeNo/[<batch>/]<patch_point>

        Intermediate key components comes from arg `info`. By default no
        "<batch>/" component is added.

        :param np.array patch_array: array of patches

        :param np.array patch_point: array of locations

        :param str tissue_type: normal, tumor, whatever...

        :param int batch: sampling batch number

        :return None
        """
        tree_path = self.h5db_tree_def.format(
            tissue_type,
            self.config['settings']['slide_level'],
            info['centre'],
            info['patient'],
            info['node']
        )
        if not batch == None:
            tree_path += '/{}'.format(batch)

        self.logger.debug("storing key: {}".format(tree_path))

        self._h5db[tree_path + '/' + 'patches'] = patch_array
        self._h5db[tree_path + '/' + 'locations'] = patch_point


    def store_patient(
            self, info, patch_array, patch_point, tissue_type, h5db,
            batch=None
    ):
        """Store a dataset of patches and locations for a given patient into `h5db`
        keyed at, respectively (blanks added for readibility):

            <tissue_type>/LevelN/CentreC/PatientP/NodeNo/[<batch>/]<patch_array>
            <tissue_type>/LevelN/CentreC/PatientP/NodeNo/[<batch>/]<patch_point>

        Intermediate key components comes from arg `info`. By default no
        "<batch>/" component is added.

        :param np.array patch_array: array of patches

        :param np.array patch_point: array of locations

        :param str tissue_type: normal, tumor, whatever...

        :param int batch: sampling batch number

        :param obj h5db: a pymod:h5py open DBfile

        :return None

        """
        tree_path = self.h5db_tree_def.format(
            tissue_type,
            self.config['settings']['slide_level'],
            info['centre'],
            info['patient'],
            info['node']
        )
        if not batch == None:
            tree_path += '/{}'.format(batch)

        self.logger.debug("storing key: {}".format(tree_path))

        h5db[tree_path + '/' + 'patches'] = patch_array
        h5db[tree_path + '/' + 'locations'] = patch_point


    def extract_patches(self):
        """
        (more doc please)
        """
        errors = 0
        warnings = 0
        settings = self.config['settings']
        for centre in self.centres:
            for patient in self.get_patients(centre):
                self.logger.info('processing patient: {}'.format(patient))

                slide_path = self.get_wsi_path(centre, patient)
                xml_path = self.get_annotation_path(centre, patient)
                info = self.get_info(centre, patient)

                pat_res_dir = self.make_patient_dir(info)
                if not pat_res_dir:
                    self.logger.error("patient {}: problems with results dir...".format(patient))
                    errors += 1
                    continue

                h5db_path = os.path.join(pat_res_dir, self.h5db_bname + '.h5')
                try:
                    h5db = hd.File(h5db_path, 'w')
                except Exception as e:
                    self.logger.error(
                        "patient {}: can't open my H5 DB '{}': {} ".format(
                            patient, h5db_path, e
                        )
                    )
                    errors += 1
                    continue

                slide, annotations_mask, rgb_im, im_contour = preprocess(
                    slide_path,
                    xml_path,
                    slide_level=settings['slide_level']
                )


                # reverting the tumor mask to find normal tissue and extract patches
                # Note :
                #    normal_mask = tissu mask(morp_im) - tummor mask(annotations_mask)
                morp_im = get_morp_im(rgb_im)
                normal_im = morp_im - annotations_mask  # np.min(normal_im) := -1.0
                normal_im = normal_im == 1.0
                normal_im = (normal_im).astype(int)
                # fimport pdb; pdb.set_trace()
                
                # masks are the same for any sample batch ;-)
                # [TO-DO] make switchable from config/CL
                plt.figure()
                plt.imshow(annotations_mask)
                img_file = self.get_image_fname(pat_res_dir, 'annotation_mask', info)
                plt.savefig(img_file)
                plt.close()
                self.logger.info(
                    'patient {}: Annotation mask image saved to: {}'.format(
                        patient, img_file
                    )
                )

                plt.figure()
                plt.imshow(normal_im)
                img_file = self.get_image_fname(pat_res_dir, 'normal_tissue_mask', info)
                plt.savefig(img_file)
                plt.close()
                self.logger.info(
                    'patient {}: Normal tissue mask image saved to: {}'.format(
                        patient, img_file
                    )
                )

                opts = dict(
                    map(
                        lambda k: (k, settings[k]), (
                            'area_overlap',
                            'bad_batch_size',
                            'gray_threshold',
                            'margin_width_x',
                            'margin_width_y',
                            'method',
                            'n_samples',
                            'patch_size',
                            'slide_level',
                            'white_level',
                            'white_threshold',
                            'white_threshold_incr',
                            'white_threshold_max',
                        )
                    )
                )

                # batch sample & store -- keep it small to avoid OOM!  In
                # "linear" sampling mode, more batches might be needed, so go
                # for a run and get the extracted pathes and the last
                # index. Loop until no patches come out

                # [TO-DO] store info in _per-patient_ H5 DB

                # a patient case (:= slide) the tumor annotation mask is
                # usually (much) smaller than the normal tissue mask, thus a
                # different number of batches is needed to extract all the
                # tumor and normal patches. So we compute then normal tissue
                # mask once. Apart from that, there's no relation between
                # tumor and normal patches, hence we batch-loop two times: a
                # first time for the tumor case and a second time for the
                # normal case. N.B. In 'random' sampling mode, just one batch
                # is ever done.

                index = 0 # ignored in 'random' mode -- only one batch done
                tum_patch_point = []
                bcnt_t, bcnt_n = 0, 0
                last_idx_t = last_idx_n = -1

                if settings['window']:
                    self.logger.info(
                        "patient {}: restricting nonzero points range to {}%, {}%".format(
                            patient, settings['window'][0], settings['window'][1]
                        )
                    )
                nzx_n, nzy_n = integral.nonzero_range(normal_im, settings['window'])

                # *** Warning! *** Split loops doesn't work if we want to show
                # images: there's data dependency on "normal_patches_locations".

                # normal tissue
                while(True):
                    self.logger.info("patient {}: >>> [normal] starting batch {}".format(patient, bcnt_n))

                    opts['start_idx'] = last_idx_n + 1
                    nor_patch_list , nor_patch_point, last_idx_n = integral.patch_sampling(
                        slide, normal_im, nzx_n, nzy_n,  **opts
                    )
                    if nor_patch_point and nor_patch_list:
                        nor_patch_array = np.asarray(nor_patch_list)
                        normal_patches_locations = np.array(nor_patch_point)
                        self.store_patient(info, nor_patch_array, nor_patch_point, 'normal', h5db, bcnt_n)
                    else:
                        self.logger.info(
                            'patient {}: batch {}: no (more) normal patches'.format(
                                patient, bcnt_n
                            )
                        )
                        break


                    self.nor_counter += len(nor_patch_array)

                    self.logger.info("patient {}: <<< [normal] done batch {}".format(patient, bcnt_n))

                    if last_idx_n == None:
                        # in 'random' method, this tells us that we're done sampling
                        break

                    bcnt_n += 1
                # {end-while}

                # TO-DO: batch runs should be better encapsulated (aux fun/method)...

                # tumors masks are usually too small for windowed sampling, so
                # take the full range
                nzx_t, nzy_t = integral.nonzero_range(annotations_mask, [])
                while(True):

                    self.logger.info("patient {}: >>> [tumor] starting batch {}".format(patient, bcnt_t))

                    opts['start_idx'] = last_idx_t + 1
                    tum_patch_list, tum_patch_point, last_idx_t = integral.patch_sampling(
                        slide, annotations_mask, nzx_t, nzy_t, **opts
                    )
                    if tum_patch_list and tum_patch_point:
                        tum_patch_array = np.asarray(tum_patch_list)
                        tum_locations = np.array(tum_patch_point)
                        self.store_patient(info, tum_patch_array, tum_locations, 'tumor', h5db, bcnt_t)
                    else:
                        self.logger.info(
                            'patient {}: batch {}: no (more) tumor patches'.format(
                                patient, bcnt_t
                            )
                        )
                        break


                    if opts['method'] == 'random':
                        if bcnt_n != bcnt_t:
                            self.logger.error("[BUG] Can't make scatter image(s): batch count mismatch")
                            errors += 1
                        else:
                            # plotting the tumor locations in the XML file Drawing the
                            # normal patches sampling points tumor_locations.png shows the
                            # tumor patches locations in red and the normal patches
                            # locations in green
                            tumor_locations_im = rgb_im
                            plt.figure()
                            plt.imshow(tumor_locations_im)
                            # Warning! Data dependency on previous normal batch run
                            for p_x, p_y in normal_patches_locations:
                                plt.scatter(p_y, p_x, c='g')
                            for p_x, p_y in tum_locations:
                                plt.scatter(p_y, p_x, c='r')

                            img_file = self.get_image_fname(pat_res_dir, 'tumor_locations', info, bcnt_t)
                            plt.savefig(img_file)
                            plt.close()
                            self.logger.info(
                                'patient {}: batch {}: tumor locations image saved to: {}'.format(
                                    patient, bcnt_t, img_file
                                )
                            )

                    self.tum_counter += len(tum_patch_array)

                    self.logger.info("patient {}: <<< [tumor] done batch {}".format(patient, bcnt_t))

                    if last_idx_t == None:
                        # in 'random' method, this tells us that we're done sampling
                        break

                    bcnt_t += 1
                # {end-while}

                h5db.close()
                self.logger.info("patient {}: processed in {} (normal) + {} (tumor) batches".format(
                    patient, bcnt_n, bcnt_t)
                )
                self.logger.info("patient {}: data saved to H5 DB: {}".format(patient, h5db_path))
            # {end-for-patient}
        # {end-for-centre}

        self.report['errors'] = errors
        self.report['warnings'] = warnings


    def __init__(
            self,
            name='',
            patients=[],
            slide_source_fld='',
            xml_source_fld='',
            results_dir='',
            # h5db_path='',
            files_counter=0,
            tum_counter=0,
            nor_counter=0,
            centres=[],
            config={},
            logger=None
    ):
        self.name = name
        self.patients = patients if patients else []  # handle None
        self.slide_source_fld = slide_source_fld
        self.xml_source_fld = xml_source_fld
        self.results_dir = results_dir
        # self._h5db_path=h5db_path,
        self.files_counter = files_counter
        self.tum_counter = tum_counter
        self.nor_counter = nor_counter
        self.centres = centres
        self.config = config
        self.logger = logger

        # self._h5db = hd.File(h5db_path, 'w')

        # prepare a bioler-plate representation
        centre_paths = map(
            lambda c: slide_source_fld + str(c), centres
        )
        self.dataset = list(
            map(
                lambda cp: ({'path' : cp[1], 'patients' : {}}),
                zip(centres, centre_paths)
            )
        )

        # validate patient names
        errs = 0
        valid_patients = []
        for patient in self.patients:
            if not re.match(config['camelyon17']['patient_name_regex'], patient):
                errs += 1
                logger.error(
                    '{}: invalid input patient: does not match regex "{}"'.format(
                        patient, config['camelyon17']['patient_name_regex']
                    )
                )
            else:
                valid_patients += [patient]

        # filter out unwanted content in input dir
        annots = filter(
            lambda x: re.search(config['camelyon17']['patient_name_regex'] + '.xml', x),
            os.listdir(xml_source_fld)
        )
        # keep only given patients. Use a for to catch errors
        if valid_patients:
            logger.debug('valid_patients: {}'.format(valid_patients))
            wannots = []
            while valid_patients:
                p = valid_patients.pop()
                x = p + '.xml'
                if x in annots:
                    # valid_patients.remove(p)
                    wannots += [x]
                else:
                    errs += 1
                    logger.error('{}: input patient not found in dataset'.format(p))

            annots = wannots

        if errs:
            msg = 'Problems with patient list: see errors above.'
            logger.fatal(msg)
            raise RuntimeError(msg)

        if not annots:
            msg = 'no (requested) patient found in dataset:\n%s'
            logger.fatal(msg)
            raise RuntimeError(msg)

        logger.debug('annots:\n%s' % pp.pformat(annots))

        # build the complete dataset representation
        for xml in annots:
            match = re.match(config['camelyon17']['patient_name_regex'] + '.xml', xml)
            try:
                pid = int(match.group('PID'))
                nid = int(match.group('NID'))
                logger.debug('XML: {}: PID={}, NID={}'.format(xml, pid, nid))
            except Exception as e:
                # this should never happen... but you _never_ know ;-)
                logger.warn('[BUG] {}: skipping malformatted XML annotation file name'.format(xml))
                continue

            # look for corresponding slides
            cid = self.pid2cid(pid)
            logger.debug('PID: {} => CID={}'.format(pid, cid))

            if cid == None:
                logger.info('{}: skipping XML annotation file name: not in our requested centre range'.format(xml))
                continue

            for sld in filter(
                lambda c: re.search(config['camelyon17']['patient_name_regex'] + '.tif', c),
                os.listdir(self.dataset[cid]['path'])
            ):
                # you never know...
                match = re.match(config['camelyon17']['patient_name_regex'] + '.tif', sld)
                try:
                    pid = int(match.group('PID'))
                    nid = int(match.group('NID'))
                    logger.debug('TIF: {}: PID={}, NID={}'.format(sld, pid, nid))
                except Exception as e:
                    logger.warn('{}: skipping malformatted WSI file name'.format(sld))
                    continue

                pid_nid = '%03d_%d' % (pid, nid)
                self.dataset[cid]['patients'][pid_nid] = (
                    os.path.join(self.dataset[cid]['path'], sld),
                    os.path.join(xml_source_fld, xml)
                )

        logger.debug('dataset: \n%s' % pp.pformat(self.dataset))
        integral.mod_init(mlogger=logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # self._h5db.close()
        pass

################################################################################
if __name__ == '__main__':
    print(__doc__)
