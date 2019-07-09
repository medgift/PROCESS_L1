import os, glob, re
from functions import preprocess, get_morp_im
import numpy as np
import matplotlib.pyplot as plt
import cv2
from integral import patch_sampling_using_integral
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
    """
    name = ''
    patients = []
    slide_source_fld = ''
    xml_source_fld = ''
    results_dir = ''
    h5db_path=''
    h5db_tree_def = '{}/level{}/centre{}/patient{}/node{}'
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

    img_fname_prefix = 'level{}_centre{}_patient{}_node{}'

    # pseudo private
    _h5db=None
    _centre_ranges = [
        # <CID>-indexed centre to patient static mapping
        #   (<start>, <end>), (...)
        range(0, 19),
        range(20, 39),
        range(40, 59),
        range(60, 79),
        range(80, 99),
    ]

    def get_image_fname(self, suffix, info={}):
        """Get an image file name composed of

             <self.results_dir>/<self.img_fname_prefix>_<suffix>

        where <self.img_fname_prefix> has placeholders for values in the
        `info` dict.

        :param  str directory: path to the results directory

        :param  str suffix: specific image suffix

        :param  dict info: must have keys

                'centre'    : <CID>,
                'patient'   : '<PID>',
                'node'      : '<NID>'

        :return str: something like:

             results_dir/level<slide_level>_centre<CID>_patient<PID>_node<NID>_<suffix>
        """
        return os.path.join(
            self.results_dir,
            (self.img_fname_prefix + '_' + suffix).format(
                self.config['settings']['slide_level'],
                info['centre'],
                info['patient'],
                info['node']
            )
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


    def count_annotation_files(self):
        raise AppError, 'Obsoleted by `get_patients()`'

        files_counter = 0
        for centre in self.centres:
            annotation_list = self.get_annotation_list(centre)
            files_counter += len(annotation_list)
        return files_counter

    def get_patients(self, centre):
        """Get a patient dict from dataset representation for `centre` (CID).

        :param int centre: centre ID

        :return dict: patients from dataset representation for `centre` (CID)
        """
        return self.dataset[centre]['patients']

    def get_annotation_list(self, centre):
        """Get an XML annotation file list for `centre` (CID).

        **Obsoleted by `get_patients()`**

        :param int centre: centre ID

        :return list: a list of
        """
        raise AppError, 'Obsoleted by `get_patients()`'

        xml_source_fld = self.xml_source_fld
        xml_of_selected_centre = []
        xml_list = os.listdir(xml_source_fld)
        for x in xml_list:
            self.logger.debug('[datasets] XML file: {}'.format(x))
            identifier = x[-13]
            if centre == 0:
                if int(identifier)<=1:
                    xml_of_selected_centre.append(x)
            elif centre == 1:
                if int(identifier) == 2 or int(identifier) == 3:
                    xml_of_selected_centre.append(x)
            elif centre == 2:
                if int(identifier) == 4 or int(identifier) == 5:
                    xml_of_selected_centre.append(x)
            elif centre == 3:
                if int(identifier) == 6 or int(identifier) == 7:
                    xml_of_selected_centre.append(x)
            elif centre == 4:
                if int(identifier) == 8 or int(identifier) == 9:
                    xml_of_selected_centre.append(x)

        self.logger.debug('[datasets] xml_of_selected_centre: {}'.format(xml_of_selected_centre))
        return np.sort(xml_of_selected_centre)

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


    def store(self, info, patch_array, patch_point, tissue_type):
        tree_path = self.h5db_tree_def.format(
            tissue_type,
            self.config['settings']['slide_level'],
            info['centre'],
            info['patient'],
            info['node']
        )
        self._h5db[tree_path + '/' + 'patches'] = patch_array
        self._h5db[tree_path + '/' + 'locations'] = patch_point


    def extract_patches(self):
        """
        (doc please)
        """
        # ...useless!?
        # self.set_files_counter(self.count_annotation_files())

        # self.logger.info(
        #     '[datasets] {0} [extract_patches] {1} total annotation files.'.
        #     format(self.name, self.files_counter)
        # )

        settings = self.config['settings']
        for centre in self.centres:
            for patient in self.get_patients(centre):
                self.logger.info('processing patient: {}'.format(patient))

                slide_path = self.get_wsi_path(centre, patient)
                xml_path = self.get_annotation_path(centre, patient)
                info = self.get_info(centre, patient)

                slide, annotations_mask, rgb_im, im_contour = preprocess(
                    slide_path,
                    xml_path,
                    slide_level=settings['slide_level']
                )

                tum_patch_list, tum_patch_point = patch_sampling_using_integral(
                    slide,
                    settings['slide_level'],
                    annotations_mask,
                    settings['patch_size'],
                    settings['n_samples'],
                    logger = self.logger
                )
                # prefer np arrays...
                tum_patch_array = np.asarray(tum_patch_list)
                tum_locations = np.array(tum_patch_point)
                # store it
                self.store(info, tum_patch_array, tum_locations, 'tumor')

                # reverting the tumor mask to find normal tissue and extract patches
                #    Note :
                #    normal_mask = tissu mask(morp_im) - tummor mask(annotations_mask)

                ##### [BUG] ??? restart from here ##

                morp_im = get_morp_im(rgb_im)
                normal_im = morp_im - annotations_mask  ## np.min(normal_im) := -1.0
                normal_im = normal_im == 1.0
                normal_im = (normal_im).astype(int)
                # sampling normal patches with uniform distribution
                nor_patch_list , nor_patch_point = patch_sampling_using_integral(
                    slide,
                    settings['slide_level'],
                    normal_im,
                    settings['patch_size'],
                    settings['n_samples'],
                    logger = self.logger
                )
                nor_patch_array = np.asarray(nor_patch_list)
                normal_patches_locations = np.array(nor_patch_point)
                self.store(info, nor_patch_array, nor_patch_point, 'normal')

                # plotting the tumor locations in the XML file Drawing the
                # normal patches sampling points tumor_locations.png shows the
                # tumor patches locations in red and the normal patches
                # locations in green
                tumor_locations_im = rgb_im
                plt.figure()
                plt.imshow(tumor_locations_im)
                for p_x, p_y in normal_patches_locations:
                    plt.scatter(p_y, p_x, c='g')
                    #cv2.circle(tumor_locations_im,(p_y,p_x),30,(0,255,0),10)
                for p_x, p_y in tum_locations:
                    plt.scatter(p_y, p_x, c='r')
                    #cv2.circle(tumor_locations_im,(p_y,p_x),30,(255,0,0), 10)

                img_file = self.get_image_fname('tumor_locations.png', info)
                plt.savefig(img_file)
                plt.close()
                self.logger.info('Tumor locations image saved to: {}'.format(img_file))

                plt.figure()
                plt.imshow(annotations_mask)
                img_file = self.get_image_fname('annotation_mask.png', info)
                plt.savefig(img_file)
                plt.close()
                self.logger.info('Annotation mask and normal tissue mask saved to: {}'.format(img_file))

                plt.figure()
                plt.imshow(normal_im)
                img_file = self.get_image_fname('normal_tissue_mask.png', info)
                plt.savefig(img_file)
                plt.close()
                self.logger.info('Normal tissue mask saved to: {}'.format(img_file))

                plt.close('all')

                self.tum_counter += len(tum_patch_array)
                self.nor_counter += len(nor_patch_array)


    def __init__(
            self,
            name='',
            patients=[],
            slide_source_fld='',
            xml_source_fld='',
            results_dir='',
            h5db_path='',
            files_counter=0,
            tum_counter=0,
            nor_counter=0,
            centres=[],
            config={},
            logger=None
    ):
        self.name = name
        self.patients = patients,
        self.slide_source_fld = slide_source_fld
        self.xml_source_fld = xml_source_fld
        self.results_dir = results_dir
        self._h5db_path=h5db_path,
        self.files_counter = files_counter
        self.tum_counter = tum_counter
        self.nor_counter = nor_counter
        self.centres = centres
        self.config = config
        self.logger = logger

        self._h5db = hd.File(h5db_path, "w")

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
        for patient in self.patients:
            if not re.match(config['camelyon17']['patient_name_regex'], patient):
                errs += 1
                @@@@ HERE @@@@

        # filter our unwanted content
        annots = filter(
            lambda c: re.search(config['camelyon17']['patient_name_regex'] + '.xml', c),
            os.listdir(xml_source_fld)
        )
        logger.debug('annots: \n%s' % pp.pformat(annots))

        # build the complete dataset representation
        for xml in annots:
            match = re.match(config['camelyon17']['patient_name_regex'] + '.xml', xml)
            try:
                pid = int(match.group('PID'))
                nid = int(match.group('NID'))
                logger.debug('XML: {}: PID={}, NID={}'.format(xml, pid, nid))
            except Exception as e:
                logger.warn('{}: skipping malformatted XML annotation file name'.format(xml))

            # look for corresponding slides
            cid = self.pid2cid(pid)
            logger.debug('PID: {} => CID={}'.format(pid, cid))

            if cid == None:
                logger.info('{}: skipping XML annotation file name -- not in our centres range'.format(xml))
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

                pid_nid = '%03d_%d' % (pid, nid)
                self.dataset[cid]['patients'][pid_nid] = (
                    os.path.join(self.dataset[cid]['path'], sld),
                    os.path.join(xml_source_fld, xml)
                )

        logger.debug('dataset: \n%s' % pp.pformat(self.dataset))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._h5db.close()
