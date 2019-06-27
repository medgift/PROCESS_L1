import os, glob, re
from functions import preprocess, load_slide
import numpy as np
import matplotlib.pyplot as plt
import cv2
import integral
import openslide
import pprint as pp

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

    ...more doc please
    """
    name = ''
    slide_source_fld = ''
    xml_source_fld = ''
    files_counter = 0
    tum_counter = 0
    nor_counter = 0
    centers = []
    config = {}
    logger = None
    # filled in at init, mapped from the description above
    dataset = {
        # '<CID>' : {
        #     'path'      : #... to the centre_<CID>/ dir
        #     'slds_anns' : [
        #         (<WSI_path>), (<XML_path>), ...
        #     ]
        # },
        # ...
    }

    def count_annotation_files(self):
        files_counter = 0
        for centre in self.centres:
            annotation_list = self.get_annotation_list(centre)
            files_counter += len(annotation_list)
        return files_counter

    def get_annotation_list(self, centre):
        """Get an XML annotation file list for `centre` (CID).

        Arguments
        +++++++++

        :param int centre: centre ID

        :return dict()
        """
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

        return np.sort(xml_of_selected_centre)

    def get_wsi_path(self, centre, xml_file):
        wsi_file = xml_file[:-3]+'tif'
        self.logger.info('Working with (WSI): {}'.format(wsi_file))

        #    wsi_file = xml_file[:]
        #    return
        source_path = os.path.join(self.slide_source_fld, 'centre_{}'.format(centre))
        return os.path.join(source_path, wsi_file)

    def set_files_counter(self, files_counter):
        self.files_counter = files_counter

    def get_info(self, xml_path, centre):
        info = {}
        info['centre']=centre
        info['patient'] = info['node'] = ''
        info['patient'] = xml_path.split('patient_')[1].split('_node_')[0]
        info['node'] = xml_path.split('patient_')[1].split('_node_')[1].strip('.xml')

        return info

    def store(self, h5db, info, patch_array, patch_point, tissue_type):
        h5db[
            '{}/level{}/centre{}/patient{}/node{}/patches'.format(
                tissue_type,
                self.settings['slide_level'],
                info['centre'],
                info['patient'],
                info['node']
            )
        ] = patch_array
        h5db[
            '{}/level{}/centre{}/patient{}/node{}/locations'.format(
                tissue_type,
                self.settings['slide_level'],
                info['centre'],
                info['patient'],
                info['node']
            )
        ] = patch_point


    def extract_patches(self, h5db, new_folder):
        '''
        for centre in self.centres:
            print '[cnn][patch_extraction] Selected Centre: ', centre
            # each centre may have more than one annotation XML file, so here we retrieve
            # a list of all the XMLs related to the current centre
            annotation_list = np.sort(self.get_annotation_list(centre, self.xml_source_fld))
            # for each XML file in the annotation list
            # we want to extract tumor and normal patches
            for xml_file in annotation_list:
                files_counter +=1 # variable to shape the final data vector
        '''
        self.set_files_counter(self.count_annotation_files())

        self.logger.info(
            '[datasets] {0} [extract_patches] {1} total annotation files.'.
            format(self.name, self.files_counter)
        )

        for centre in self.centres:
            annotation_list = self.get_annotation_list(centre)
            for xml_file in annotation_list:
                self.logger.debug('[datasets] XML annotation file: {}'.format(xml_file))
                slide_path = self.get_wsi_path(centre, xml_file)
                xml_path = os.path.join(self.xml_source_fld, xml_file)
                # retrieving the information about the file analysed.
                #   info is a dictionary with the following keys:
                #   info['centre'], current centre number
                #   info['patient'], current patient number
                #   info['node'], current WSI node
                info = self.get_info(xml_path, centre)
                #functions.setDBHierarchy(h5db, self.settings,info)
                if info['patient']=='008_Mask.tif':
                    continue
                if xml_path != None: ## add check slide is open and ok
                    slide, annotations_mask, rgb_im, im_contour = preprocess(
                        slide_path,
                        xml_path,
                        slide_level=self.settings['slide_level']
                    )


                    tum_patch_list, tum_patch_point = integral.patch_sampling_using_integral(
                        slide,
                        self.settings['slide_level'],
                        annotations_mask,
                        self.settings['patch_size'],
                        self.settings['n_samples']
                    )
                    # conversion of the lists to np arrays
                    tum_patch_array = np.asarray(tum_patch_list)
                    tum_locations = np.array(tum_patch_point)
                    # storage in the HDF5 db
                    self.store(h5db, info, tum_patch_array, tum_locations, 'tumor')

                    # reverting the tumor mask to find normal tissue and extract patches
                    #    Note :
                    #    normal_mask = tissu mask(morp_im) - tummor mask(annotations_mask)

                    ##### restart from here ##

                    morp_im = functions.get_morp_im(rgb_im)
                    normal_im = morp_im - annotations_mask  ## np.min(normal_im) := -1.0
                    normal_im = normal_im == 1.0
                    normal_im = (normal_im).astype(int)
                    # sampling normal patches with uniform distribution
                    nor_patch_list , nor_patch_point = integral.patch_sampling_using_integral(
                                                        slide,
                                                        self.settings['slide_level'],
                                                        normal_im,
                                                        self.settings['patch_size'],
                                                        self.settings['n_samples']
                                                        )
                    nor_patch_array = np.asarray(nor_patch_list)
                    normal_patches_locations = np.array(nor_patch_point)
                    # storing the normal patches and their locations
                    self.store(h5db, info, nor_patch_array, nor_patch_point, 'normal')


                    ''' Visualisation '''

                    # plotting the tumor locations in the XML file
                    # Drawing the normal patches sampling points
                    # tumor_locations.png shows the tumor patches locations in red
                    # and the normal patches locations in green
                    tumor_locations_im = rgb_im
                    plt.figure()
                    plt.imshow(tumor_locations_im)
                    for p_x,p_y in normal_patches_locations:
                        plt.scatter(p_y, p_x, c='g')
                        #cv2.circle(tumor_locations_im,(p_y,p_x),30,(0,255,0),10)
                    for p_x,p_y in tum_locations:
                        plt.scatter(p_y, p_x, c='r')
                        #cv2.circle(tumor_locations_im,(p_y,p_x),30,(255,0,0), 10)
                    print '[cnn][patch_extraction] Saving tumor locations image'
                    plt.savefig(
                        os.path.join(
                        new_folder,'level{}_centre{}_patient{}_node{}_tumor_locations.png'.format(
                            self.settings['slide_level'],
                            info['centre'],
                            info['patient'],
                            info['node'])))
                    plt.close()
                    #print 'Saving tumor locations image'
                    #plt.savefig('tumor_locations_patient0{}_node{}'.format(info['patient'], info['node']))

                    print '[cnn][patch_extraction] Saving annotation mask and normal tissue mask'
                    plt.figure()
                    plt.imshow(annotations_mask)
                    plt.savefig(
                            os.path.join(
                            new_folder,
                            'level{}_centre{}_patient{}_node{}_annotation_mask.png'.format(
                                                                                    self.settings['slide_level'],
                                                                                    info['centre'],
                                                                                    info['patient'],
                                                                                    info['node']
                                                                                    )
                            ))
                    plt.close()

                    plt.figure()
                    plt.imshow(normal_im)
                    plt.savefig(
                        os.path.join(
                            new_folder,
                            'level{}_centre{}_patient{}_node{}_normal_tissue_mask.png'.format(
                                                                                        self.settings['slide_level'],
                                                                                        info['centre'],
                                                                                        info['patient'],
                                                                                        info['node'])
                        ))
                    plt.close()
                    plt.close('all')

                    self.tum_counter += len(tum_patch_array)
                    self.nor_counter += len(nor_patch_array)
                    #self.nor_counter = 0


    def __init__(
            self,
            name='',
            slide_source_fld='',
            xml_source_fld='',
            files_counter=0,
            tum_counter=0,
            nor_counter=0,
            centres=[],
            config={},
            logger=None
    ):
        self.name = name
        self.slide_source_fld = slide_source_fld
        self.xml_source_fld = xml_source_fld
        self.files_counter = files_counter
        self.tum_counter = tum_counter
        self.nor_counter = nor_counter
        self.centres = centres
        self.config = config
        self.logger = logger

        # match XML files in 'lesion_annotations/' corresponding
        # to the TIFFs in 'centre_<CID>/'
        centre_paths = map(
            lambda c: slide_source_fld + str(c), centres
        )
        # for (c, p) in zip(centers, centre_paths):
        #     self.dataset[c] = p

        self.dataset = dict(
            map(
                lambda cp: (cp[0], { 'path' : cp[1], 'slds_anns' : [] }),
                zip(centres, centre_paths)
            )
        )

        logger.debug('dataset: \n%s' % pp.pformat(self.dataset))

        annots = filter(
            lambda c: re.search(config['camelyon17']['patient_name_regex'] + '.xml', c),
            os.listdir(xml_source_fld)
        )
        for cdir in centre_paths:
            slds_anns = []        # for tuples <slide, annotations>
            slds = filter(
                lambda c: re.search(config['camelyon17']['patient_name_regex'] + '.tif', c),
                os.listdir(xml_source_fld)
            )
