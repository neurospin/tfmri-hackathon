"""
Script to process the data for the Neurospin 2018 hackathon

Author: Bertrand Thirion, 2018
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from nistats.first_level_model import FirstLevelModel
from nistats.thresholding import map_threshold
from nistats.reporting import plot_design_matrix, plot_contrast_matrix
from nistats.reporting import get_clusters_table

t_r = 2.4
slice_time_ref = 0.5

#########################################################################
# Prepare data

data_dir = '/neurospin/tmp/tfmri-hackathon-2018/data'
subjects = ['sub-S%02d' % i for i in range(1, 21)]
session = 'ses-V1'
task = 'localizer'

for subject in subjects:
    source_dir = os.path.join(data_dir, task, 'sourcedata', subject, session,
                              'func')
    paradigm_file = os.path.join(source_dir, '%s_%s_task-%s_events.tsv'
                                 % (subject, session, task))
    derivative_dir = os.path.join(data_dir, task, 'derivatives',
                                  'spmpreproc_%s' % session, subject)
    fmri_img = os.path.join(derivative_dir, 'wrr%s_%s_task-%s_bold.nii.gz'
                            % (subject, session, task))
    paradigm = pd.read_csv(paradigm_file, sep='\t')
    paradigm['trial_type'] = paradigm['trial_name']

    #########################################################################
    # Prepare output directory
    write_dir = os.path.join(derivative_dir, 'glm')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    #########################################################################
    # Perform first level analysis
    # ----------------------------
    # Setup and fit GLM
    first_level_model = FirstLevelModel(t_r, slice_time_ref,
                                        hrf_model='glover + derivative')
    first_level_model = first_level_model.fit(fmri_img, paradigm)

    #########################################################################
    # Estimate contrasts
    # ------------------
    # get the design matrix and save it to disk
    design_matrix = first_level_model.design_matrices_[0]
    design_matrix.to_csv(os.path.join(write_dir, 'design_matrix.csv'))
    ax = plot_design_matrix(design_matrix)
    plt.savefig(os.path.join(write_dir, 'design_matrix.png'))

    # Specify the contrasts
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])

    contrasts["audio"] =\
        contrasts["r_hand_audio"] + contrasts["l_hand_audio"] +\
        contrasts["computation_audio"] + contrasts["sentence_audio"]
    contrasts["video"] =\
        contrasts["r_hand_video"] + contrasts["l_hand_video"] +\
        contrasts["computation_video"] + contrasts["sentence_video"]
    contrasts["computation"] = contrasts["computation_audio"] +\
        contrasts["computation_video"]
    contrasts["sentences"] = contrasts["sentence_audio"] +\
        contrasts["sentence_video"]

    #########################################################################
    # Short list of more relevant contrasts
    contrasts = {
        "left-right": (contrasts["l_hand_audio"] + contrasts["l_hand_video"]
                       - contrasts["r_hand_audio"]
                       - contrasts["r_hand_video"]),
        "H-V": contrasts["h_checkerboard"] - contrasts["v_checkerboard"],
        "audio-video": contrasts["audio"] - contrasts["video"],
        "video-audio": -contrasts["audio"] + contrasts["video"],
        "computation-sentences": (contrasts["computation"] -
                                  contrasts["sentences"]),
        "reading-visual": (contrasts["sentence_video"] -
                           contrasts["h_checkerboard"])
        }

    # plot_contrast_matrix(pd.DataFrame(contrasts), design_matrix)
    # plt.savefig(os.path.join(write_dir, 'contrasts.png'))

    #########################################################################
    # contrast estimation

    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        z_map = first_level_model.compute_contrast(contrast_val,
                                                   output_type='z_score')
        effect_map = first_level_model.compute_contrast(
            contrast_val, output_type='effect_size')
        # Create snapshots of the contrasts
        _, threshold = map_threshold(z_map, threshold=.05,
                                     height_control='fdr')
        out_file = os.path.join(write_dir, '%s_z_map.png' % contrast_id)
        display = plotting.plot_stat_map(z_map, display_mode='z',
                                         threshold=threshold,
                                         title=contrast_id,
                                         output_file=out_file)
        # write image to disk
        z_map.to_filename(os.path.join(write_dir, '%s_z_map.nii.gz'
                          % contrast_id))
        effect_map.to_filename(os.path.join(write_dir, '%s_effects.nii.gz'
                               % contrast_id))
        clusters_table = get_clusters_table(z_map, threshold, 10)
        clusters_table.to_csv(os.path.join(write_dir, '%s_clusters.csv'
                                           % contrast_id))

plt.close('all')
import pandas as pd
n_subjects = len(subjects)
group_design_matrix = pd.DataFrame([1] * n_subjects, columns=['intercept'])
group_dir = os.path.join(data_dir, task, 'derivatives', 'group')
if not os.path.exists(group_dir):
    os.mkdir(group_dir)

from nistats.second_level_model import SecondLevelModel
for contrast_id in contrasts.keys():
    cmap_filenames = [
        os.path.join(data_dir, task, 'derivatives', 'spmpreproc_%s' %
                     session, subject, 'glm', '%s_effects.nii.gz' % contrast_id)
                     for subject in subjects]
    second_level_model = SecondLevelModel().fit(
        cmap_filenames, design_matrix=group_design_matrix)
    z_map = second_level_model.compute_contrast(output_type='z_score')
    thresholded_map, threshold = map_threshold(
        z_map, threshold=.1, height_control='fdr', cluster_threshold=10)
    z_map.to_filename(os.path.join(group_dir, '%s_z_map.nii.gz'
                      % contrast_id))
    output_file = os.path.join(group_dir, '%s_z_map.png'
                               % contrast_id)
    plotting.plot_stat_map(thresholded_map,
                           title='%s, fdr = .1' % contrast_id,
                           threshold=threshold, output_file=output_file)
    clusters_table = get_clusters_table(z_map, threshold, 10)
    clusters_table.to_csv(os.path.join(group_dir, '%s_clusters.csv'
                                       % contrast_id))

# plotting.show()
