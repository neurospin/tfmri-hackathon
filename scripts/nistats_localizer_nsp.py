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
from contrasts_localizer import make_localizer_contrasts

t_r = 2.4
slice_time_ref = 0.5

#########################################################################
# Prepare data

data_dir = '/neurospin/tmp/tfmri-hackathon-2018/shared_data'
subjects = ['sub-S%02d' % i for i in range(1, 21)]
session = 'ses-V1'
task = 'localizer'

for subject in subjects:
    #########################################################################
    # Define data paths
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
    contrasts = make_localizer_contrasts(design_matrix)

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
        _, threshold = map_threshold(z_map, level=.05,
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

plt.close('all')
import pandas as pd
n_subjects = len(subjects)
group_design_matrix = pd.DataFrame([1] * n_subjects, columns=['intercept'])
group_dir = os.path.join(data_dir, task, 'derivatives', 'pypreprocess', 'group')
if not os.path.exists(group_dir):
    os.mkdir(group_dir)

from nistats.second_level_model import SecondLevelModel
for contrast_id in contrasts.keys():
    cmap_filenames = [
        os.path.join(data_dir, task, 'derivatives', 'spmpreproc_%s' %
                     session, subject, 'glm', '%s_effects.nii.gz' % contrast_id)
                     for subject in subjects]
    second_level_model = SecondLevelModel(smoothing_fwhm=5).fit(
        cmap_filenames, design_matrix=group_design_matrix)
    z_map = second_level_model.compute_contrast(output_type='z_score')
    thresholded_map, threshold = map_threshold(
        z_map, level=.1, height_control='fdr', cluster_threshold=10)
    z_map.to_filename(os.path.join(group_dir, '%s_z_map.nii.gz'
                      % contrast_id))
    output_file = os.path.join(group_dir, '%s_z_map.png'
                               % contrast_id)
    plotting.plot_stat_map(thresholded_map,
                           title='%s, fdr = .1' % contrast_id,
                           threshold=threshold, output_file=output_file)
    clusters_table = get_clusters_table(z_map, 3., 10)
    clusters_table.to_csv(os.path.join(group_dir, '%s_clusters.csv'
                                       % contrast_id))

# plotting.show()
