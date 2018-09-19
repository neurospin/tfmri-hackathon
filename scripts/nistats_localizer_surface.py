"""
Script to process the Localizer data for the Neurospin 2018 hackathon on the
cortical surface.

Author: Bertrand Thirion, 2018
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
from nistats.first_level_model import FirstLevelModel
from nistats.thresholding import fdr_threshold
from nistats.reporting import plot_design_matrix, plot_contrast_matrix
from contrasts_localizer import make_localizer_contrasts

t_r = 2.4
slice_time_ref = 0.5
n_scans = 128

#########################################################################
# Prepare data

data_dir = '/neurospin/tmp/tfmri-hackathon-2018/data'
subjects = ['sub-S%02d' % i for i in range(1, 21)]
session = 'ses-V1'
task = 'localizer'
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
lh_effects = {}
rh_effects = {}
for subject_idx, subject in enumerate(subjects):
    #########################################################################
    # Define data paths
    source_dir = os.path.join(data_dir, task, 'sourcedata', subject, session,
                              'func')
    paradigm_file = os.path.join(source_dir, '%s_%s_task-%s_events.tsv'
                                 % (subject, session, task))
    derivative_dir = os.path.join(data_dir, task, 'derivatives',
                                  'freesurfer_projection_%s' % session, subject)
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
    # load the data
    from nibabel.gifti import read
    from nistats.first_level_model import run_glm
    from nistats.contrasts import compute_contrast
    from nistats.design_matrix import make_design_matrix
    frame_times = t_r * (np.arange(n_scans) + .5)
    design_matrix = make_design_matrix(
        frame_times, paradigm=paradigm, hrf_model='glover + derivative')

    #########################################################################
    # Estimate contrasts
    # ------------------
    # get the design matrix and save it to disk
    design_matrix.to_csv(os.path.join(write_dir, 'design_matrix.csv'))
    ax = plot_design_matrix(design_matrix)
    plt.savefig(os.path.join(write_dir, 'design_matrix.png'))

    # Specify the contrasts
    contrasts = make_localizer_contrasts(design_matrix)
    # plot_contrast_matrix(pd.DataFrame(contrasts), design_matrix)
    # plt.savefig(os.path.join(write_dir, 'contrasts.png'))

    for hemisphere in ['left', 'right']:
        effects = lh_effects
        if hemisphere == 'right':
            effects = rh_effects
        fmri_img = os.path.join(
            derivative_dir, 'wrr%s_%s_task-%s_bold.ico7.s5.%sh.gii' %
            (subject, session, task, hemisphere[0]))
        texture = np.array([
            darrays.data for darrays in read(fmri_img).darrays]).T
        labels, res = run_glm(texture.T, design_matrix.values)
        #######################################################################
        # contrast estimation
        for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
            print('  Contrast % 2i out of %i: %s' %
                  (index + 1, len(contrasts), contrast_id))
            if subject_idx == 0:
                effects[contrast_id] = []

            contrast_ = compute_contrast(labels, res, contrast_val)
            z_map = contrast_.z_score()
            effect = contrast_.effect
            effects[contrast_id].append(effect)
            # Create snapshots of the contrasts
            threshold = fdr_threshold(z_map, alpha=.05)
            out_file = os.path.join(write_dir, '%s_%s_z_map.png' % (
                                    contrast_id, hemisphere))
            plotting.plot_surf_stat_map(
                fsaverage['infl_%s' % hemisphere], z_map, hemi=hemisphere,
                title=contrast_id, colorbar=True, output_file=out_file,
                threshold=threshold, bg_map=fsaverage['sulc_%s' % hemisphere])


import pandas as pd
n_subjects = len(subjects)
group_design_matrix = pd.DataFrame([1] * n_subjects, columns=['intercept'])
group_dir = os.path.join(data_dir, task, 'derivatives',
                         'freesurfer_projection_%s' % session, 'group')
if not os.path.exists(group_dir):
    os.mkdir(group_dir)

from nistats.second_level_model import SecondLevelModel
for contrast_id in contrasts.keys():
    for hemisphere, effects in zip(['left', 'right'],
                                   ['lh_effects', 'right_effects']):
        Y = effects[contrast_id]
        labels, res = run_glm(texture.T, group_design_matrix.values)
        contrast_ = compute_contrast(labels, res, [1])
        z_map = contrast_.z_score()
        threshold = fdr_threshold(z_map, alpha=.05)
        out_file = os.path.join(write_dir, '%s_%s_z_map.png' % (
                                contrast_id, hemisphere))
        plotting.plot_surf_stat_map(
            fsaverage['infl_%s' % hemisphere], z_map, hemi=hemisphere,
            title=contrast_id, colorbar=True, output_file=out_file,
            threshold=threshold, bg_map=fsaverage['sulc_%s' % hemisphere])
