"""
Attempt to have a full BIDS-based analysis of the hackathon dataset
"""
import os
from nistats.first_level_model import first_level_models_from_bids
task_label = 'localizer'
space_label = 'MNI152NLin2009cAsym'
derivatives_folder = ''
data_dir = os.path.join('/neurospin/tmp/tfmri-hackathon-2018/data/localizer',
                        'derivatives', 'pypreprocess')
models, models_run_imgs, models_events, models_confounds = \
    first_level_models_from_bids(
        data_dir, task_label, space_label, smoothing_fwhm=5.0,
        derivatives_folder=derivatives_folder)
