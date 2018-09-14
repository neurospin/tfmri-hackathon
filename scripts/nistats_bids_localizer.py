"""
Attempt to have a full BIDS-based analysis of the hackathon dataset
"""
import os
from nistats.first_level_model import first_level_models_from_bids

# first do some renaming -> symlinks
subjects = ['S%02d' % i for i in range(1, 21)]
data_dir = '/neurospin/tmp/tfmri-hackathon-2018/data'
for subject in subjects:
    src = os.path.join(data_dir, 'localizer', 'sourcedata', 'sub-' + subject, 'ses-V1',
                       'func', 'sub-%s_ses-V1_task-localizer_events.tsv'
                       % subject)
    dst = os.path.join(
        data_dir, 'localizer', 'derivatives', 'pypreprocess', 'sub-' + subject, 'ses-V1',
        'func', 'sub-%s_ses-V1_task-localizer_events.tsv' % subject)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    print(dst)
    src = os.path.join(
        data_dir, 'localizer', 'derivatives', 'pypreprocess', 'sub-' + subject, 'ses-V1',
        'func', 'sub-%s_ses-V1_task-localizer_space-MNI_bold.nii.gz' % subject)
    dst = os.path.join(
        data_dir, 'localizer', 'derivatives', 'pypreprocess', 'sub-' + subject, 'ses-V1',
        'func', 'sub-%s_ses-V1_task-localizer_space-MNI_bold_preproc.nii.gz' % subject)
    if not os.path.exists(dst):
        os.symlink(src, dst)

task_label = 'localizer'
space_label = 'MNI'
derivatives_folder = ''
data_dir = os.path.join('/neurospin/tmp/tfmri-hackathon-2018/data/localizer',
                        'derivatives', 'pypreprocess')
models, models_run_imgs, models_events, models_confounds = \
    first_level_models_from_bids(
        data_dir, task_label, space_label, smoothing_fwhm=5.0,
        derivatives_folder=derivatives_folder)

for (model, img, events) in zip(models, models_run_imgs, models_events):
    model.t_r = 2.0
    model.fit(img, events=events, )
