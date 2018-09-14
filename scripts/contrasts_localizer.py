""" utility function to create contrasts
"""
import numpy as np


def make_localizer_contrasts(design_matrix):
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
    return contrasts
