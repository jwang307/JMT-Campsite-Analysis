import collections
import os
import csv
import statistics

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


percent_to_class = collections.defaultdict(lambda: 1, {"0-5%": 1,
                                                       "6-25%": 2,
                                                       "26-50%": 3,
                                                       "51-75%": 4,
                                                       "76-95%": 5,
                                                       "96-100%": 5})

use_to_class = collections.defaultdict(lambda: 1, {"Low": 1,
                                                   "Medium": 2,
                                                   "Hard": 3})

condition_to_class = collections.defaultdict(int, {"Class 1": 1,
                                                   "Class 2": 2,
                                                   "Class 3": 3,
                                                   "Class 4": 4,
                                                   "Class 5": 5})


def soil_to_class(soil: int):
    if soil <= 5:
        return 1
    elif soil <= 25:
        return 2
    elif soil <= 50:
        return 3
    elif soil <= 75:
        return 4
    return 5


def area_to_class(area: float):
    if area <= 250:
        return 1
    elif area <= 500:
        return 2
    elif area <= 750:
        return 3
    elif area <= 1000:
        return 4
    return 5


def encode(in_path, out_path, output_dir, features) -> tuple:
    '''

    :param in_path:
    :param out_path:
    :param scaled_path:
    :param output_dir:
    :return:
    '''

    raw_data = pd.read_csv(in_path)
    encoded_data = pd.DataFrame(columns=features)
    composite_scores = []
    for campsite in raw_data.iterrows():
        encoded_campsite = {"vegetation_diff": (percent_to_class[campsite["vegetation_ground_cover_off_site"]] -
                                                percent_to_class[campsite["vegetation_ground_cover_on_site"]]),
                            "grass_diff": (percent_to_class[campsite["grassedge_cover_off_site"]] -
                                           percent_to_class[campsite["grassedge_cover_on_site"]]),
                            "area": (area_to_class(float(campsite["sum_area_feet_squared"]))),
                            "condition_class": (condition_to_class[campsite["condition_class"]]),
                            "exposed_soil": (int(campsite["exposed_soil"])),
                            "dist_water": (int(campsite["distance_to_water"]))}
        composite_scores.append(
            min(round(statistics.mean([percent_to_class[campsite["grassedge_cover_off_site"]] / 1.25,
                                       campsite["distance to water"], campsite["condition class"] / 1.25])), 4))
        encoded_data.append(encoded_campsite)

    encoded_data.to_csv(os.path.join(output_dir, out_path))
    return encoded_data, composite_scores


def preprocessed(in_path, out_path, scaled_path, output_dir, features):
    '''

    :param in_path:
    :param out_path:
    :param scaled_path:
    :param output_dir:
    :param features:
    :return:
    '''
    campsites, scores = encode(in_path, out_path, output_dir, features)

    scaled_data = StandardScaler().fit_transform(campsites)
    scaled_data.to_csv(os.path.join(output_dir, scaled_path))
    return scaled_data, scores
