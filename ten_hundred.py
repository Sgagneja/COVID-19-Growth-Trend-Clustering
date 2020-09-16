#!/usr/bin/python3
"""
This file implements a HAC.
@Author: Shaan Gagneja
@Email: sgagneja@wisc.edu
"""

import csv
import numpy as np
from scipy.spatial.distance import euclidean


def load_data(filepath):
    """
    takes in a string with a path to a CSV file and creates a list of dictionaries
    :param filepath: path to csv file
    :return: list of dictionaries with data
    """
    ret = []
    with open(filepath) as csvfile:
        dict_reader = csv.DictReader(csvfile)   # creates list of dictionaries
        for row in dict_reader:
            del row["Lat"]      # removes lat
            del row["Long"]     # removes long
            ret.append(row)
    return ret


def calculate_x_y(time_series):
    """
    calculates the x and y values for a given dictionary from the list of dictionaries
    :param time_series: a specific line (dictionary)
    :return: the x and y values as a tuple
    """
    infections = []
    for key in time_series:     # gets the int values from the dict
        if key != "Province/State" and key != "Country/Region":
            infections.append(int(time_series[key]))

    if infections[len(infections)-1] == 0:  # returns nan, nan if current is = 0
        return np.nan, np.nan

    n = infections[len(infections)-1]
    x = -1
    y = -1
    for i in range(len(infections)-1, -1, -1):  # gets the x (n/10) value
        if infections[i] <= n/10 and infections[i] <= infections[i + 1]:
            x = i
            break

    for j in range(x, -1, -1):  # gets the y (n/100) value and returns nan if no y value found
        if infections[j] <= n/100 and infections[j] <= infections[j + 1]:
            y = j
            break
    if y == -1:
        return len(infections)-1-x, np.nan

    return len(infections)-1-x, x-y


def hac(dataset):
    """
    performs single linkage hierarchical agglomerative clustering on the regions with the (x,y) feature representation
    :param dataset: the preprocessed dataset
    :return: a representation of the clustering
    """
    # remove any invalid data
    data = []
    for tupl in dataset:
        if tupl[0] is not np.nan and tupl[1] is not np.nan:
            data.append(tupl)

    clusters = []  # current clusters
    for i in range(len(data)):
        # adds points, index, size of points list in dictionary
        clusters.append({'points': [data[i]], 'size': 1, 'index': i})

    index = len(clusters)   # start index for new clusters
    z = []
    while len(clusters) != 1:
        x, y, dist = get_next(clusters)     # find index of next clusters to be agglomerated

        # retrieves the clusters if index = x and y respectively
        c1 = next(item for item in clusters if item["index"] == x)
        c2 = next(item for item in clusters if item["index"] == y)

        # merge data from both clusters and append new cluster to z
        new_p = c1['points'] + c2['points']
        new_s = c1['size'] + c2['size']
        z.append([x, y, dist, new_s])

        # add merged cluster and remove the two clusters that were merged
        clusters.append({'points': new_p, 'size': new_s, 'index': index})
        clusters.remove(c1)
        clusters.remove(c2)
        index += 1

    return np.array(z)


def get_next(clusters):
    """
    Determines which clusters should be agglomerated next
    :param clusters: list of clusters
    :return: x, y, (of clusters to be agglomerated), min (single linkage distance between 2 clusters)
    """
    mini = float('inf')
    x = -1
    y = -1

    for i in range(len(clusters)):  # calculate euclidean distances between all points in each cluster
        for j in range(i + 1, len(clusters)):
            c1_points = clusters[i]['points']
            c2_points = clusters[j]['points']
            for p1 in c1_points:
                for p2 in c2_points:
                    dist = euclidean(p1, p2)
                    if dist < mini: # sets values if distance is minimum
                        mini = dist
                        x = min(clusters[i]['index'], clusters[j]['index'])
                        y = max(clusters[i]['index'], clusters[j]['index'])
    return x, y, mini
