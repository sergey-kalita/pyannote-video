# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 18:09:26 2018

@author: sergii.kalyta
"""
import os
from pyannote.video.face.clustering import FaceClustering
#, hdbscan_cluster


import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

from pandas import read_table
import numpy as np
from sklearn.datasets import make_blobs

#from clustering import hdbscan_cluster, dbscan_cluster

import hdbscan


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def hdbscan_cluster(feature, min_cluster_size=5, min_samples=None, metric='euclidean'):
    
    #data = [fi for (ni, nj, fi) in feature]
    data = [fi for (fi) in feature]

    if len(data) < min_cluster_size:
        return np.asarray((0,0)), {}

    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean').fit(data)
    labels = db.labels_
    probabilities = db.probabilities_

    # Number of clusters in labels, INCLUDING NOISE -1
    #n_clusters = len(set(labels))
    unique, counts = np.unique(labels, return_counts=True)

    classes = {}
    for cls in set(labels):
        classes[cls] = []
    
    #for i, (ni, nj, fi) in enumerate(feature):
    for i, (fi) in enumerate(feature):
        label = labels[i]
        proba = probabilities[i]
        #classes[label].append((ni, nj, proba, fi))
        classes[label].append((proba, fi))

    return np.asarray((unique, counts)).T, classes

def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
#    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
#        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
#    X_img = face_recognition.load_image_file(X_img_path)
#   X_face_locations = face_recognition.face_locations(X_img)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



os.chdir('D:/Projects/AvidPoC/SpeakerRecognition/pyannote-video/scripts')



clustering = FaceClustering(threshold=0.6)#get the most representative sample of embeddings
#face_tracks, embeddings = clustering.model.preprocess('../../pyannote-data/Video2-Scene-019.embedding.txt')
#face_tracks, embeddings = clustering.model.preprocess('../../pyannote-data/Video1.embedding.txt')
face_tracks, embeddings = clustering.model.preprocess('../../pyannote-data/TheBigBangTheory.embedding.txt')
#print(embeddings)
embeddings = embeddings[(embeddings['d0']!=0.0) &(embeddings['d63']!=0.0) & (embeddings['d127']!=0.0)]
#.reset_index(drop=True)
#face_tracks, embeddings = clustering.model.preprocess('../../pyannote-data/VideoWithTramp.embedding.txt')
#print(face_tracks.get_timeline())
#print(face_tracks)
#print(embeddings)
features = []
for index, row in embeddings.iterrows():
    #print(row.name, row["start_time"], row["end_time"])
    #segments[row.name] = Segment(row["start_time"],row["end_time"])
    #segments[Segment(row["start_time"],row["end_time"])] = row.name
    #print( [ row["time"],row["track"],row["time"])
    feature = []
    for i in range(128):
        feature.append(row['d{0}'.format(i)])
    #features.append([ row["time"],row["track"],row["d0"] ])
    features.append(feature)
    #print(tracking[int(row.id["median"])])


#print(features)
#print([fi for (ni, nj, fi) in features])


#data, _ = make_blobs(1000)

#print(data)
#clusterer = hdbscan.HDBSCAN(min_cluster_size=10,min_samples=5)
#cluster_labels = clusterer.fit_predict(features)
#print(cluster_labels)

#db = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2, metric='euclidean').fit(features)
#labels = db.labels_
#probabilities = db.probabilities_
#print(db.labels_)
#print(db.probabilities_)
    
#print("\nHere\n")
#mtrx, cluster_cls = hdbscan_cluster(features, min_cluster_size=4,min_samples=2, metric='euclidean')
#print("Here")
#print(mtrx)
#print(cluster_cls)

#features = self._extractFramesToFeatures(video, index_spread)
##frames = self._extractFrames(video, index_spread)
##aligned = self._detectFaces(frames, debug_path)
##features = self._extractFeatures(aligned)
#video.release()
#gc.collect()

#_, cluster_cls = hdbscan_cluster(features, min_cluster_size=1, min_samples=1, metric='euclidean')
#_, cluster_cls = hdbscan_cluster(features, min_cluster_size=1, min_samples=1, metric='euclidean')

##print(cluster_cls)
#if len(cluster_cls) <= 1: # Only -1 noise cluster found
#    print 'DBSCAN FALLBACK'
#    _, cluster_cls = dbscan_cluster(features, eps=0.7, min_samples=self.min_cluster_size, metric='euclidean')


## discard -1 label noise cluster
#cluster_cls.pop(-1, None)
#
#classes = []
#for cls in cluster_cls:
#    feature = cluster_cls[cls]
#    if len(feature) >= self.actual_min_cluster_size:
#        # Duration estimation using frame numbers and framerate
#        timestamps = [ti for (ti, tj, p, f) in feature]
#        dists = [b-a for a, b in self._pairwise(timestamps)]
#        md = np.mean(dists)
#        duration = sum([i for i in dists if i<=md]) + len(timestamps) # If higher skips in frames are present, we try to eliminate these with mean comparison
#        duration = float(duration / frame_rate)
#
#        # Max feature cut off with sorted proba
#        feature.sort(key=lambda tup: tup[2], reverse=True)
#        feature = feature[:self.max_feature_per_class]
#        feature = [f for (ni, nj, p, f) in feature]
#        feature = np.array(feature, copy=False)
#        classes.append((duration, feature))
                        
                        
                        

result = clustering(face_tracks, features=embeddings)
from pyannote.core import notebook, Segment
notebook.reset()
notebook.crop = Segment(0, 30)
print(result)
###cropping
tracking = '../../pyannote-data/TheBigBangTheory.track.txt'
names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
dtype = {'left': np.float32, 'top': np.float32,
         'right': np.float32, 'bottom': np.float32}
tracking = read_table(tracking, delim_whitespace=True, header=None,
                      names=names, dtype=dtype)
tracking = tracking[(tracking['status']=="detection")]
print(tracking)

#embeddings.index.rename('id', inplace=True)
embeddings.index.name='id'
embeddings['id']=embeddings.index
#embeddings= embeddings.reset_index(drop=True)
print(embeddings)
embeddings = embeddings.sort_values('time')
embeddings = embeddings.groupby(['track']).aggregate({"id": [np.median]})
print (embeddings)
#print (embeddings[1])
for index, row in embeddings.iterrows():
    #print(row.name, row["start_time"], row["end_time"])
    #segments[row.name] = Segment(row["start_time"],row["end_time"])
    #segments[Segment(row["start_time"],row["end_time"])] = row.name
    print( row.name, int(row.id["median"]))
    #print(tracking[int(row.id["median"])])

###
#prediction 1
#tracking = '../../pyannote-data/TheBigBangTheory.track.txt'
#names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
#dtype = {'left': np.float32, 'top': np.float32,
#         'right': np.float32, 'bottom': np.float32}
#tracking = read_table(tracking, delim_whitespace=True, header=None,
#                      names=names, dtype=dtype)
#tracking = tracking[(tracking['status']=="detection")]
#print(tracking)
#
##embeddings.index.rename('id', inplace=True)
#embeddings.index.name='id'
#embeddings['id']=embeddings.index
##embeddings= embeddings.reset_index(drop=True)
#print(embeddings)
#embeddings = embeddings.sort_values('time')
#embeddings = embeddings.groupby(['track']).aggregate({"id": [np.median]})
#print (embeddings)
##print (embeddings[1])
#for index, row in embeddings.iterrows():
#    #print(row.name, row["start_time"], row["end_time"])
#    #segments[row.name] = Segment(row["start_time"],row["end_time"])
#    #segments[Segment(row["start_time"],row["end_time"])] = row.name
#    print( row.name, int(row.id["median"]))
#    print(tracking[int(row.id["median"])])

#####################
#print(embeddings.columns.get_level_values(0))
#print(embeddings.xs('track', axis=0, drop_level=True))
    

#for image_file in os.listdir("knn_examples/test"):
#    full_file_path = os.path.join("knn_examples/test", image_file)
#
#    print("Looking for faces in {}".format(image_file))
#
#    # Find all people in the image using a trained classifier model
#    # Note: You can pass in either a classifier file name or a classifier model instance
#    predictions = predict(full_file_path, model_path="trained_knn_model.clf")
#
#    # Print results on the console
#    for name, (top, right, bottom, left) in predictions:
#        print("- Found {} at ({}, {})".format(name, left, top))
#        

#Kady Simpson, Ellen Moroh"

#Kady Simpson, Donald Trump"
#TV Anchor
#"Donald Trump, Justin Trudo, Unknown"

#mapping = {12: 'Leonard', 14: 'Sheldon', 13: 'Receptionist', 5: 'False_alarm'}
#mapping = {4: 'Leonard', 6: 'Sheldon', 12: 'Receptionist', 5: 'False_alarm'}
#mapping = {11: 'Leonard', 13: 'Sheldon', 9: 'Receptionist', 4: 'False_alarm'}
#mapping = {1:'Somebody'}
#result = result.rename_labels(mapping=mapping)
#print(result)
#with open('../../pyannote-data/Video1.labels.txt', 'w') as fp:
with open('../../pyannote-data/TheBigBangTheory.labels.txt', 'w') as fp:
#with open('../../pyannote-data/Video2-Scene-019.labels.txt', 'w') as fp:
    for _, track_id, cluster in result.itertracks(yield_label=True):
        fp.write('%s %s\n'%(track_id,cluster))       
        print (track_id,cluster)
#fp.write(f'{track_id} {cluster}\n')