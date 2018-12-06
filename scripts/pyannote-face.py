#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""Face detection and tracking

The standard pipeline is the following

      face tracking => feature extraction => face clustering

Usage:
  pyannote-face track [options] <video> <shot.json> <tracking>
  pyannote-face extract [options] <video> <tracking> <landmark_model> <embedding_model> <landmarks> <embeddings>
  pyannote-face demo [options] <video> <tracking> <output>
  pyannote-face train [options] <train_dir> <landmark_model> <embedding_model> <train_model>
  pyannote-face recognize [options] <embeddings> <train_model> <labels>
  pyannote-face annotate [options] <shot.json> <tracking> <output>
  pyannote-face evaluate [options] <reference.json> <validation.json>
  pyannote-face (-h | --help)
  pyannote-face --version

General options:

  --ffmpeg=<ffmpeg>         Specify which `ffmpeg` to use.
  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show processing progress.
  --report=<path>           Path to report result file.

Face tracking options (track):

  <video>                   Path to video file.
  <shot.json>               Path to shot segmentation result file.
  <tracking>                Path to tracking result file.

  --min-size=<ratio>        Approximate size (in video height ratio) of the
                            smallest face that should be detected. Default is
                            to try and detect any object [default: 0.0].
  --every=<seconds>         Only apply detection every <seconds> seconds.
                            Default is to process every frame [default: 0.0].
  --min-overlap=<ratio>     Associates face with tracker if overlap is greater
                            than <ratio> [default: 0.5].
  --min-confidence=<float>  Reset trackers with confidence lower than <float>
                            [default: 10.].
  --max-gap=<float>         Bridge gaps with duration shorter than <float>
                            [default: 1.].

Feature extraction options (features):

  <video>                   Path to video file.
  <tracking>                Path to tracking result file.
  <landmark_model>          Path to dlib facial landmark detection model.
  <embedding_model>         Path to dlib feature extraction model.
  <landmarks>               Path to facial landmarks detection result file.
  <embeddings>              Path to feature extraction result file.

Visualization options (demo):

  <video>                   Path to video file.
  <tracking>                Path to tracking result file.
  <output>                  Path to demo video file.

  --height=<pixels>         Height of demo video file [default: 400].
  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --shift=<sec>             Shift result files by <sec> seconds [default: 0].
  --landmark=<path>         Path to facial landmarks detection result file.
  --label=<path>            Path to track identification result file.

Annotation options (train):

  <train_dir>               Path to training set.
  <landmark_model>          Path to dlib facial landmark detection model.
  <embedding_model>         Path to dlib feature extraction model.
  <train_model>                   Path to model result file.
  
Annotation options (recognize):

  <embeddings>              Path to feature extraction input file.
  <train_model>             Path to trained model file.
  <labels>                  Path to track identification result file.

Annotation options (annotate):

  <shot.json>               Path to shot segmentation result file.
  <tracking>                Path to tracking result file.
  <output>                  Path to annotate file.

  --from=<sec>              Encode demo from <sec> seconds [default: 0].
  --until=<sec>             Encode demo until <sec> seconds.
  --shift=<sec>             Shift result files by <sec> seconds [default: 0].
  
Evaluation options (evaluate):

  <reference.json>             Path to shot segmentation result file.
  <validation.json>            Path to tracking result file.


"""

from __future__ import division

from docopt import docopt

from pyannote.core import Annotation
import pyannote.core.json

from pyannote.video import __version__
from pyannote.video import Video
from pyannote.video import Face
from pyannote.video import FaceTracking
from pyannote.core import Segment
from pyannote.video.face.clustering import FaceClustering


from pandas import read_table
import pandas as pd
import simplejson as json

from six.moves import zip
import numpy as np
import cv2
import operator
import datetime
import time
import math
import dlib
import os
import pickle
import functools as ft

from PIL import Image, ImageDraw
from skimage import io, exposure

from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

#from pyannote.core import Annotation, Segment
#from pyannote.metrics.identification import IdentificationPrecision    

MIN_OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 10.
MAX_GAP = 1.

FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                 '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f} '
                 '{status:s}\n')


def getFaceGenerator(tracking, frame_width, frame_height, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # load tracking file and sort it by timestamp
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    #tracking = tracking[(tracking['status']=="detection")].reset_index(drop=True)
    tracking = tracking.sort_values('t')

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    faces = []
    currentT = None

    for _, (T, identifier, left, top, right, bottom, status) in tracking.iterrows():

        left = int(left * frame_width)
        right = int(right * frame_width)
        top = int(top * frame_height)
        bottom = int(bottom * frame_height)

        face = rectangle(left, top, right, bottom)

        # load all faces from current frame and only those faces
        if T == currentT or currentT is None:
            faces.append((identifier, face, status))
            currentT = T
            continue

        # once all faces at current time are loaded
        # wait until t reaches current time
        # then returns all faces at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all faces at once
            t = yield currentT, faces

            # reset current time and corresponding faces
            faces = [(identifier, face, status)]
            currentT = T
            break

    while True:
        t = yield t, []


def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def getLandmarkGenerator(shape, frame_width, frame_height):
    """Parse precomputed shape file and generate timestamped shapes"""

    # load landmarks file
    shape = read_table(shape, delim_whitespace=True, header=None)

    # deduce number of landmarks from file dimension
    _, d = shape.shape
    n_points = (d - 2) / 2

    # t is the time sent by the frame generator
    t = yield

    shapes = []
    currentT = None

    for _, row in shape.iterrows():

        T = float(row[0])
        identifier = int(row[1])
        landmarks = np.float32(list(pairwise(
            [coordinate for coordinate in row[2:]])))
        landmarks[:, 0] = np.round(landmarks[:, 0] * frame_width)
        landmarks[:, 1] = np.round(landmarks[:, 1] * frame_height)

        # load all shapes from current frame
        # and only those shapes
        if T == currentT or currentT is None:
            shapes.append((identifier, landmarks))
            currentT = T
            continue

        # once all shapes at current time are loaded
        # wait until t reaches current time
        # then returns all shapes at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all shapes at once
            t = yield currentT, shapes

            # reset current time and corresponding shapes
            shapes = [(identifier, landmarks)]
            currentT = T
            break

    while True:
        t = yield t, []


def track(video, shot, output,
          detect_min_size=0.0,
          detect_every=0.0,
          track_min_overlap_ratio=MIN_OVERLAP_RATIO,
          track_min_confidence=MIN_CONFIDENCE,
          track_max_gap=MAX_GAP):
    """Tracking by detection"""

    tracking = FaceTracking(detect_min_size=detect_min_size,
                            detect_every=detect_every,
                            track_min_overlap_ratio=track_min_overlap_ratio,
                            track_min_confidence=track_min_confidence,
                            track_max_gap=track_max_gap)

    with open(shot, 'r') as fp:
        shot = pyannote.core.json.load(fp)

    if isinstance(shot, Annotation):
        shot = shot.get_timeline()

    with open(output, 'w') as foutput:

        for identifier, track in enumerate(tracking(video, shot)):

            for t, (left, top, right, bottom), status in track:

                foutput.write(FACE_TEMPLATE.format(
                    t=t, identifier=identifier, status=status,
                    left=left, right=right, top=top, bottom=bottom))

            foutput.flush()

def extract(video, landmark_model, embedding_model, tracking, landmark_output, embedding_output):
    """Facial features detection"""
    faces_folder = "./Video2Faces/"
    # face generator
    frame_width, frame_height = video.frame_size
    faceGenerator = getFaceGenerator(tracking,
                                     frame_width, frame_height,
                                     double=False)
    
    
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    tracking = tracking[(tracking['status'].str.contains("detection"))]
    #tracking = tracking.sort_values('t')
    print(tracking)    
    tracking_ids = set(tracking.index.values.tolist())
    #list(tracking.select_dtypes(include=['bool']).columns)
    
    #tracking_ts = pd.DataFrame(tracking, columns=['t','track'])
    tracking_ts = (tracking[['t','track']])
    #for t, track in tracking_ts .iteritems():
     #   print(t, track)    
    times_and_tracks = []
    for index, row in tracking_ts.iterrows():
        times_and_tracks.append((row["t"],int(row["track"])))
    
    detected_faces = set(times_and_tracks)
    print(times_and_tracks)
    faceGenerator.send(None)

    face = Face(landmarks=landmark_model,
                embedding=embedding_model)
    print(face)
    embed_id = 0
    with open(landmark_output, 'w') as flandmark, \
         open(embedding_output, 'w') as fembedding:

        for timestamp, rgb in video:
            #print(timestamp)
            #if timestamp in tracking_ts:
            #print(timestamp)
            # get all detected faces at this time
            T, faces = faceGenerator.send(timestamp)
            # not that T might be differ slightly from t
            # due to different steps in frame iteration
            #

            for identifier, bounding_box, _ in faces:
                
                #if True:
                #print(identifier)
                if (T,identifier) in detected_faces: 
                #if True:
                #embed_id in tracking_ids: 
                    #print((T,identifier))
                    landmarks = face.get_landmarks(rgb, bounding_box)
                    embedding = face.get_embedding(rgb, landmarks)
                    
                    #face_in_rgb = rgb.copy()
                    #for p in landmarks.parts():
                     #   x, y = p.x, p.y
                     #   cv2.rectangle(copy, (x, y), (x, y), (0, 255, 0), 2)
                    if len(faces_folder) > 0:
                        #print (bounding_box)
                        #print("rgb.shape:",rgb.shape)
                        (height,width, _) = rgb.shape
#                        face_in_rgb = rgb[max(bounding_box.top() - int(bounding_box.height()*0.8),0):min(bounding_box.bottom() + int(bounding_box.height()*0.8),rgb.shape[0]-1),
#                                          max(bounding_box.left() - int(bounding_box.width()*0.8),0):min(bounding_box.right() + int(bounding_box.width()*0.8),rgb.shape[1]-1)]
                        face_in_rgb = rgb[max(bounding_box.top(),0):min(bounding_box.bottom() ,height - 1),
                                          max(bounding_box.left() ,0):min(bounding_box.right(),width -1)]
                        
#                        #copy = cv2.resize(copy, (self.size, self.size))
                        #return copy
        
                        io.imsave(faces_folder + '{identifier:2d}_{t:.3f}_img.jpg'.format(t=T, identifier=identifier), face_in_rgb)
    
                    flandmark.write('{t:.3f} {identifier:d}'.format(
                        t=T, identifier=identifier))
                    for p in landmarks.parts():
                        x, y = p.x, p.y
                        flandmark.write(' {x:.5f} {y:.5f}'.format(x=x / frame_width,
                                                                y=y / frame_height))
                    flandmark.write('\n')
        
                    fembedding.write('{t:.3f} {identifier:d}'.format(
                        t=T, identifier=identifier))
                    for x in embedding:
                        fembedding.write(' {x:.5f}'.format(x=x))
                    fembedding.write('\n')

                else:
                    flandmark.write('{t:.3f} {identifier:d}'.format(
                        t=T, identifier=identifier))
                    landmarks = [(0.0,0.0)]*68
                    for p in landmarks:
                        x, y = p
                        flandmark.write(' {x:.5f} {y:.5f}'.format(x=x / frame_width,
                                                                y=y / frame_height))
                    flandmark.write('\n')
        
                    fembedding.write('{t:.3f} {identifier:d}'.format(
                        t=T, identifier=identifier))
                    embedding = [0.0]*128
                    for x in embedding:
                        fembedding.write(' {x:.5f}'.format(x=x))
                    fembedding.write('\n')
                embed_id+=1
                    
            flandmark.flush()
            fembedding.flush()


def get_make_frame(video, tracking, landmark=None, labels=None,
                   height=200, shift=0.0):

    COLORS = [
        (240, 163, 255), (  0, 117, 220), (153,  63,   0), ( 76,   0,  92),
        ( 25,  25,  25), (  0,  92,  49), ( 43, 206,  72), (255, 204, 153),
        (128, 128, 128), (148, 255, 181), (143, 124,   0), (157, 204,   0),
        (194,   0, 136), (  0,  51, 128), (255, 164,   5), (255, 168, 187),
        ( 66, 102,   0), (255,   0,  16), ( 94, 241, 242), (  0, 153, 143),
        (224, 255, 102), (116,  10, 255), (153,   0,   0), (255, 255, 128),
        (255, 255,   0), (255,  80,   5)
    ]

    video_width, video_height = video.size
    ratio = height / video_height
    width = int(ratio * video_width)
    video.frame_size = (width, height)

    faceGenerator = getFaceGenerator(tracking, width, height, double=True)
    faceGenerator.send(None)

    if landmark:
        landmarkGenerator = getLandmarkGenerator(landmark, width, height)
        landmarkGenerator.send(None)

    if labels is None:
        labels = dict()

    def make_frame(t):

        frame = video(t)
        _, faces = faceGenerator.send(t - shift)

        if landmark:
            _, landmarks = landmarkGenerator.send(t - shift)

        cv2.putText(frame, '{t:.3f}'.format(t=t), (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8, False)
        for i, (identifier, face, _) in enumerate(faces):
            color = COLORS[identifier % len(COLORS)]

            # Draw face bounding box
            pt1 = (int(face.left()), int(face.top()))
            pt2 = (int(face.right()), int(face.bottom()))
            cv2.rectangle(frame, pt1, pt2, color, 2)

            # Print tracker identifier
            cv2.putText(frame, '#{identifier:d}'.format(identifier=identifier),
                        (pt1[0], pt2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, 8, False)

            # Print track label
            label = labels.get(identifier, '')
            cv2.putText(frame,
                        '{label:s}'.format(label=label),
                        (pt1[0], pt1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, 8, False)

            # Draw nose
            if landmark:
                points = landmarks[i][0].parts()
                pt1 = (int(points[27, 0]), int(points[27, 1]))
                pt2 = (int(points[33, 0]), int(points[33, 1]))
                cv2.line(frame, pt1, pt2, color, 1)

        return frame

    return make_frame


def demo(filename, tracking, output, t_start=0., t_end=None, shift=0.,
         labels=None, landmark=None, height=200, ffmpeg=None):

    # parse label file
    if labels is not None:
        with open(labels, 'r') as f:
            labels = {}
            for line in f:
#                identifier, label = line.strip().split()
#                identifier = int(identifier)
#                labels[identifier] = label
                pos = line.strip().index(' ')
                identifier = line.strip()[0:pos]
                label  = line.strip()[pos+1:]
                identifier = int(identifier)
                labels[identifier] = label
                

    video = Video(filename, ffmpeg=ffmpeg)

    from moviepy.editor import VideoClip, AudioFileClip

    make_frame = get_make_frame(video, tracking, landmark=landmark,
                                labels=labels, height=height, shift=shift)
    video_clip = VideoClip(make_frame, duration=video.duration)
    audio_clip = AudioFileClip(filename)
    clip = video_clip.set_audio(audio_clip)

    if t_end is None:
        t_end = video.duration

    clip.subclip(t_start, t_end).write_videofile(output, fps=video.frame_rate)

def train(train_dir, model_save_path, landmark_model, embedding_model, verbose=False):

    knn_algo='ball_tree'
    n_neighbors=10
    verbose=True
    X = []
    Y = []

    face = Face(landmarks=landmark_model, embedding=embedding_model)
        
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            
            image = face_recognition.load_image_file(img_path)
#            faces = []
#            for f in face.iterfaces(image):
#                faces.append(f)
#            
#            #io.imsave('1_img.jpg', image)
#        
#            face_bounding_boxes = faces
            face_bounding_boxes = [dlib.rectangle(0,0,image.shape[1]-1, image.shape[0]-1)]

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                if verbose:
                    print("Image {} is processing".format(img_path))
                landmarks = face.get_landmarks(image, face_bounding_boxes[0])
                embedding = face.get_embedding(image, landmarks)                
                X.append(list(embedding) )
                Y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, Y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

def predict(X_emb, knn_clf=None, model_path=None, distance_threshold=0.5):
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
    #X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    #if len(X_face_locations) == 0:
        #return []

    # Find encodings for faces in the test iamge
    #faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    faces_encodings=[ X_emb ]
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=10)
    print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(faces_encodings))]

    # Predict classes and remove classifications that aren't within the threshold
    #return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    return [(pred) if rec else ("Unknown") for pred, rec in zip(knn_clf.predict(faces_encodings), are_matches)]
    
def recognize(embeddings, train_model, labels, verbose=False, report = None):

    from pyannote.core import Annotation, Segment
    from pyannote.metrics.identification import IdentificationPrecision    
    from pyannote.metrics.identification import IdentificationRecall
    try:
        with open(train_model, 'rb') as f:
            train_model = pickle.load(f)
    except:       
         train_model = None   
    if verbose:
        print("Clustering...")
    clustering = FaceClustering(threshold=0.55)
    face_tracks, embeddings = clustering.model.preprocess(embeddings)
    embeddings = embeddings[ft.reduce(lambda x, y: x & y,[embeddings['d{0}'.format(i)] != 0.0 for i in range(128)])]
    result = clustering(face_tracks, features=embeddings)
    print(result)
    if verbose:
        print("Recognizing...")
        
    clusters = {}   
    for seg, track_id, cluster in result.itertracks(yield_label=True):
        clusters[track_id] = cluster 
        #print(seg)
        
    recogn_votes = dict({})
    for index, row in embeddings.iterrows():
        embd = [ row['d{0}'.format(i)] for i in range(128) ]
        try:
            predictions = predict(embd, knn_clf=train_model)
        except:
            predictions = [("Unknown")]
        
        #print(predictions)
        for name in predictions:
           if clusters[row["track"]] in recogn_votes.keys():
               if name in recogn_votes[clusters[row["track"]]].keys():
                   recogn_votes[clusters[row["track"]]][name] +=1
               else:
                   recogn_votes[clusters[row["track"]]][name] = 1
           else:
               recogn_votes[clusters[row["track"]]] = { name : 1 }
    if verbose:           
        print("Clusters recogniton with votes: ", recogn_votes)       
    #sorted_segments = {'L':11, 'K':32}
    #sorted_segments = dict(sorted(sorted_segments.items(), key=operator.itemgetter(1), reverse=True))
    #print(sorted_segments)
    #print(result)
    #with open('../../pyannote-data/TheBigBangTheory.labels.txt', 'w') as fp:
    annotation = Annotation()
    with open(labels, 'w') as fp:
        for seg, track_id, cluster in result.itertracks(yield_label=True):
            recogn_pair = sorted(recogn_votes[cluster].items(), key=operator.itemgetter(1), reverse=True)[0]
            if recogn_pair[0] is not "Unknown":
                fp.write('%s %s\n'%(track_id,recogn_pair[0]))       
                print (track_id,'{name}'.format(name = recogn_pair[0]))
                annotation[seg, track_id] = recogn_pair[0]
            else:
                fp.write('%s %s %d\n'%(track_id,recogn_pair[0],cluster))       
                print (track_id,'{name} {id:d}'.format(name = recogn_pair[0], id = cluster))
                annotation[seg, track_id] = '{name} {id:d}'.format(name = recogn_pair[0], id = cluster)
            print(seg)
    #print(annotation)
    #from pyannote.metrics.identification import IdentificationPrecision    
#    identificationPrecisions = IdentificationPrecision()
#    precision = identificationPrecisions(annotation, annotation2)
#    
#    print("prec ",precision," %")
#    identificationRecall = IdentificationRecall()
#    recall = identificationRecall(annotation, annotation2)
#    
#    print("recall ",recall," %")
#    json_data = []
#    annotation = Annotation()
#    for seg, track_id, cluster in result.itertracks(yield_label=True):
#        #clusters[track_id] = cluster 
#        print(seg.start, seg.end)
#        annotation[seg] = 
#        json_data.append({"range":  seg, 
#                          "track":  track_id,
#                          "cluster":cluster,
#        })
        
#    json_data = []
#    for shot_seg, face_labels in shots_with_face_ids:
#        print (shot_seg, face_labels)
#        json_data.append({"time":"%02d:%02d" % divmod(math.ceil(shot_seg.start), 60), 
#                          "Speaker":"unknown",
#                          "FrameFaces": ','.join(sorted(set(face_labels))),
#                          "Topics": [{
#                        			"Unknown": "100"
#                          }]
#        })
#        
    if report:#pyannote.core.json.dump(annotation, fp)
        with open(report, 'w') as fp:
            pyannote.core.json.dump(annotation, fp)
        #json.dump(annotation, fp)
        
    #with open("./test.json", 'r') as fp:
        #shots = pyannote.core.json.load(fp)

    #if isinstance(shots, Annotation):
        #pass#shots = shots.get_timeline()
    #print(shots)
        
    
def evaluate(reference, hypothesis, report = None):

    from pyannote.core import Annotation, Segment
    from pyannote.metrics.identification import IdentificationPrecision    
    from pyannote.metrics.identification import IdentificationRecall
    from pyannote.metrics.identification import IdentificationErrorRate
    from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
    
#    from pyannote.metrics.diarization import DiarizationErrorRate
#    from pyannote.metrics.diarization import DiarizationPurity
#    from pyannote.metrics.diarization import DiarizationCoverage
    
    with open(reference, 'r') as fp:
        reference = pyannote.core.json.load(fp)
        
    with open(hypothesis, 'r') as fp:
        hypothesis = pyannote.core.json.load(fp)

#    diarizationErrorRate = DiarizationErrorRate()
#    print("DER = {0:.3f}".format(diarizationErrorRate(reference, hypothesis, uem=Segment(0, 40))))
    
#    mapping = diarizationErrorRate.optimal_mapping(reference, hypothesis)
#    print("Mapping = ", mapping)
    
    #detailErr = diarizationErrorRate(reference, hypothesis, detailed=True)
    #print("Detail err = {0}".format(detailErr))

#    purity = DiarizationPurity()
#    print("Purity = {0:.3f}".format(purity(reference, hypothesis, uem=Segment(0, 40))))

#    coverage = DiarizationCoverage()
#    print("Coverage = {0:.3f}".format(coverage(reference, hypothesis, uem=Segment(0, 40))))
   
    identificationPrecisions = IdentificationPrecision(collar = 0.0, skip_overlap = False)
    precision = identificationPrecisions(reference, hypothesis)
    
    identificationRecall = IdentificationRecall(collar = 0.0, skip_overlap = False)
    recall = identificationRecall(reference, hypothesis)
    
     
    identificationErrorRate = IdentificationErrorRate(collar = 0.0, skip_overlap = False)
    errorRate = identificationErrorRate(reference, hypothesis)
    errorDetail = identificationErrorRate.compute_components(reference, hypothesis)
    
    identificationErrorAnalysis = IdentificationErrorAnalysis(collar = 0.0, skip_overlap = False)
    errorAnalysis = identificationErrorAnalysis.difference(reference, hypothesis, uem=None, uemified=False)
#    print(diff)
    
    #matrix = identificationErrorAnalysis.matrix(reference, hypothesis, uem=None)
    #print(matrix)

    if report:
        with open(report, 'w') as fp:
             fp.write('Precision = %f\n'%(precision))       
             fp.write('Recall = %f\n'%(recall))       
             fp.write("IER = {0:f}\n".format(errorRate))
             fp.write("Error details: %s\n"%(errorDetail))
             fp.write("Error analysis:\n%s"%(errorAnalysis))
    else:
        print('Precision = %f'%(precision))       
        print('Recall = %f'%(recall))       
        print("IER = {0:f}".format(errorRate))
        print("Error details: %s"%(errorDetail))
        print("Error analysis:\n%s"%(errorAnalysis))
        
        
def annotate(output, tracking, shots,t_start=0., t_end=None, shift=0., labels=None):
    if not shots:
        
        return None
    with open(shots, 'r') as fp:
        shots = pyannote.core.json.load(fp)

    if isinstance(shots, Annotation):
        shots = shots.get_timeline()

    with open(output, 'w') as foutput:

        for shot in shots:
            print (shot)
            print (type(shot))
#            for t, (left, top, right, bottom), status in track:
#
#                foutput.write(FACE_TEMPLATE.format(
#                    t=t, identifier=identifier, status=status,
#                    left=left, right=right, top=top, bottom=bottom))
#
#            foutput.flush()


    # parse label file
    if labels is not None:
        with open(labels, 'r') as f:
            labels = {}
            for line in f:
                #print(line.strip().split())
                pos = line.strip().index(' ')
                #l = line.strip().split()
                identifier = line.strip()[ 0:pos ]
                label  = line.strip()[ pos+1: ]
                #print(identifier, label  )
                #print( identifier  )
                identifier = int(identifier)
                labels[identifier] = label
#        for id,label in labels.items():
#            print(id,label)
#            
      ##################
        
#    names = ['time', 'track']
#    for i in range(128):
#        names += ['d{0}'.format(i)]
#    data = read_table(tracking, delim_whitespace=True,
#                      header=None, names=names)
#    data.sort_values(by=['track', 'time'], inplace=True)
#    starting_point = Annotation(modality='face')
#     #Segment(np.min(group.time), np.max(group.time))
#    for track, segment in data.groupby('track').apply(self._to_segment).iteritems():
#        if not segment:
#            continue
#        starting_point[segment, track] = track
            
            
            
            ######################
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    
#    print(tracking)
#    tracking = tracking[(tracking['status']=="detection")].reset_index(drop=True)
#    print(tracking)
    #return
    #tracking = tracking.sort_values('t')
    #print(tracking)
    tracking = tracking.sort_values('track')
    #print(tracking)
#    aggregations = {
#    'end': max,
#    'start':min
#    }
    #print(tracking.groupby('track')['t'].aggregate(min))
    #print(tracking.groupby('track')['t'].aggregate(aggregations))
    #print(tracking.groupby('track').aggregate({'t': [min, max]}))
    grouped = tracking.groupby('track').aggregate({'t': [min, max]})
    grouped.columns = grouped.columns.droplevel(level=0)
    grouped = grouped.rename(columns={"min": "start_time", "max": "end_time"})
    print(grouped)
    #print(grouped.columns.get_level_values(0))
    #print(grouped.xs('track', axis=2, drop_level=True))
    segments = dict()
#    agg_segments = list()
    if labels is None:
        labels = {}
    for index, row in grouped.iterrows():
        #print(row.name, row["start_time"], row["end_time"])
        #segments[row.name] = Segment(row["start_time"],row["end_time"])
        #segments[Segment(row["start_time"],row["end_time"])] = row.name
        segments[row.name] = Segment(row["start_time"],row["end_time"])
        #if None
        #labels[int(row.name)] = str(row.name)
#        agg_segments.append(row["start_time"])
#        agg_segments.append(row["end_time"])
     
    #print(face_tracks.get_timeline())
    
    #agg_segments = agg_segments.sorted()
    #segments = list(segments)
#    for key, value in segments.items():
#        print(key,value)
#
#    sorted_segments = dict(sorted(segments.items(), key=operator.itemgetter(1), reverse=False))
#    #print(dict(sorted_segments) )
#    segments_of_scenes = list()
#    for key, value in sorted_segments.items():
#        print(key,value)
#        segments_of_scenes.append((sorted_segments[key],key))
#        
    #!print(segments )
    
    frame_times = set(tracking["t"].tolist())
    frame_times_dict = dict()
    #frame_times_dict[round(6.36,2)] = 1
#    print (frame_times)
    for t in frame_times:
        for face_id, segment in segments.items():
            if segment.overlaps(t):
                if t in frame_times_dict.keys():
                    frame_times_dict[t].append(face_id)
                else:
                    frame_times_dict[t]=[face_id]
                
    #print(frame_times_dict)
    frame_times_dict = dict(sorted(frame_times_dict.items(), key=operator.itemgetter(1), reverse=False))
    
    shots_with_face_ids = dict()
    for shot in shots:
        for track_id, track_segment in segments.items():
            if shot.intersects(track_segment):
                if shot in shots_with_face_ids.keys():
                    shots_with_face_ids[shot].append(labels[track_id])
                else:
                    shots_with_face_ids[shot] = [labels[track_id]]

    
    shots_with_face_ids= sorted(shots_with_face_ids.items(), key=operator.itemgetter(0), reverse=False)
    
    print(shots_with_face_ids)
#    	{
#		"time": "01:35",
#		"Speaker": "Kady Simpson",
#		"FrameFaces": "Kady Simpson",
#		"Topics": [{
#			"Love": "57",
#			"Trade": "62"
#		}]
#	},

    json_data = []
    for shot_seg, face_labels in shots_with_face_ids:
        print (shot_seg, face_labels)
        json_data.append({"time":"%02d:%02d" % divmod(math.ceil(shot_seg.start), 60), 
                          "Speaker":"unknown",
                          "FrameFaces": ','.join(sorted(set(face_labels))),
                          "Topics": [{
                        			"Unknown": "100"
                          }]
        })
        
    with open(output, 'w') as fp:
        json.dump(json_data, fp)
#    {'start': self.start, 'end': self.end}
#    
#    	{
#		"time": "01:35",
#		"Speaker": "Kady Simpson",
#		"FrameFaces": "Kady Simpson",
#		"Topics": [{
#			"Love": "57",
#			"Trade": "62"
#		}]
#	},
    #data = [{"name": "Jane", "age": 17}, {"name": "Thomas", "age": 27}]

    #json_data = json.dumps(data)
    #print(repr(json_data))
    
    #!print(frame_times_dict)
#    agg_segments = dict()
#    value = set()
#    key  = Segment()
#    prev_seg = Segment()
#    for segment, face_id in segments.items():
#        key = segment
#        value+=face_id
##            if segment.overlaps(t):
#        if(prev_seg)
#            agg_segments
#
#        prev_seg = segment        
#        print(key,value)
        


#    agg_segments = dict()
#    for t in frame_times:
#        for face_id, segment in segments.items():
##            if segment.overlaps(t):
#                
#            print(key,value)
#        
    
        #print(type(row))
    #print(grouped['track'][:])
    #duration = [abs(j-i) for i,j in zip(coords_ofs['Time'][:-1], coords_ofs['Time'][1:])]
    #duration = [abs(j-i) for i,j in zip(coords_ofs['Time'][:-1], coords_ofs['Time'][1:])]
#    shots = Shot(video, height=height, context=window, threshold=threshold)
#    shots = Timeline(shots)
#    with open(output, 'w') as fp:
#        pyannote.core.json.dump(shots, fp)
        
        
#    video = Video(filename, ffmpeg=ffmpeg)
#
#    from moviepy.editor import VideoClip, AudioFileClip
#
#    make_frame = get_make_frame(video, tracking, landmark=landmark,
#                                labels=labels, height=height, shift=shift)
#    video_clip = VideoClip(make_frame, duration=video.duration)
#    audio_clip = AudioFileClip(filename)
#    clip = video_clip.set_audio(audio_clip)
#
#    if t_end is None:
#        t_end = video.duration
#
#    clip.subclip(t_start, t_end).write_videofile(output, fps=video.frame_rate)
#
#
#
#
#
#    shots = Shot(video, height=height, context=window, threshold=threshold)
#    shots = Timeline(shots)
#    with open(output, 'w') as fp:
#        pyannote.core.json.dump(shots, fp)
        

if __name__ == '__main__':

    print("Here1")
    # parse command line arguments
    version = 'pyannote-face {version}'.format(version=__version__)
    arguments = docopt(__doc__, version=version)
    print("Here2")
    # initialize video
    filename = arguments['<video>']
    ffmpeg = arguments['--ffmpeg']
    print("Here3")
    verbose = arguments['--verbose']
    #if not (arguments['annotate'] | arguments['train']):
    if filename:
        print("Video(filename, ffmpeg=ffmpeg, verbose=verbose)")
        video = Video(filename, ffmpeg=ffmpeg, verbose=verbose)

    # face tracking
    if arguments['track']:

        shot = arguments['<shot.json>']
        tracking = arguments['<tracking>']
        detect_min_size = float(arguments['--min-size'])
        detect_every = float(arguments['--every'])
        track_min_overlap_ratio = float(arguments['--min-overlap'])
        track_min_confidence = float(arguments['--min-confidence'])
        track_max_gap = float(arguments['--max-gap'])
        track(video, shot, tracking,
              detect_min_size=detect_min_size,
              detect_every=detect_every,
              track_min_overlap_ratio=track_min_overlap_ratio,
              track_min_confidence=track_min_confidence,
              track_max_gap=track_max_gap)

    # facial features detection
    if arguments['extract']:

        tracking = arguments['<tracking>']
        landmark_model = arguments['<landmark_model>']
        embedding_model = arguments['<embedding_model>']
        landmarks = arguments['<landmarks>']
        embeddings = arguments['<embeddings>']
        extract(video, landmark_model, embedding_model, tracking,
                landmarks, embeddings)


    if arguments['demo']:

        tracking = arguments['<tracking>']
        output = arguments['<output>']

        t_start = float(arguments['--from'])
        t_end = arguments['--until']
        t_end = float(t_end) if t_end else None

        shift = float(arguments['--shift'])

        labels = arguments['--label']
        if not labels:
            labels = None

        landmark = arguments['--landmark']
        if not landmark:
            landmark = None

        height = int(arguments['--height'])

        demo(filename, tracking, output,
             t_start=t_start, t_end=t_end,
             landmark=landmark, height=height,
             shift=shift, labels=labels, ffmpeg=ffmpeg)

    if arguments['train']:
        print("Training")

        train_dir = arguments['<train_dir>']
        train_model = arguments['<train_model>']
        landmark_model = arguments['<landmark_model>']
        embedding_model = arguments['<embedding_model>']
        verbose = arguments['--verbose']
        
        #print(train_dir, train_model, landmark_model, embedding_model, verbose)            
        train(train_dir, train_model, landmark_model, embedding_model, verbose)

    if arguments['recognize']:

        embeddings = arguments['<embeddings>']
        train_model = arguments['<train_model>']

        labels = arguments['<labels>']
        if not labels:
            labels = None
        verbose = arguments['--verbose']        
        
        report_file = arguments['--report']        

        #print(embeddings, train_model, labels, labels )            
        recognize(embeddings, train_model, labels, verbose, report_file)
        
    if arguments['annotate']:
        print("Here")
        shots = arguments['<shot.json>']
        if not shots:
            shots = None
        tracking = arguments['<tracking>']
        output = arguments['<output>']

        t_start = float(arguments['--from'])
        t_end = arguments['--until']
        t_end = float(t_end) if t_end else None

        shift = float(arguments['--shift'])

        labels = arguments['--label']
        if not labels:
            labels = None
        

        print(shots,tracking,output,labels)            
        annotate(output, tracking, shots,
                 t_start=t_start, t_end=t_end,
                 shift=shift, labels=labels)
    
    if arguments['evaluate']:
        print("Here")
        reference_file = arguments['<reference.json>']

        hypothesize_file = arguments['<validation.json>']
                
        report_file = arguments['--report']   

        #print(shots,tracking,output,labels)            
        
        evaluate(reference_file, hypothesize_file, report_file)
