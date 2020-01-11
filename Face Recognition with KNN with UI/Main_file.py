from multiprocessing import freeze_support
import cv2
import numpy as np
import os
import dlib
import pandas as pd
from imutils import face_utils
from imutils.face_utils import FaceAligner
from tkinter import *
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import shutil
from random import shuffle
from sklearn import svm, neighbors
import warnings
freeze_support()
root=Tk()

root.configure(background="white")

#root.geometry("300x300")
#face recognition api
face_detector = dlib.get_frontal_face_detector()

predictor_model = 'shape_predictor_68_face_landmarks.dat'
pose_predictor = dlib.shape_predictor(predictor_model)

face_recognition_model = 'dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _rect_to_tuple(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _tuple_to_rect(rect):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param rect:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(rect[3], rect[0], rect[1], rect[2])


def _trim_rect_tuple_to_bounds(rect, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param rect:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(rect[0], 0), min(rect[1], image_shape[1]), min(rect[2], image_shape[0]), max(rect[3], 0)


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_image_file(filename, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param filename: image file to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    img = cv2.imread(filename)

    # If very large size image, Resize the image
    if img.shape[0] > 800:
        baseheight = 500
        w = (baseheight / img.shape[0])
        p = int(img.shape[1] * w)
        img =cv2.resize(img, (baseheight, p))
    elif img.shape[1] > 800:
        baseheight = 500
        w = (baseheight / img.shape[1])
        p = int(img.shape[0] * w)
        img = cv2.resize(img, (p, baseheight))

    return img


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return face_detector(img, number_of_times_to_upsample)


def face_location(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of tuples of found face locations in tuple (top, right, bottom, left) order
    """
    return [_trim_rect_tuple_to_bounds(_rect_to_tuple(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_tuple_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def face_encoding(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimentional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
def get_prediction_images(prediction_dir):
    files = [x[2] for x in os.walk(prediction_dir)][0]
    l = []
    exts = [".jpg", ".jpeg", ".png"]
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in exts:
            l.append(os.path.join(prediction_dir, file))

    return l

def function1():
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)

    # face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    name = input("Enter name of person:")
    user = pd.read_csv('user_names.csv')
    user_name = []
    for i in range(0, len(user)):
        user_name.append(str(user.iloc[i, 1]))
    try:
        if user_name.index(name) >= 0:
            print('-----------this name is already registered--------------')
            name = input("please enter your name with some prefix and use underscore '-' like aditya_mangal\n")
    except:
        print('-----------------------------Welcome--------------------------')
    path = 'training-images'
    directory = os.path.join(path, name)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok='True')

    number_of_images = 0
    MAX_NUMBER_OF_IMAGES = 30
    count = 0

    while number_of_images < MAX_NUMBER_OF_IMAGES:
        ret, frame = video_capture.read()

        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        faces = detector(frame_gray)
        if len(faces) == 1:
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = frame_gray[y - 50:y + h + 100, x - 50:x + w + 100]
            face_aligned = face_aligner.align(frame, frame_gray, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            if count == 2:
                cv2.imwrite(os.path.join(directory, str(name + str(number_of_images) + '.jpg')), face_aligned)
                number_of_images += 1
                count = 0
            # print(count)
            count += 1

        cv2.imshow('Video', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    def fun(rows, max):
        le = len(rows)
        lis = []
        for i in range(0, le):
            lis.append(max + i + 1)
        return lis

    def _get_training_dirs(training_dir_path):
        return [x[0] for x in os.walk(training_dir_path)][1:]

    def _get_training_labels(training_dir_path):
        # print([x[1] for x in os.walk(training_dir_path)][0])
        return [x[1] for x in os.walk(training_dir_path)][0]

    def _get_each_labels_files(training_dir_path):
        return [x[2] for x in os.walk(training_dir_path)][1:]

    def _filter_image_files(training_dir_path):
        exts = [".jpg", ".jpeg", ".png"]

        training_folder_files_list = []
        for list_files in _get_each_labels_files(training_dir_path):
            l = []
            for file in list_files:
                imageName, ext = os.path.splitext(file)
                if ext.lower() in exts:
                    l.append(file)
            training_folder_files_list.append(l)

        return training_folder_files_list

    def _zipped_folders_labels_images(training_dir_path, labels):
        return list(zip(_get_training_dirs(training_dir_path),
                        labels,
                        _filter_image_files(training_dir_path)))

    def create_dataset(training_dir_path, labels):
        X = []
        for i in _zipped_folders_labels_images(training_dir_path, labels):
            for fileName in i[2]:
                file_path = os.path.join(i[0], fileName)
                img = load_image_file(file_path)
                imgEncoding = face_encoding(img)

                if len(imgEncoding) > 1:
                    print('\x1b[0;37;43m' + 'More than one face found in {}. Only considering the first face.'.format(
                        file_path) + '\x1b[0m')
                if len(imgEncoding) == 0:
                    print('\x1b[0;37;41m' + 'No face found in {}. Ignoring file.'.format(file_path) + '\x1b[0m')
                else:
                    # print('Encoded {} successfully.'.format(file_path))
                    X.append(np.append(imgEncoding[0], i[1]))
        return X

    encoding_file_path = './encoded-images-data.csv'
    training_dir_path = './training-images'
    labels_fName = "labels.pkl"

    # Get the folder names in training-dir as labels
    # Encode them in numerical form for machine learning
    labels = _get_training_labels(training_dir_path)
    user_name = pd.DataFrame(labels)
    user_name_file = 'user_names.csv'
    if os.path.isfile(user_name_file):
        user_name.to_csv('user_names.csv', mode='a', header=False)
    else:
        user_name.to_csv('user_names.csv')
    user_name = pd.read_csv('user_names.csv')
    labels = []
    for i in range(0, len(user_name)):
        labels.append(str(user_name.iloc[i, 1]))
    # print(labels)
    user_name = pd.read_csv('user_names.csv')
    # print(user_name.iloc[0,1])
    labels.insert(0, str(user_name.iloc[0, 1]))
    # print(labels)
    le = LabelEncoder().fit(labels)
    # print(le)
    # labelsNum = le.transform(labels)
    labelsNum = []
    if os.path.isfile(encoding_file_path):
        data = pd.read_csv('encoded-images-data.csv')
        max_last1 = int(max(data.iloc[:, -1]))
        labelsNum.append(max_last1 + 1)
    else:
        labelsNum.append(int(0))

    # print(labelsNum)
    # nClasses = len(le.classes_)
    dataset = create_dataset(training_dir_path, labelsNum)
    df = pd.DataFrame(dataset)
    # print(df)

    # if file with same name already exists, backup the old file
    if os.path.isfile(encoding_file_path):
        # print("{} already exists. Backing up.".format(encoding_file_path))
        data = pd.read_csv('encoded-images-data.csv')
        max_0 = max(data.iloc[:, 0])
        max_last = max(data.iloc[:, -1])
        # print(max_last)
        df.index = fun(df, max_0)
        for i in range(0, len(df)):
            df.iloc[i, -1] = max_last + 1
        df.to_csv('encoded-images-data.csv', mode='a', header=False)
        shutil.rmtree('training-images')
        os.mkdir('training-images')

    else:
        df.to_csv(encoding_file_path)
        shutil.rmtree('training-images')
        os.mkdir('training-images')

    # print("{} classes created.".format(nClasses))
    print('\x1b[6;30;42m' + "Saving labels pickle to'{}'".format(labels_fName) + '\x1b[0m')
    with open(labels_fName, 'wb') as f:
        pickle.dump(le, f)
    print('\x1b[6;30;42m' + "Training Image's encodings saved in {}".format(encoding_file_path) + '\x1b[0m')

    encoding_file_path = './encoded-images-data.csv'
    labels_fName = 'labels.pkl'

    if os.path.isfile(encoding_file_path):
        df = pd.read_csv(encoding_file_path)
    else:
        print('\x1b[0;37;41m' + '{} does not exist'.format(encoding_file_path) + '\x1b[0m')
        quit()

    if os.path.isfile(labels_fName):
        with open(labels_fName, 'rb') as f:
            le = pickle.load(f)
    else:
        print('\x1b[0;37;41m' + '{} does not exist'.format(labels_fName) + '\x1b[0m')
        quit()

    # Read the dataframe into a numpy array
    # shuffle the dataset
    full_data = np.array(df.astype(float).values.tolist())
    shuffle(full_data)
    # print("full_data")
    # print(full_data)
    # Extract features and labels
    # remove id column (0th column)
    X = np.array(full_data[:, 1:-1])
    y = np.array(full_data[:, -1:])

    # fit the data into a support vector machine
    # clf = svm.SVC(C=1, kernel='linear', probability=True)
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
    clf.fit(X, y.ravel())

    fName = "./classifier.pkl"
    # if file with same name already exists, backup the old file
    if os.path.isfile(fName):
        print('\x1b[0;37;43m' + "{} already exists. Backing up.".format(fName) + '\x1b[0m')
        os.remove(fName)

    # save the classifier pickle
    with open(fName, 'wb') as f:
        pickle.dump((le, clf), f)
    print('\x1b[6;30;42m' + "Saving classifier to '{}'".format(fName) + '\x1b[0m')

    print('-----------------------------------------------------------------------')
    print('-------------Successfully Registered------------------------------------')
    print('------------------------------------------------------------------------')

def function2():
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)
    shutil.rmtree('test-images')
    os.mkdir('test-images')
    video_capture = cv2.VideoCapture(0)

    # Load Face Recogniser classifier
    fname = 'classifier.pkl'
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            (le, clf) = pickle.load(f)
    else:
        print('\x1b[0;37;43m' + "Classifier '{}' does not exist".format(fname) + '\x1b[0m')
        quit()


    number_of_images = 0
    MAX_NUMBER_OF_IMAGES = 5
    count = 0

    while number_of_images < MAX_NUMBER_OF_IMAGES:
        ret, frame = video_capture.read()

        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        faces = detector(frame_gray)
        if len(faces) == 1:
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = frame_gray[y - 50:y + h + 100, x - 50:x + w + 100]
            face_aligned = face_aligner.align(frame, frame_gray, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            if count == 2:
                cv2.imwrite("test-images/" + str(str(number_of_images) + '.jpg'), face_aligned)
                number_of_images += 1
                count = 0
            # print(count)
            count += 1
        cv2.imshow('Video', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()





    fname = 'classifier.pkl'
    prediction_dir = './test-images'

    encoding_file_path = './encoded-images-data.csv'
    df = pd.read_csv(encoding_file_path)
    full_data = np.array(df.astype(float).values.tolist())
    X = np.array(full_data[:, 1:-1])
    y = np.array(full_data[:, -1:])

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            (le, clf) = pickle.load(f)
    image = cv2.imread('wait.jpg')
    resize = cv2.resize(image, (640,480))
    # image.resize((320,240))
    cv2.imshow('img', resize)
    check_name = " "
    try:
        for image_path in get_prediction_images(prediction_dir):
            img = load_image_file(image_path)
            X_faces_loc = face_location(img)

            faces_encodings = face_encoding(img, known_face_locations=X_faces_loc)
            closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)

            is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(X_faces_loc))]

            # predict classes and cull classifications that are not with high confidence
            predictions = [(le.inverse_transform([int(pred)])[0], loc) if rec else ("Unknown", loc) for pred, loc, rec in
                           zip(clf.predict(faces_encodings), X_faces_loc, is_recognized)]

            # print(predictions[0][0])
            name = predictions[0][0]
            if check_name == name:
                count+=1
            else:
                check_name = name
                count=1
            if count == 5:
                break
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        cv2.destroyAllWindows()
        shutil.rmtree('test-images')
        os.mkdir('test-images')
        if check_name!= 'Unknown':
            print("hi "+check_name)
    except :
        print("Try again")
        cv2.destroyAllWindows()
        shutil.rmtree('test-images')
        os.mkdir('test-images')

def function5():
    root.destroy()



#stting title for the window
root.title("AUTOMATIC ATTENDANCE MANAGEMENT USING FACE RECOGNITION")

#creating a text label
Label(root, text="FACE RECOGNITION ATTENDANCE SYSTEM",font=("times new roman",20),fg="white",bg="maroon",height=2).grid(row=0,rowspan=2,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

#creating first button
Button(root,text="Create Dataset",font=("times new roman",20),bg="#0D47A1",fg='white',command=function1).grid(row=3,columnspan=2,sticky=W+E+N+S,padx=5,pady=5)

#creating second button
Button(root,text="Recognition",font=("times new roman",20),bg="#0D47A1",fg='white',command=function2).grid(row=4,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)


Button(root,text="Exit",font=('times new roman',20),bg="maroon",fg="white",command=function5).grid(row=9,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)


root.mainloop()
