import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets
from numpy import *
from pandas import *
import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from scipy.sparse import *
from scipy import sparse as ssp
from sklearn.svm import SVC
from numpy import inf
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import scipy.sparse
from time import time

def get_photo_face_feature(mode = 0, has_flag =  False):
    # text = '[[0.0252, 0, 19, 50], [0.0283, 0, 22, 71]]'
    print("start load face feature")
    t1 = time()
    path = "/home/wulikang/kuaishou/noprenn/"
    if(mode == 0):
       f = open(path+'fstrain_face.txt', 'r')
    else:
       f = open(path+'fstest_face.txt', 'r')
    face_feature_list = []
    photo_id_list = []
    face_rate_mean_list = []
    for line in f.readlines():
        line = line.strip()
        photo_id, face = line.split('\t')
        face_list = json.loads(face)
        words = ''
        face_rate_list = []
        age_list = []
        appearance_list = []
        sex_list = []
        face_feature = []
        for face in face_list:
            face_rate_list.append(np.float(face[0]))
            sex_list.append(int(face[1]))
            age_list.append(int(face[2]))
            appearance_list.append(int(face[3]))
        photo_id_list.append(photo_id)
        face_rate_mean_list.append(mean(face_rate_list))
        # face_feature_list.append(int(photo_id))
        face_feature.append(int(photo_id))
        face_feature.append(mean(face_rate_list))
        face_feature.extend([max(age_list),mean(age_list)])
        face_feature.extend([mean(appearance_list)])
        face_feature.append(sum(sex_list))
        face_feature.append(len(sex_list) - sum(sex_list))
        face_feature_list.append(face_feature)
    f.close()
    face_columns_name = ['photo_id', 'face_rate_mean', 'age_max','age_mean','app',
                         'man_num', 'woman_num']
    t2 = time()
    print(str(t2 - t1)+'s')
    print("load face data end")
    face_df = DataFrame(face_feature_list, columns = face_columns_name)
    face_df['face_rate_mean'] = (face_df['face_rate_mean'] * 1000).astype(int)
    face_df['age_mean'] = (face_df['age_mean']).astype(int)
    face_df['app'] = (face_df['app']).astype(int)

    if has_flag:
        face_df['face_flag'] = 1
    return face_df

