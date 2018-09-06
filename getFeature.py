from numpy import *
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from scipy.sparse import *
from scipy import sparse as ssp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import scipy.sparse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from common import *
from readData import *




def get_merge(data_origin, text_feature, face_feature, visual_feature):
    if not (data_origin is None):
        data_origin = merge(data_origin, text_feature, on ='photo_id', how='left')
    if not(face_feature is None):
        data_origin = merge(data_origin, face_feature, on ='photo_id', how='left')
    if not (visual_feature is None):
        data_origin = merge(data_origin, visual_feature, on ='photo_id', how='left')
    return data_origin
    
def get_auc(data):
    return metrics.roc_auc_score(data["click"], data["prob"])

def get_auc_with_time(data, group_num = 50):
    data['time'] = data['time'] - data['time'].min()+1.0
    data["auc_time"] = (data['time'] / ((data['time'].max()+1)/50.0)).astype(int)
    time_auc = data.groupby(["auc_time"]).apply(get_auc)
    return time_auc

def dump_object(filename, dump_object):
    f = open(filename, "wb")
    pickle.dump(dump_object, f)
    f.close()
    
def load_object(filename):
    f = open(filename, "rb")
    load_object = pickle.load(f)
    f.close()
    return load_object


def get_photo_face_feature(path="../", has_flag =  False):
    # text = '[[0.0252, 0, 19, 50], [0.0283, 0, 22, 71]]'
    back_file = path + 'face_feature.csv'
    print("start load face feature")
    if os.path.isfile(back_file):
        face_df = read_csv(back_file)
        print("file exist load face feature end")
        return face_df
    face_feature_list = []
    photo_id_list = []
    file_list = ['train_face.txt', 'test_face.txt']
    for open_file in file_list:
        f = open(path + open_file, 'r')
        for line in f.readlines():
            line = line.strip()
            photo_id, face = line.split('\t')
            face_list = json.loads(face)
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
            face_feature.append(int(photo_id))
            face_feature.extend([int(mean(face_rate_list)*1000), int(sum(face_rate_list)*1000)])
            face_feature.extend([max(age_list), min(age_list), int(mean(age_list))])
            face_feature.extend([max(appearance_list), min(appearance_list),int(mean(appearance_list))])
            face_feature.append(sum(sex_list))
            face_feature.append(len(sex_list) - sum(sex_list))
            face_feature_list.append(face_feature)
        f.close()
    face_columns_name = ['photo_id', 'face_rate_mean', 'face_rate_sum','age_max', 'age_min', 'age_mean',
                         'appearance_max', 'appearance_min', 'appearance_mean',
                         'man_num', 'woman_num']
    print("load face data end")
    face_df = DataFrame(face_feature_list, columns = face_columns_name)
    if has_flag:
        face_df['face_flag'] = 1
    face_df.to_csv(back_file)
    return face_df

def get_visual_feature(visual_dim, path = "../"):
    print("start load ", visual_dim, " visual feature")
    visual_df = read_csv(path +"visual_"+str(visual_dim)+"_mean_5.csv")
    for i in visual_df.columns:
        if i == "photo_id":
            continue
        visual_df[i] = visual_df[i].astype(np.float16)
    print("load ", visual_dim, " visual feature end")
    return visual_df

def get_all_feature(path="../", data_type="train", visual_dim = 128):
    text_feature= get_text_feature(30, path)
    face_feature = get_photo_face_feature()#get_face_tfidf(max_feature_num = 32, path="../")
    visual_feature = get_visual_feature(visual_dim)
    return face_feature, text_feature, visual_feature

def get_text_feature(feature_num, path = "../"):
    print("start load text feature")
    # photo_text_df_full_10.csv
    back_file = path + "text_feature_4.csv"
    if os.path.isfile(back_file):
        photo_text_df = read_csv(back_file)
        for i in photo_text_df.columns:
            if i == "photo_id":
                continue
            photo_text_df[i] = photo_text_df[i].apply(lambda x:float("%.4f"%x))
        print("file exist load text feature end")
        return photo_text_df
    result = []
    photo_id_list = []
    cnt = 0
    key_value = []
    f = open(path + "train_text.txt", "r")
    for line in f.readlines():
        photo_id, text = line.strip().split('\t')
        text = text.strip().split(",")
        result.append(text)
        photo_id_list.append(int(photo_id))
    f.close()
    f = open(path + "test_text.txt", "r")
    for line in f.readlines():
        photo_id, text = line.strip().split('\t')
        text = text.strip().split(",")
        result.append(text)
        photo_id_list.append(int(photo_id))
    f.close()
    print("parse text end\nstart train word2vec")
    st = time.clock()
    model = Word2Vec(result, size=feature_num, window=5, min_count=1, workers=10, sg=0)
    ed = time.clock()
    print("word2vec cost:",ed-st)
    print("train word2vec end\nstart get text feature dataFrame")


    print("there are 20 fold")
    split_result = splitKFold(result, 20)
    workers = []
    result = []
    st = time.clock()
    pool = Pool(processes=12)
    for i in split_result:
        workers.append(pool.apply_async(getWordVec, args=(model, i, )))
    pool.close()
    pool.join()
    print("start merge text feature")
    for i in workers:
        result.extend(i.get())
    ed = time.clock()
    print(ed - st)
    print(result[0])

    columns_name = ["text_mean_"+str(i) for i in range(feature_num)]
    model.save(path+"word2vec_"+str(feature_num)+".model")
    photo_text_df = DataFrame(result, columns = columns_name)
    photo_text_df['photo_id'] = photo_id_list
    photo_text_df.to_csv(back_file)
    print("load text data end")
    print(photo_text_df.shape)
    return photo_text_df

def get_train_test_data(dataSet="small", path="../", submit=False):
    print("start read train&test data")
    data_train_org = None
    if submit:
        names = ['user_id', 'photo_id', 'click','like',  'follow', 'time', 'playing_time', 'duration_time']
        data_train_org = read_csv(path + "train_interaction.txt", sep="\t", names = names,header=None)
    else:
        names = ['user_id', 'photo_id', 'click', 'time', 'duration_time']
        data_train_org = read_csv(path + dataSet + "_train")[names]
    names = ['user_id', 'photo_id', 'click', 'time', 'duration_time']
    data_test_org = read_csv(path + dataSet + "_test")[names]
    data_train_org = get_process_data(data_train_org)
    data_test_org = get_process_data(data_test_org)
    print("read train&test data end")
    return data_train_org, data_test_org
    
def get_submit_data(path="../"):
    names = ["user_id", "photo_id", "time", "duration_time"]
    data_test_org = read_csv(path + "test_interaction.txt", sep="\t", names = names,header=None)
    data_test_org = get_process_data(data_test_org)
    print("read submit data end & process data end")
    return data_test_org 
    
def get_process_data(data):
    data.loc[:,'time'] = data['time'].copy() - data['time'].min()
    photo_time_start = data[['photo_id', 'time']].groupby(['photo_id']).min().reset_index()
    photo_time_start.columns = ['photo_id', 'start_time']
    data = merge(data, photo_time_start, on = 'photo_id', how='left').fillna(0)
    data['relative_time'] = ((data['time'] - data['start_time'])/(1e8/1000)).astype(int)
    return data


def getWordVec(model, docs):
    result = []
    for doc in docs:
        photo_word = []
        for word in doc:
            vt = model.wv[word]
            photo_word.append(list(vt))
        mean_vt = list(mean(photo_word, axis=0))
        result.append(mean_vt)
    print("finish one text word2vec fold")
    return result




def get_split_data_by_time(dataSet ="small", path ="../", split_rate = (2/3), data_train_origin=None, save=False):
    # read data
    file = path + dataSet
    print("read data ing...")
    names = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
    if data_train_origin is None:
        save = True
        data_train_origin = read_csv(path+"train_interaction.txt",header=None,sep = '\t',names=names)
    # select random user by dataSet size
    print("select random user by dataSet size")
    all_user_id = unique(data_train_origin['user_id'].values)
    data_rate = 0.2
    if dataSet == "full":
        data_rate = 1.0
    elif dataSet == "big":
        data_rate = 0.8
    elif dataSet == "mid":
        data_rate = 0.5
    user_size = int(data_rate*len(all_user_id))
    user_id_st = set(np.random.choice(all_user_id, user_size, replace=False))
    data = data_train_origin[data_train_origin['user_id'].isin(user_id_st)]
    print("split photo by time")
    all_photo_time = unique(data['time'].values)
    split_idx = int(len(all_photo_time)*split_rate)
    split_time = sort(all_photo_time)[split_idx]
    data_train =  data[data['time']<=split_time]
    data_test = data[data['time'] > split_time]
    del data
    gc.collect()
    test_photo_id = set(data_test['photo_id'].values)-set(data_train['photo_id'].values)
    data_test = data_test[data_test['photo_id'].isin(test_photo_id)]
    feature_col = ['click', 'user_id', 'duration_time', 'time', 'photo_id']
    if save:
        data_train[feature_col].to_csv(path+dataSet+"_train", index=False)
        data_test[feature_col].to_csv(path+dataSet+"_test", index=False)
    return data_train, data_test


def get_visual_feature(visual_dim, path = "../"):
    print("start load ", visual_dim, " visual feature")
    visual_df = read_csv(path + +"visual_"+str(visual_dim)+"_mean.csv")
    for i in visual_df.columns:
        if i == "photo_id":
            continue
        visual_df[i] = visual_df[i].astype(np.float16)
    print("load ", visual_dim, " visual feature end")
    return visual_df

def readSingleFile(path, files, block_num):
    result = []
    for file in files:
        arr = []
        try:
            arr = np.load(path+file)
        except OSError:
            print(file+" error ")
            continue
        arr = skimage.measure.block_reduce(arr, (1, block_num), np.mean).ravel().tolist()
        arr.append(int(file))
        result.append(arr)
    print("finish one fold")
    return result

def splitKFold(myList, num):
    length = len(myList)
    result = []
    sz = int((length+num-1)/num)
    for i in range(num-1):
        result.append(myList[i*sz:(i+1)*sz])
    result.append(myList[(i+1)*sz:])
    return result

def getVisualFeature(path, visual_dim):
    print("read list dir")
    onlyFiles = [f for f in listdir(path)]
    split_num = 48
    splitFiles = splitKFold(onlyFiles, split_num)
    print("read list dir end \nstart multiply process read file\n there are "+str(split_num)+" fold\n")
    workers = []
    result = []
    st = time.clock()
    pool = Pool(processes=12)
    for i in splitFiles:
        workers.append(pool.apply_async(readSingleFile, args=(path,i,int(2048/visual_dim))))
    pool.close()
    pool.join()
    print("start construct dataFrame")
    for i in workers:
        result.extend(i.get())
    print("dataFrame save ...")
    columns_name = ["fenmian_"+str(i) for i in range(visual_dim)]+['photo_id']
    visual_df = DataFrame(result, columns = columns_name)
    for i in visual_df.columns:
        if i == "photo_id":
            continue
        visual_df[i] = visual_df[i].apply(lambda x:float("%.5f"%x))
    visual_df.to_csv("../visual_"+str(visual_dim)+"_mean_test_5.csv")
    print("dataFrame save end")
    ed = time.clock()
    print(ed - st)
    return visual_df

def generate_feature(path):
    get_photo_face_feature(path)
    get_text_feature(30, path)
    getVisualFeature(visual_path, 128)
    

if __name__ == "__main__":
    generate_feature("./")
