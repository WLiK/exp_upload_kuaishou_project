import platform
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
from getFeature import *
from readData import *
import gc
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgb_lr import *

# dump_object("base_xgb.model", base_xgb)
def hashstr(str, nr_bins):
    return int(int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1)

def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

def convert_xgb2ffm(data, model_path, out_path, submit=False, user_filter = set(), mod=1e7+8):
    gc.collect()
    one_hot_cols = ["user_id","relative_time"]
    not_base_cols = ["user_id", "photo_id", "click", "like", "follow", "playing_time"]
    time_cols = ["relative_time", "time"]
    base_cols = [x for x in data.columns if x not in not_base_cols]
    base_xgb = load_object(model_path+"base_xgb.model")
    time_xgb = load_object(model_path+"time_xgb.model")
    print("load from local success")
    print("start transform data")
    all_idx = splitKFold(arange(data.shape[0]), 10)
    print("transform data end")
    print("start generate data")
    chunk_size = 500000
    line_cnt = 0
    lines = []
    if submit:
        data["click"] = 0
    with open(out_path, "w") as f:
        for idx in all_idx:
            gc.collect()
            sub_base_leaves = base_xgb.apply(data.loc[idx, base_cols])
            sub_one_hot_data = data.loc[idx, one_hot_cols].values
            sub_click = data.loc[idx, "click"].values
            for one_hot_feat, xgb_feat, click in zip( sub_one_hot_data, sub_base_leaves,sub_click):
                feats = []
                field = 0
                for feat in one_hot_feat:
                    if field == 1 and feat in user_filter:
                        feats.append((field, str(field)+":"+"less_id"))
                    else:
                        feats.append((field, str(field)+":"+str(int(feat)) ))
                    field = field + 1
                for feat in xgb_feat:
                    feats.append((field, str(field)+":"+str(int(feat))))
                    field = field + 1
                feats = gen_hashed_fm_feats(feats, mod)
                line_cnt = line_cnt + 1
                lines.append(str(click) +" "+ ' '.join(feats) + '\n')
                if line_cnt >= chunk_size:
                    f.write("".join(lines))
                    line_cnt = 0
                    lines = []
        if line_cnt > 0:
            f.write("".join(lines))

def xgb_ffm():
     path = "../"
    one_hot_cols = ["user_id","relative_time"]
    face_feature, text_feature, visual_feature = get_all_feature(visual_dim = 128)

    dataSet = "full"
    model_path = "../xgb_ffm/tmp/model_full/"
    submit = False
    data_train, data_test = get_train_test_data(dataSet, submit=submit)
    data_lr, data_xgb = get_split_data_by_time(dataSet="full", data_train_origin=data_train, split_rate=0.8)
    data_xgb = get_merge(data_xgb, text_feature, face_feature,  visual_feature)
    data_test = get_merge(data_test, text_feature, face_feature,  visual_feature)
    load_from_local = False
    if not load_from_local:
        base_xgb, time_xgb, xgb_enc = train_xgb(data_xgb, data_test)
        one_hot_enc = get_one_hot_enc(np.concatenate((data_train[one_hot_cols], data_test[one_hot_cols]), axis=0))
        dump_object(model_path+"base_xgb.model", base_xgb)
        dump_object(model_path+"time_xgb.model", time_xgb)
        dump_object(model_path+"xgb_enc", xgb_enc)
        dump_object(model_path+"one_hot_enc", one_hot_enc)
    else:
        base_xgb = load_object(model_path+"base_xgb.model")
        time_xgb = load_object(model_path+"time_xgb.model")
        xgb_enc = load_object(model_path+"xgb_enc")
        one_hot_enc = load_object(model_path+"one_hot_enc")
        print("load from local success")
    data_lr = get_merge(data_lr,  text_feature, face_feature,  visual_feature)
    gc.collect()
    convert_xgb2ffm(data_lr, model_path, model_path+"train.ffm")
    convert_xgb2ffm(data_test[data_lr.columns], model_path, model_path+"test.ffm")
    convert_xgb2ffm(data_submit[lr_cols], model_path, model_path+"submit.ffm", submit=True)
    

    
def get_merge(data_origin, text_feature, face_feature, visual_feature):
    if not (data_origin is None):
        data_origin = merge(data_origin, text_feature, on ='photo_id', how='left')
    if not(face_feature is None):
        data_origin = merge(data_origin, face_feature, on ='photo_id', how='left')
    if not (visual_feature is None):
        data_origin = merge(data_origin, visual_feature, on ='photo_id', how='left')
    return data_origin
    
def dump_all_object(base_xgb, time_xgb, xgb_enc, one_hot_enc, lr_model, path="../"):
    dump_object(path+"base_xgb.model", base_xgb)
    dump_object(path+"time_xgb.model", time_xgb)
    dump_object(path+"xgb_enc", xgb_enc)
    dump_object(path+"one_hot_enc", one_hot_enc)
    dump_object(path+"lr_model.model", lr_model)
    
def load_all_object():
    base_xgb = load_object("base_xgb.model")
    time_xgb = load_object("time_xgb.model")
    xgb_enc = load_object("xgb_enc")
    one_hot_enc = load_object("one_hot_enc")
    lr_model = load_object("lr_model.model")
    return base_xgb, time_xgb, xgb_enc, one_hot_enc, lr_model


def get_one_hot_enc(data):
    one_hot_enc = OneHotEncoder()
    one_hot_enc.fit(data)
    return one_hot_enc

     
def train_xgb(data_xgb, data_test, not_base_cols = ["user_id", "photo_id", "click", "like", "follow", "playing_time"],
             time_cols = ["relative_time", "time"], one_hot_cols = ["user_id","relative_time"]):
    #build  base tree
    base_cols = [x for x in data_xgb.columns if x not in not_base_cols]
    print("build base tree\n")
    base_xgb = xgb.XGBClassifier(max_depth=6, learning_rate = 0.2, scale_pos_weight=3,
                 n_estimators=60,objective="binary:logistic",nthread=16,booster='gbtree')
    
    base_xgb.fit( data_xgb[base_cols], data_xgb['click'],
        eval_set=[( data_xgb[base_cols], data_xgb['click']),(data_test[base_cols], data_test['click'])],eval_metric="auc")
    y_pred_test = base_xgb.predict_proba(data_test[base_cols])[:, 1]
    xgb_test_auc = metrics.roc_auc_score(data_test['click'], y_pred_test)
    print('base xgboost test auc: %.5f' % xgb_test_auc)
    print("build base tree end\n")
    
    print("build time tree\n ")
    time_xgb = xgb.XGBClassifier(max_depth=6, learning_rate = 0.1, scale_pos_weight=2,
                 n_estimators=2,objective="binary:logistic", nthread=8,booster='gbtree')
    time_xgb.fit( data_xgb[time_cols], data_xgb['click'],
        eval_set=[( data_xgb[time_cols], data_xgb['click']),(data_test[time_cols], data_test['click'])],eval_metric="auc")
    y_pred_test = time_xgb.predict_proba(data_test[time_cols])[:, 1]
    xgb_test_auc = metrics.roc_auc_score(data_test['click'], y_pred_test)
    print('time xgboost test auc: %.5f' % xgb_test_auc)
    print("time tree params:",time_xgb.get_params())
    
    
    train_base_leaves  = base_xgb.apply(data_xgb[base_cols])
    test_base_leaves  = base_xgb.apply(data_test[base_cols])
    train_time_leaves  = time_xgb.apply(data_xgb[time_cols])
    test_time_leaves  = time_xgb.apply(data_test[time_cols])
    train_data_after_xgb = np.concatenate(( train_base_leaves, train_time_leaves), axis=1)
    test_data_after_xgb = np.concatenate(( test_base_leaves,test_time_leaves), axis=1)
    xgb_enc = OneHotEncoder()
    xgb_enc.fit(np.concatenate((train_data_after_xgb, test_data_after_xgb), axis=0))
    return base_xgb, time_xgb, xgb_enc

def generate_feature(path):
    get_photo_face_feature(path)
    get_text_feature(30, path)
    getVisualFeature(visual_path, 128)
    

if __name__ == "__main__":
    xgb_ffm()



