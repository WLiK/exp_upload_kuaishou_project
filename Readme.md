# 用户兴趣建模大赛 Readme



--------------------------------
xgb+ffm

## 项目结构
```
getFeature.py   # 预处理相关特征
xgb_ffm.py      # 生成经过xgb模型处理后的符合ffm格式的数据
```

## 运行环境
运行此项目，需要先安装好anaconda，然后配置好word2vec, xgboost相关库
同时需要安装v123版本的[libffm](https://github.com/guestwalk/libffm/releases)

## 数据预处理
运行getFeature.py，预先生成处理后的特征
生成后会有text_feature_4.csv，visual_128_mean_5.csv, face_feature.csv三个文件

## xgb+ffm 运行流程
运行xgb_ffm, 运行成功后会有train.ffm,test.ffm,submit.ffm三个文件
调用libffm库生成ffm模型，并用生成的模型对submit.ffm进行预测


---------------------------------
dnn
## 项目结构
```
getface.py      # 预处理face特征
text_dnn_32.py  # 深度学习模型
```

## 运行环境
核心idea:对特征进行分类别，进行embedding后做attention,即优化后的dnn模型来预测ctr

## 运行环境
运行此项目，需要先安装好python3，tensorflow-gpu1.4.0,keras2.1.2


## 数据预处理
需要运行getFeature.py生成visual_32_mean_5.csv文件，生成photo_text_df_full_10.csv文件（text特征文件）

## xgb+dnn 运行流程
运行text_dnn_32.py文件，每运行一个epoch便会生成一个预测文件32epoch*.xt






