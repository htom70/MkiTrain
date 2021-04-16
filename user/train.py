import math
import time
import pickle
import flask
import mariadb
from flask import request, jsonify
import mysql.connector
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from nltk import DecisionTreeClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, \
    Normalizer
from xgboost import XGBClassifier
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True


def getStoreConnection():
    connection = mariadb.connect(
        pool_name="store",
        pool_size=32,
        host="store.usr.user.hu",
        user="mki",
        password="pwd"
    )
    return connection


def getLocalConnection():
    connection = mysql.connector.connect(
        pool_name="local",
        pool_size=16,
        host="localhost",
        user="root",
        password="TOmi_1970")
    return connection


def getAllRecordsFromDatabase(databaseName):
    start = time.time()
    connection = getStoreConnection()
    cursor = connection.cursor()
    sql_use_Query = "USE " + databaseName
    cursor.execute(sql_use_Query)
    sql_select_Query = "select * from transaction order by timestamp"
    cursor.execute(sql_select_Query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    numpy_array = np.array(result)
    end = time.time()
    elapsedTime = end - start
    print(f'{databaseName} beolvasva, betöltési idő: {elapsedTime}, rekordszám: {numpy_array.shape}')
    return numpy_array[:, :]


def getRandomSamplers():
    samplers = {
        'UnderSampler': RandomUnderSampler(sampling_strategy=0.5),
        'OverSampler': RandomOverSampler(sampling_strategy=0.5),
    }
    return samplers


def getScalers():
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler(),
        'QuantileTransformer-Normal': QuantileTransformer(output_distribution='normal'),
        'QuantileTransformer-Uniform': QuantileTransformer(output_distribution='uniform'),
        'Normalizer': Normalizer(),
    }
    return scalers


def getFeatureSelectors():
    featureSelectors = {
        'RFE': RFECV(estimator=XGBClassifier(tree_method='gpu_hist', gpu_id=0), n_jobs=-1),
        'PCA': PCA(n_components=0.95, svd_solver='full'),
        'SVD': TruncatedSVD()
    }
    return featureSelectors


def getModels():
    models = {
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1),
        # 'DecisionTree': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_jobs=-1),
        'Light GBM': LGBMClassifier(n_jobs=-1),
        'XGBoost': XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    }
    return models


def storePickledPipeline(pipeline, pickledPipelineName):
    file = pickledObjectsFolder + pickledPipelineName + ".pickle"
    fileHandler = open(file, 'wb')
    pickle.dump(pipeline, fileHandler)
    fileHandler.close()


def calculateF(beta, precision, recall):
    temp = beta * beta * precision + recall
    if temp != 0:
        f_beta = (1 + beta) * (1 + beta) * precision * recall / temp
    else:
        f_beta = 0
    return f_beta


def qualifyPipeline(pipeline, pipelineName, testFeatures, testLabels, databaseName, samplerName, scalerName,
                    featureSelectorName, modelName):
    try:
        connection = getLocalConnection()
        cursor = connection.cursor()
        sqlUSEQuery = "USE mki"
        cursor.execute(sqlUSEQuery)

        predictedLabels = pipeline.predict(testFeatures)
        confusionMatrix = confusion_matrix(testLabels, predictedLabels)
        print(f"Confusion Matrix: {confusionMatrix}")
        TN = int(confusionMatrix[0][0])
        FP = int(confusionMatrix[0][1])
        FN = int(confusionMatrix[1][0])
        TP = int(confusionMatrix[1][1])
        temp = TP + FN
        sensitivity = 0
        if temp != 0:
            sensitivity = TP / (TP + FN)
        temp = TN + FP
        specificity = 0
        if temp != 0:
            specificity = TN / (TN + FP)
        accuracy = float(accuracy_score(testLabels, predictedLabels))
        balanced_accuracy = float(balanced_accuracy_score(testLabels, predictedLabels))
        precision = 0
        temp = TP + FP
        if temp != 0:
            precision = TP / (TP + FP)
        recall = float(recall_score(testLabels, predictedLabels))
        temp = TP + FN
        PPV = 0
        if temp != 0:
            PPV = TP / (TP + FN)
        temp = TN + FN
        NPV = 0
        if temp != 0:
            NPV = TN / (TN + FN)
        temp = FN + TP
        FNR = 0
        if temp != 0:
            FNR = FN / (FN + TP)
        temp = FP + TN
        FPR = 0
        if temp != 0:
            FPR = FP / (FP + TN)
        FDR = 0
        temp = FP + TP
        if temp != 0:
            FDR = FP / (FP + TP)
        temp = FN + TN
        FOR = 0
        if temp != 0:
            FOR = FN / (FN + TN)
        f1 = float(f1_score(testLabels, predictedLabels))
        f_05 = calculateF(0.5, precision, recall)
        f2 = calculateF(2, precision, recall)
        temp = math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN)
        MCC = 0
        if temp != 0:
            MCC = (TP * TN - FP * FN) / temp
        ROCAUC = float(roc_auc_score(testLabels, predictedLabels))
        Youdens_statistic = sensitivity + specificity - 1

        sql_insert_Query = "INSERt INTO metrics (pipeline_name, database_name,sampler_name, scaler_name, feature_selector_name, model_name, " \
                           "TP,FP,TN,FN,sensitivity,specificity,accuracy,balanced_accuracy,prec,recall,PPV,NPV,FNR,FPR,FDR,F_OR,f1,f_05,f2,MCC,ROCAUC,Youdens_statistic) VALUES" \
                           "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
        values = (
            pipelineName, databaseName, samplerName, scalerName, featureSelectorName, modelName,
            TP, FP, TN, FN, sensitivity, specificity, accuracy, balanced_accuracy, precision, recall, PPV, NPV, FNR,
            FPR, FDR, FOR, f1, f_05, f2, MCC, ROCAUC, Youdens_statistic)
        cursor.execute(sql_insert_Query, values)
        connection.commit()
    except Exception as e:
        print(type(e, e))
        print(
            f'Hiba a mutatószámok képzésekor vagy adatbázisba íráskor, adatbásis neve {databaseName}; sampler: {samplerName}; skálázó: {scalerName}; feature selector: {featureSelectorName}, model: {modelName}')
    finally:
        cursor.close()
        connection.close()


def trainPipeLine(databaseName, samplerName, scalerName, featureSelectorName, modelName, expectedVariance):
    dataSet = getAllRecordsFromDatabase(databaseName)
    availableSamplers = getRandomSamplers()
    availableScalers = getScalers()
    availableFeatureSelectors = getFeatureSelectors()
    availableModels = getModels()
    features = dataSet[:, 1:-1]
    binaries = dataSet[:, -1:]
    binaries = binaries.astype(int)
    sampler = availableSamplers.get(samplerName)
    sampledFeatures, sampledLabels = sampler.fit_resample(features, binaries)
    scaler = availableScalers.get(scalerName)
    featureSelector = availableFeatureSelectors.get(featureSelectorName)
    model = availableModels.get(modelName)
    pipeline = Pipeline(
        [('scaler', scaler),
         ('featureSelector', featureSelector),
         ('m', model)])
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(sampledFeatures, sampledLabels,
                                                                            test_size=0.2, random_state=0)
    pipeline.fit(trainFeatures, trainLabels)
    pickledPipelineName = "Pipeline_" + databaseName + "_" + samplerName + "_" + scalerName + "_" + featureSelectorName + "_" + modelName
    storePickledPipeline(pipeline, pickledPipelineName)
    qualifyPipeline(pipeline, pickledPipelineName, testFeatures, testLabels, databaseName, samplerName, scalerName,
                    featureSelectorName, modelName)


# @app.route('/api/train', methods=['POST'])
def train():
    # content = request.get_json()
    # trainTaskNumber = content.get("trainTaskNumber")
    trainTaskNumber = 1
    connection = getLocalConnection()
    cursor = connection.cursor()
    sqlUseQuery = "use mki"
    cursor.execute(sqlUseQuery)
    sqlSelectQuery = "select * from train_task where task_number = %s"
    n = (trainTaskNumber,)
    cursor.execute(sqlSelectQuery, n)
    result = cursor.fetchone()
    databaseName = result[2]
    samplerName = result[3]
    scalerName = result[4]
    featureSelectorName = result[5]
    modelName = result[6]

    expectedVariance = result[7]
    trainPipeLine(databaseName, samplerName, scalerName, featureSelectorName, modelName, expectedVariance)


if __name__ == '__main__':
    connection = getLocalConnection()
    cursor = connection.cursor()
    sqlUseQuery = "USE mki"
    cursor.execute(sqlUseQuery)
    file = open("SQL create table metrics.txt", "r")
    sqlCreataTableScript = file.read()
    cursor.execute(sqlCreataTableScript)
    connection.commit()
    sqlSelectQuery = "select port_number from config_train_application"
    cursor.execute(sqlSelectQuery)
    result = cursor.fetchone()
    portNumber = result[0]
    connection.commit()
    sqlSelectQuery = "select pickled_objects_folder from config_train_application"
    cursor.execute(sqlSelectQuery)
    result = cursor.fetchone()
    pickledObjectsFolder = result[0]
    cursor.close()
    connection.close()
    # app.run(port=portNumber)
    train()
