
# -*- coding: utf-8 -*-

# Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
# Date created: 08/17/2018

# This script predicts whether the client will subscribe (Yes/No) to a term deposit from direct marketing campaigns (phone calls) of a Portuguese banking institution.
usage: python3 SparkML_MLib_Classification.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('bank.csv', header = True, inferSchema = True)
df.printSchema()



import pandas as pd
pd.DataFrame(df.take(5), columns=df.columns).transpose()

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']

# CHECK FOR HIHGLY CORRELATED VARIABLES AND REMOVE IF EXIST

numeric_data = df.select(numeric_features).toPandas()
axs = pd.scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# NO HIGHLY COORELATED VARIABLE EXIST, SO KEEP ALL NUMERIC VARIABLES

df = df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')
cols = df.columns
df.printSchema()


# PREPARE DATA FOR ML USING CATEGORY INDEXING, ONE-HOT ENCODING, VECTORASSEMBLER (FEATURE MERGER) 

# TRANSFORM CATEGORICAL VARIABLE USING ONE-HOT ENCODING FOR ML COMPATIBILITY
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []

for categoricalCol in categoricalColumns:

   # INDEX EACH CATEGORICAL VARIABLE USING STRINGINDEXER
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    
   # CONVERT THE INDEXED CATEGORICAL VARIABLES INTO ONE-HOT ENCODED VARIBALES
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder] 
    
# ENCODE LABELS TO LABEL INDICES USING STRING INDEXER
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]

# PREPARE NUMERICAL VARIABLES
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

#PREPARE TO MERGE TRANSFORMED CATEGORICAL VARIBALES WITH NUMERICAL VARIABLES

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# PIPELINE TO CHAIN TRANSFORMERS AND ESTIMATOR
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

# SPLIT DATA INTO TRAIN AND TEST RANDOMLY
train_set, test_set = df.randomSplit([0.7, 0.3], seed = 2000)
print("Training Dataset Count: " + str(train_set.count()))
print("Test Dataset Count: " + str(test_set.count()))

# USE LOGISTIC REGRESSION
lr_clf = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr_clf.fit(train_set)

#PLOT ROC CURVE ON TRAINING SET
trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set AUC: ' + str(trainingSummary.areaUnderROC))

#PLOT PR CURVE
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# PREDCIT TEST SET
predictions = lrModel.transform(test_set)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

# TEST AUC-ROC
lr_evaluator = BinaryClassificationEvaluator()
print('Test AUC', lr_evaluator.evaluate(predictions))


# RANDOM FOREST CLASSIFIER
rf_clf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf_clf.fit(train_set)
predictions = rfModel.transform(test_set)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

rf_evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(rf_evaluator.evaluate(predictions, {rf_evaluator.metricName: "areaUnderROC"})))

sys.exit()
