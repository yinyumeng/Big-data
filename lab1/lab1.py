
import re
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
import time 
import math

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

start = time.time()

df = spark.read.json(sys.argv[1])

N = df.count()

# read from stopwords.txt and split to get stopword list
f = open('stopwords.txt', 'r+')
stopwords = re.split(r"[^\w]+", f.read())
stopwords = [x for x in stopwords if x]
f.close()

# read from query.txt and split to get stopword list
f = open('query.txt', 'r+')
query = re.split(r"[^\w]+", f.read())
f.close()

# Calculate the root of sum of square of query (where all word are 1)
q_sqrt = math.sqrt(len(query))

# Combine reviewerID and asin to get documentID
df = df.withColumn('documentID', 
                    F.concat(F.col('reviewerID'),F.lit(''), F.col('asin')))

# Combine reviewText and summary to get document
df = df.withColumn('document', 
                    F.concat(F.col('reviewText'),F.lit(' '), F.col('summary')))
columns_to_drop = ["reviewerID","asin","reviewerName","helpful","reviewText","overall","summary","unixReviewTime","reviewTime"]
df = df.drop(*columns_to_drop)


# Create RDD (documentID, document)
rdd = df.rdd.map(lambda x: (x[0],x[1]))
# flatmap to get (documentID, word)
rdd = rdd.flatMapValues(lambda  doc:  re.split('[^A-Za-z]', doc.lower()))
# remove stopwords and empty word
rdd = rdd.filter(lambda x: (len(x[1])>0) and (x[1] not in stopwords))
# ((documentID, word), 1)
pairs = rdd.map(lambda x: (x, 1))

# get result of TF using reduceByKey 
# in format ((documentID, word), TF)
TF = pairs.reduceByKey(lambda n1, n2: n1 + n2)

# get result of DF with TF using following method 
# 1. map from ((documentID, word), TF) to (word, ((documentID, TF), 1))
# 2. reduceByKey from (word, ((documentID, TF), 1)) to (word, ([(documentID_1, TF_1), (documentID_2, TF_2) ... , (documentID_k, TF_k)], DF)). k is DF
TF_DF = TF.map(lambda x: (x[0][1],([(x[0][0], x[1])], 1))).reduceByKey(lambda n1, n2: (n1[0]+n2[0], n1[1]+n2[1]))

# From TF and DF, we can get the result of TF_IDF using following method
# 1. flatMap for (word, ([(documentID_1, TF_1), (documentID_2, TF_2) ... , (documentID_k, TF_k)], DF)): 
#    loop through the list of (documentID, TF) and calculate the result of (1+log(TF))*log(N/DF)
# 2. TF_IDF in format (documentID, word, tf_idf)
TF_IDF = TF_DF.flatMap(lambda x: [(id_tf[0],x[0],(1 + math.log(id_tf[1],10))*(math.log((N/float(x[1][1])),10))) for id_tf in x[1][0]])

# Calculate the result of sum of squares of all tf_idf for each documentID
# 1. map from (documentID, word, tf_idf) to (documentID, ([(word_1, tf_idf_1)], tf_idf**2))
# 2. reduce to (documentID, (list of (word, tf_idf), sum of tf_idf**2 for documentID))
Nom_TF_IDF = TF_IDF.map(lambda x: (x[0], ([(x[1],x[2])], x[2]**2))).reduceByKey(lambda n1, n2: (n1[0]+n2[0], n1[1]+n2[1]))

# Calculated the normalized TF_IDF
# 1. flatMap for (documentID, (list of (word, tf_idf), sum of tf_idf**2 for documentID)):
#    loop through the list of (word, tf_idf) and calculate the result of tf_idf/(square root of sum of tf_idf**2 for documentID)
# 2. Nom_TF_IDF in format (documentID, word, nor_tf_idf)
Nom_TF_IDF = Nom_TF_IDF.flatMap(lambda x: [(x[0], word_tf_idf[0], word_tf_idf[1]/math.sqrt(x[1][1])) for word_tf_idf in x[1][0]])

# Calculated the relevance of document to query
# 1. filter from Nom_TF_IDF where the word doesn't exist in query
# 2. map to get (documentID, nor_tf_idf/(norm of query))
# 3. reduceByKey to get (documentID, relevance score)
# 4. sort using the relevance score
Relevance = Nom_TF_IDF.filter(lambda x: x[1] in query).map(lambda x: (x[0],x[2]/q_sqrt)).reduceByKey(lambda n1, n2: n1+n2).sortBy(lambda x: x[1], ascending=False)
top_20 = Relevance.collect()[0:20]
end = time.time()

with open(sys.argv[2], 'w') as f:
    for item in top_20:
        print(item)
        f.write("%s\n" % str(item))

sc.stop()

