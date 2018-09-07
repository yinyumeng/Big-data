import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
# Add the filter to remove non-alphabetic characters
words = lines.flatMap(lambda l: re.split(r'[^\w]+',l)).filter(lambda w: len(w)>0)
# in map, only use the lower case of first characters for words as the key
pairs = words.map(lambda c: (c[0].lower(), 1))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
counts.saveAsTextFile(sys.argv[2])
sc.stop()
