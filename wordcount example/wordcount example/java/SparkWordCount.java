import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;

import java.util.Arrays;
import java.util.List;
import java.util.Collection;

public final class SparkWordCount {

  public static void main(String[] args) throws Exception {

    //create Spark context with Spark configuration
    JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("wordcount")); 

    //set the input file
    JavaRDD<String> textFile = sc.textFile("in.txt");

    //word count process
    JavaPairRDD<String, Integer> counts = textFile
    .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
    .mapToPair(word -> new Tuple2<>(word, 1))
    .reduceByKey((a, b) -> a + b);

    //set the output folder
    counts.saveAsTextFile("outfile");
    //stop spark
  }
}
