package optimization

import org.apache.spark.sql.SparkSession

object BayesianOptimization {

    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder
            .appName("automl")
            .master("local[*]")
            .getOrCreate()


    }
}
