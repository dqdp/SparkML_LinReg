package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.Assertion
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.001

  lazy val df: DataFrame = LinearRegressionTest._df
  lazy val weights: DenseVector[Double] = LinearRegressionTest._weights
  lazy val bias: Double = LinearRegressionTest._bias
  lazy val y: DenseVector[Double] = LinearRegressionTest._y


  "Model" should "evaluate predictions" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(
        DenseVector.vertcat(weights, DenseVector[Double](bias))
      )
    ).setInputCol("features")
     .setOutputCol("target")

    validateModel(model, model.transform(df))
  }

  "Estimator" should "produce functional model" in {
    val estimator = new LinearRegression()
      .setInputCol("features").setOutputCol("target")
    val model = estimator.fit(df)

    var idx = 0;
    for( idx <- 0 to model.weights.size - 2){
         model.weights(idx) should be(weights(idx) +- delta);
      }
    model.weights(model.weights.size - 1) should be(bias +- delta);

    validateModel(model, model.transform(df))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("target")
    ))

    val tempDir = Files.createTempDir()

    pipeline.write.overwrite().save(tempDir.getAbsolutePath)

    val model = Pipeline.load(tempDir.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]

    var idx = 0;
    for( idx <- 0 to model.weights.size - 2){
         model.weights(idx) should be(weights(idx) +- delta);
      }
    model.weights(model.weights.size - 1) should be(bias +- delta);
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("target")
    ))
    val model = pipeline.fit(df)
    val tempDir = Files.createTempDir()

    model.write.overwrite().save(tempDir.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tempDir.getAbsolutePath)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(df))
  }

  private def validateModel(model: LinearRegressionModel, df: DataFrame): Unit = {
    val vectors: Array[Double] = df.collect().map(_.getAs[Double](1))

    vectors.length should be(10000)
    for (i <- vectors.indices)
      vectors(i) should be(y(i) +- delta)
  }
}

object LinearRegressionTest extends WithSpark {

  import sqlc.implicits._

  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](10000, 5)
  lazy val _weights: DenseVector[Double] = DenseVector(0.6, 1.2, -1.3, 0.4, -1.9)
  lazy val _bias: Double = 1.2
  lazy val _y: DenseVector[Double] = _X * _weights + _bias

  lazy val mat: DenseMatrix[Double] = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)

  lazy val df: DataFrame = mat(*, ::).iterator
    .map(x => (x(0), x(1), x(2), x(3), x(4), x(5)))
    .toSeq
    .toDF("x0", "x1", "x2", "x3", "x4", "target")

  lazy val vector_assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("x0", "x1", "x2", "x3", "x4"))
    .setOutputCol("features")

  lazy val _df: DataFrame = vector_assembler
    .transform(df)
    .select("features", "target")
}
