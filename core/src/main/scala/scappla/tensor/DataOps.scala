package scappla.tensor

import scappla.Elemwise
import scala.util.Random

sealed trait Condition
case class GreaterThan(value: Float) extends Condition

trait TensorData[D] extends Elemwise[D] {

  // (de)constructing values

  def fill(value: Float, dims: Int*): D

  def gaussian(shape: Int*): D

  def count(d: D, cond: Condition): Int

  def get(d: D, indices: Int*): Float

  def put(d: D, value: Float, indices: Int*): Unit

  def imax(d: D): Seq[Int]

  def cumsum(a: D, dim: Int): D

  // shape-affecting operations

  def sum(a: D, dim: Int): D

  def broadcast(a: D, dimIndex: Int, dimSize: Int): D

  def tensordot(a: D, b: D, ab: List[(Int, Int)], bc: List[(Int, Int)], ca: List[(Int, Int)]): D
}

case class ArrayTensor(shape: Seq[Int], data: Array[Float])

object TensorData {

  implicit val arrayOps = new TensorData[ArrayTensor] {

    override def fill(value: Float, shape: Int*): ArrayTensor =
      ArrayTensor(shape.toSeq, Array.fill(shape.product)(value))

    override def gaussian(shape: Int*): ArrayTensor = {
      ArrayTensor(shape.toSeq, Array.fill(shape.product)(Random.nextGaussian().toFloat))
    }

    override def count(d: ArrayTensor, cond: Condition): Int = {
      cond match {
        case GreaterThan(value) =>
          d.data.count(_ > value)
      }
    }

    override def get(d: ArrayTensor, indices: Int*): Float = {
      val (shape, data) = (d.shape, d.data)
      val index = shape.zip(indices).foldLeft(0) {
        case (cum, (dimShape, dimIdx)) =>
          cum * dimShape + dimIdx
      }
      data(index)
    }

    override def put(d: ArrayTensor, value: Float, indices: Int*): Unit = {
      val (shape, data) = (d.shape, d.data)
      val index = shape.zip(indices).foldLeft(0) {
        case (cum, (dimShape, dimIdx)) =>
          cum * dimShape + dimIdx
      }
      data(index) = value
    }

    override def imax(d: ArrayTensor): Seq[Int] = {
      val (_, index) = d.data.zipWithIndex.maxBy(_._1)
      val accs = d.shape.scanRight(1) {
        case (dimSize, acc) => dimSize * acc
      }
      for {
        (dimProd, dimSize) <- accs.drop(1).zip(d.shape)
      } yield {
        (index / dimProd) % dimSize
      }
    }

    override def plus(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) + bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def minus(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) - bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def times(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      if(a.shape != b.shape) {
        println(s"SHAPE A: ${a.shape}, B: ${b.shape}")
      }
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) * bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def div(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) / bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def pow(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.pow(ad(i), bd(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def negate(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = -ad(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def sqrt(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.sqrt(ad(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def log(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.log(ad(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def exp(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.exp(ad(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def sigmoid(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = (1.0 / (1.0 + scala.math.exp(-ad(i)))).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def sum(a: ArrayTensor, dimIndex: Int): ArrayTensor = {
      val (shape, data) = (a.shape, a.data)
      val dimSize = shape(dimIndex)
      val outerShape = shape.take(dimIndex)
      val innerShape = shape.takeRight(shape.size - dimIndex - 1)
      val newShape = outerShape ++ innerShape

      val totalSize = newShape.product
      val innerSize = innerShape.product
      val output = Array.fill(totalSize)(0F)
      for {outer <- Range(0, outerShape.product)} {
        val baseOutputIdx = outer * innerSize
        for {i <- Range(0, innerSize)} {
          val outputIdx = i + baseOutputIdx
          var value = 0f
          for {d <- Range(0, dimSize)} {
            val inputIdx = i + d * innerSize + outer * innerSize * dimSize
            value += data(inputIdx)
          }
          output(outputIdx) += value
        }
      }
      ArrayTensor(newShape, output)
    }

    override def cumsum(a: ArrayTensor, dimIndex: Int): ArrayTensor = {
      val (shape, data) = (a.shape, a.data)
      val dimSize = shape(dimIndex)
      val outerShape = shape.take(dimIndex)
      val innerShape = shape.takeRight(shape.size - dimIndex - 1)

      val totalSize = shape.product
      val innerSize = innerShape.product
      val output = Array.fill(totalSize)(0F)
      for {outer <- Range(0, outerShape.product)} {
        for {i <- Range(0, innerSize)} {
          var value = 0f
          for {d <- Range(0, dimSize)} {
            val inputIdx = i + innerSize * (d + outer * dimSize)
            value += data(inputIdx)
            output(inputIdx) = value
          }
        }
      }
      ArrayTensor(shape, output)
    }

    override def broadcast(a: ArrayTensor, dimIndex: Int, dimSize: Int): ArrayTensor = {
      val (shape, data) = (a.shape, a.data)
      val outerShape = shape.take(dimIndex)
      val innerShape = shape.takeRight(shape.size - dimIndex)
      val newShape = (outerShape :+ dimSize) ++ innerShape
      val output = Array.ofDim[Float](newShape.product)

      val innerSize = innerShape.product
      for {outer <- Range(0, outerShape.product)} {
        val baseInput = outer * innerSize
        for {i <- Range(0, innerSize)} {
          val inputIdx = i + baseInput
          val value = data(inputIdx)
          for {d <- Range(0, dimSize)} {
            val outputIdx = i + d * innerSize + outer * innerSize * dimSize
            output(outputIdx) = value
          }
        }
      }
      ArrayTensor(newShape, output)
    }

    override def sumAll(a: ArrayTensor): Float = {
      val data = a.data
      val len = data.length
      var result = 0f
      var i = 0
      while (i < len) {
        result += data(i)
        i += 1
      }
      result
    }

    override def tensordot(a: ArrayTensor, b: ArrayTensor, ab: List[(Int, Int)], bc: List[(Int, Int)], ca: List[(Int, Int)]): ArrayTensor = {
//      println(s"AB: ${ab}")
//      println(s"BC: ${bc}")
//      println(s"CA: ${ca}")
      val newShape: Seq[Int] = (bc.map {
          case (bi, ci) => (ci, b.shape(bi))
        } ++ ca.map {
          case (ci, ai) => (ci, a.shape(ai))
        }).sortBy { _._1 }
          .map { _._2 }

      val aDims = a.shape.scanRight(1) {
        case (dimSize, stride) => dimSize * stride
      }.drop(1).zipWithIndex.map {
        _.swap
      }.toMap
//      println(s"A DIMS: ${aDims}")

      val bDims = b.shape.scanRight(1) {
        case (dimSize, stride) => dimSize * stride
      }.drop(1).zipWithIndex.map {
        _.swap
      }.toMap
//      println(s"B DIMS: ${bDims}")

      val cDims = newShape.scanRight(1) {
        case (dimSize, stride) => dimSize * stride
      }.drop(1).zipWithIndex.map {
        _.swap
      }.toMap
//      println(s"C DIMS: ${cDims}")

      val abDimensions = ab.map {
        case (ai, bi) =>
          ((aDims(ai), bDims(bi)), a.shape(ai))
      }
//      println(s"AB DIMS: ${abDimensions}")
      val caDimensions = ca.map {
        case (ci, ai) =>
          ((cDims(ci), aDims(ai)), a.shape(ai))
      }
//      println(s"CA DIMS: ${caDimensions}")
      val bcDimensions = bc.map {
        case (bi, ci) =>
          ((bDims(bi), cDims(ci)), b.shape(bi))
      }
//      println(s"BC DIMS: ${bcDimensions}")

      // take pairs of matching strides and dimension size, generate indices
      def nestedIter(parts: Seq[((Int, Int), Int)]): Iterator[(Int, Int)] = {
        parts.foldLeft(Iterator.single((0, 0))) {
          case (current, ((strideLeft, strideRight), dimSize)) =>
            current.flatMap { case (indexLeft, indexRight) =>
              (0 until dimSize).iterator.map { i =>
                (indexLeft + i * strideLeft, indexRight + i * strideRight)
              }
            }
        }
      }

      val result = Array.ofDim[Float](newShape.product)
      for { (cBaseIndex, aBaseIndex) <- nestedIter(caDimensions)} {
        for { (aRelIndex, bBaseIndex) <- nestedIter(abDimensions)} {
          val aIndex: Int = aBaseIndex + aRelIndex
//          println(s"A: $aBaseIndex + $aRelIndex => $aIndex")

          val aValue = a.data(aIndex)
          for {(bRelIndex, cRelIndex) <- nestedIter(bcDimensions)} {
            val bIndex: Int = bBaseIndex + bRelIndex
//            println(s"B: $bBaseIndex + $bRelIndex => $bIndex")

            val cIndex: Int = cBaseIndex + cRelIndex
//            println(s"C: $cBaseIndex + $cRelIndex => $cIndex")

            val bValue = b.data(bIndex)
            result(cIndex) += aValue * bValue
          }
        }
      }

      ArrayTensor(newShape, result)
    }
  }

}
