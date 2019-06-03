package scappla.tensor

import scappla.Elemwise
import scala.util.Random
import scala.collection.immutable.SortedSet
import scala.collection.mutable

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

case class SparseTensor(shape: Seq[Int], var data: Array[Float], var indices: Array[Array[Int]])

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

    override def logistic(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        val adi = ad(i)
        result(i) = if (adi > 0.0) {
          1.0f / (1.0 + scala.math.exp(-adi)).toFloat
        } else {
          val ea = scala.math.exp(adi).toFloat
          ea / (1.0f + ea)
        }
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def softplus(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.log1p(scala.math.exp(ad(i))).toFloat
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

  implicit val sparseIndexOrder: Ordering[Array[Int]] = new Ordering[Array[Int]] {

    def compare(x: Array[Int], y: Array[Int]): Int = {
      x.zip(y).foldLeft(0) { case (cur, (xi, yi)) =>
        cur match {
          case 0 => Integer.compare(xi, yi)
          case _ => cur
        }
      }
    }
  }

  implicit val sparseOps = new TensorData[SparseTensor] {

    def plus(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { _ + bF.getOrElse(0f) }.orElse(bF)
      })
    }

    def minus(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { _ - bF.getOrElse(0f) }.orElse(bF.map { -_ })
      })
    }

    def times(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.flatMap { af => bF.map { af * _ }}
      })
    }

    def div(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { af => af / bF.getOrElse(0f) }
      })
    }

    def pow(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { af => bF.map { scala.math.pow(af, _).toFloat }.getOrElse(1f) }
      })
    }

    private def binary(
      a: SparseTensor,
      b: SparseTensor,
      fn: (Option[Float], Option[Float]) => Option[Float]
    ): SparseTensor = {

      val aIter = a.indices.zip(a.data).iterator
      val bIter = b.indices.zip(b.data).iterator

      val resultIter = new Iterator[(Array[Int], Float)] {
        var aCursor: Option[(Array[Int], Float)] = None
        var bCursor: Option[(Array[Int], Float)] = None

        var out: Option[(Array[Int], Float)] = None

        def hasNext: Boolean = {
          step()
          out.isDefined
        }

        def next(): (Array[Int], Float) = {
          step()
          out.get
        }

        private def step(): Unit = {
          while (!out.isDefined) {
            if (aCursor.isEmpty && aIter.hasNext) {
              aCursor = Some(aIter.next())
            }
            if (bCursor.isEmpty && bIter.hasNext) {
              bCursor = Some(bIter.next())
            }
            if (aCursor.isDefined && bCursor.isDefined) {
              val cmp = sparseIndexOrder.compare(aCursor.get._1, bCursor.get._1)
              if (cmp < 0) {
                out = fn(aCursor.map { _._2 }, None).map { (aCursor.get._1, _) }
                aCursor = None
              } else if (cmp > 0) {
                out = fn(None, bCursor.map { _._2 }).map { (bCursor.get._1, _) }
                bCursor = None
              } else {
                out = fn(aCursor.map{ _._2 }, bCursor.map{ _._2}).map { (aCursor.get._1, _) }
                aCursor = None
                bCursor = None
              }
            } else if (aCursor.isDefined) {
              out = fn(aCursor.map{ _._2 }, None).map { (aCursor.get._1, _) }
              aCursor = None
            } else if (bCursor.isDefined) {
              out = fn(None, bCursor.map{ _._2 }).map { (bCursor.get._1, _) }
              bCursor = None
            } else {
              return
            }
          }
        }
      }

      val zipped = resultIter.toArray
      SparseTensor(a.shape, zipped.map { _._2 }, zipped.map { _._1 })
    }

    def negate(a: SparseTensor): SparseTensor = {
      unary(a, -_)
    }

    def sqrt(a: SparseTensor): SparseTensor = {
      unary(a, scala.math.sqrt(_).toFloat)
    }

    def log(a: SparseTensor): SparseTensor = {
      unary(a, scala.math.log(_).toFloat)
    }

    def exp(a: SparseTensor): SparseTensor = {
      unary(a, scala.math.exp(_).toFloat)
    }

    def logistic(a: SparseTensor): SparseTensor = {
      unary(a, { x => 1f / (1f + scala.math.exp(-x).toFloat)})
    }

    def softplus(a: SparseTensor): SparseTensor = {
      unary(a, { x => scala.math.log1p(scala.math.exp(x)).toFloat})
    }

    private def unary(in: SparseTensor, fn: Float => Float): SparseTensor = {
      SparseTensor(in.shape, in.data.map(fn), in.indices)
    }

    def sumAll(a: SparseTensor): Float = {
      a.data.sum
    }

    // (de)constructing values

    def fill(value: Float, dims: Int*): SparseTensor = {
      SparseTensor(dims.toSeq, Array.fill(dims.product)(value), fullRank(dims))
    }

    def gaussian(dims: Int*): SparseTensor = {
      SparseTensor(dims.toSeq, Array.fill(dims.product)(Random.nextGaussian().toFloat), fullRank(dims))
    }

    private def fullRank(dims: Seq[Int]): Array[Array[Int]] = {
      dims.toSeq.foldLeft(Seq.empty[Array[Int]]) {
        case (curIndices, dim) =>
          curIndices.flatMap { curIndex =>
            (0 until dim).map { i => curIndex :+ i }
          }
      }.toArray
    }

    def count(d: SparseTensor, cond: Condition): Int = {
      cond match {
        case GreaterThan(value) if value > 0f => 
          d.data.count(_ > value)
        case GreaterThan(value) if value <= 0f => 
          d.shape.product - d.data.length + d.data.count(_ > value)
      }
    }

    def get(d: SparseTensor, indices: Int*): Float = {
      val index = d.indices.indexOf(indices.toArray)
      if (index >= 0) {
        d.data(index)
      } else {
        0f
      }
    }

    def put(d: SparseTensor, value: Float, indices: Int*): Unit = {
      val index = d.indices.indexOf(indices.toArray)
      if (index >= 0) {
        d.data(index) = value
        d
      } else {
        val spliceAt = d.indices.count(sparseIndexOrder.compare(_, indices.toArray) < 0)

        val newData = Array.ofDim[Float](d.data.length + 1)
        d.data.copyToArray(newData, 0, spliceAt)
        d.data(spliceAt) = value
        d.data.copyToArray(newData, spliceAt + 1, d.data.length - spliceAt)

        val newIndices = Array.ofDim[Array[Int]](d.indices.length + 1)
        d.indices.copyToArray(newIndices, 0, spliceAt)
        d.indices(spliceAt) = indices.toArray
        d.indices.copyToArray(newIndices, spliceAt + 1, d.data.length - spliceAt)
        SparseTensor(d.shape, newData, newIndices)
      }
    }

    def imax(d: SparseTensor): Seq[Int] = {
      d.data.zip(d.indices).maxBy(_._1)._2.toSeq 
    }

    def cumsum(a: SparseTensor, dim: Int): SparseTensor = ???

    // shape-affecting operations

    def sum(a: SparseTensor, dim: Int): SparseTensor = {
      val sums = a.data.zip(a.indices).groupBy { case (value, indices) => 
        val ic = indices.clone()
        ic(dim) = 0
        ic
      }.mapValues { _.map(_._1).sum }
    }

    def broadcast(a: SparseTensor, dimIndex: Int, dimSize: Int): SparseTensor = ???

    def tensordot(a: SparseTensor, b: SparseTensor, ab: List[(Int, Int)], bc: List[(Int, Int)], ca: List[(Int, Int)]): SparseTensor = {
      val bByShared: Map[Map[Int, Int], Seq[(Map[Int, Int], Float)]] = {
        val partitioned = mutable.HashMap[Map[Int, Int], Seq[(Map[Int, Int], Float)]]()
            .withDefaultValue(Seq.empty)
        b.indices.zip(b.data).foreach { case (indices, value) =>
            val aIndex = ab.map { case (aDim, bDim) =>
                aDim -> indices(bDim)
            }.toMap
            val cIndex = bc.map { case (bDim, cDim) =>
                cDim -> indices(bDim)
            }.toMap
            partitioned(aIndex) = partitioned(aIndex) :+ (cIndex, value)
        }
        partitioned.toMap
      }

    }
  }

}
