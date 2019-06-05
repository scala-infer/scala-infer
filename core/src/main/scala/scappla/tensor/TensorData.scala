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

case class SparseTensor(shape: Seq[Int], var data: Array[Float], var coordinates: Seq[Array[Int]])

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

  implicit val sparseIndexOrder: Ordering[Seq[Int]] = new Ordering[Seq[Int]] {

    def compare(x: Seq[Int], y: Seq[Int]): Int = {
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
      assert(a.coordinates.size == b.coordinates.size)

      val aIter = a.data.zipWithIndex.iterator
      val bIter = b.data.zipWithIndex.iterator

      val resultIter = new Iterator[(Array[Int], Float)] {
        var aCursor: Option[(Float, Array[Int])] = None
        var bCursor: Option[(Float, Array[Int])] = None

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
              val (aVal, aIndex) = aIter.next()
              aCursor = Some((aVal, a.coordinates.map { _(aIndex) }.toArray))
            }
            if (bCursor.isEmpty && bIter.hasNext) {
              val (bVal, bIndex) = bIter.next()
              bCursor = Some((bVal, b.coordinates.map { _(bIndex) }.toArray))
            }
            if (aCursor.isDefined && bCursor.isDefined) {
              val cmp = sparseIndexOrder.compare(aCursor.get._2, bCursor.get._2)
              if (cmp < 0) {
                out = fn(aCursor.map { _._1 }, None).map { (aCursor.get._2, _) }
                aCursor = None
              } else if (cmp > 0) {
                out = fn(None, bCursor.map { _._1 }).map { (bCursor.get._2, _) }
                bCursor = None
              } else {
                out = fn(aCursor.map{ _._1 }, bCursor.map{ _._1}).map { (aCursor.get._2, _) }
                aCursor = None
                bCursor = None
              }
            } else if (aCursor.isDefined) {
              out = fn(aCursor.map{ _._1 }, None).map { (aCursor.get._2, _) }
              aCursor = None
            } else if (bCursor.isDefined) {
              out = fn(None, bCursor.map{ _._1 }).map { (bCursor.get._2, _) }
              bCursor = None
            } else {
              return
            }
          }
        }
      }

      val zipped = resultIter.toArray
      SparseTensor(
        a.shape,
        zipped.map { _._2 },
        for { coordinate <- a.coordinates.indices } yield  {
          zipped.map { _._1(coordinate) }
        }
      )
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

    private def fullRank(dims: Seq[Int]): Seq[Array[Int]] = {
      val totalSize = dims.product
      dims.scanLeft((1, totalSize)) { case ((prevSize, prevStride), dimSize) =>
        val size = prevSize * dimSize
        val stride = prevStride / dimSize
        (size, stride)
      }.zip(dims).map { case ((size, stride), dimSize) =>
        val result = Array.ofDim[Int](totalSize)
        for {
          i <- 0 until size
          j <- 0 until stride
        } {
          val index = i * stride + j
          result(index) = i % dimSize
        }
        result
      }
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
      val (idx, idxp) = d.coordinates.zip(indices.toSeq).foldLeft((0, d.data.length)) {
        case ((start, end), (coordinates, toFind)) =>
          Range(start, end).foldLeft((end, end)) {
            case ((s, e), i) =>
              if (coordinates(i) < toFind) {
                (i + 1, i + 1)
              } else if (coordinates(i) == toFind) {
                (s, i + 1)
              } else {
                (s, e)
              }
            }
        }
      if (idxp == idx + 1) {
        d.data(idxp)
      } else {
        0f
      }
    }

    def put(d: SparseTensor, value: Float, indices: Int*): Unit = ???

    def imax(d: SparseTensor): Seq[Int] = {
      val index = d.data.zipWithIndex.maxBy(_._1)._2
      d.coordinates.map { _(index) }
    }

    def cumsum(a: SparseTensor, dim: Int): SparseTensor = ???

    // shape-affecting operations

    def sum(a: SparseTensor, dim: Int): SparseTensor = ???

    def broadcast(a: SparseTensor, dimIndex: Int, dimSize: Int): SparseTensor = ???

    def tensordot(a: SparseTensor, b: SparseTensor, ab: List[(Int, Int)], bc: List[(Int, Int)], ca: List[(Int, Int)]): SparseTensor = {
      val newShape: Seq[Int] = (bc.map {
          case (bi, ci) => (ci, b.shape(bi))
        } ++ ca.map {
          case (ci, ai) => (ci, a.shape(ai))
        }).sortBy { _._1 }
          .map { _._2 }

      val bByShared: Map[Array[Int], Seq[(Array[Int], Float)]] = {
        // map of aIndex -> Seq[(cIndex, value)]
        val partitioned = mutable.HashMap[Array[Int], Seq[(Array[Int], Float)]]()
            .withDefaultValue(Seq.empty)

        b.data.zipWithIndex.foreach { case (value, index) =>
          val bIndices = b.coordinates.map { _(index) }
            val aIndices = ab.map { case (_, bDim) =>
              bIndices(bDim)
            }.toArray
            val cIndices = bc.map { case (bDim, _) =>
              bIndices(bDim)
            }.toArray
            partitioned(aIndices) :+= (cIndices, value)
        }
        partitioned.toMap
      }

      val products = a.data.toSeq.zipWithIndex.flatMap { case (aValue, aIndex) =>
        val aIndices = a.coordinates.map { _(aIndex) }
        val abIndices = ab.map { case (aDim, _) =>
          aIndices(aDim)
        }.toArray
        val acIndices: Seq[(Int, Int)] = ca.map { case (cDim, aDim) =>
          cDim -> aIndices(aDim)
        }.toSeq
        for { (bcIndices, bValue) <- bByShared(abIndices) } yield {
          val fullBc = bcIndices.toSeq.zip(bc).map { case (idx, (_, cDim)) =>
            cDim -> idx
          }
          val cIndex = (acIndices ++ fullBc).sortBy { _._1 }.map { _._2 }
          cIndex -> aValue * bValue
        }
      }

      val out = products
        .groupBy(_._1)
        .mapValues(_.map { _._2 }.sum)
        .toSeq
        .sortBy{ _._1 }(sparseIndexOrder)


      SparseTensor(
        newShape,
        out.map { _._2 },

      )
    }
  }

}
