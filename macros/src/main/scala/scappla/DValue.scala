package scappla

import scala.language.experimental.macros

trait DValue[X] {

  def v: X

  def dv(v: X): Unit

  def buffer(implicit num: Numeric[X]): Buffer[X] =
    new Buffer[X](this)

  def const: Constant[X] =
    new Constant[X](this)
}

class Buffer[X](upstream: DValue[X])(implicit num: Numeric[X]) extends DValue[X] {

  private var refCount: Int = 1

  private var grad: X =
    num.zero

  override def v: X =
    upstream.v

  override def dv(v: X): Unit = {
    grad = num.plus(grad, v)
  }

  /**
   * basic refcounting; when a function needs a buffer instead of a plain DValue
   * (i.e. when it has potentially more than one use of the value), it further defers
   * the backpropagation of the gradient.  When all references have completed, can the
   * gradient be propagated further backwards.
   */
  override def buffer(implicit num: Numeric[X]) = {
    refCount += 1
    this
  }

  def complete(): Unit = {
    refCount -= 1
    if (refCount == 0) {
      upstream.dv(grad)
    }
  }

}

class Constant[X](upstream: DValue[X]) extends DValue[X] {

  override def v: X = upstream.v

  override def dv(v: X): Unit = {}
}

abstract class LazyDValue[X](private var value: X) extends DValue[X] {

  private var isset: Boolean = false

  final override def v: X = {
    if (!isset) {
      value = _v
      isset = true
    }
    value
  }

  final override def dv(dx: X): Unit = {
    _dv(dx)
    isset = false
  }

  protected def _v: X

  protected def _dv(dx: X): Unit
}

trait DFunction1[From, To] extends (From => To) {

  def apply(in: DValue[From]): DValue[To]
}

trait DFunction2[From1, From2, To] extends ((From1, From2) => To) {

  def apply(in1: DValue[From1], in2: DValue[From2]): DValue[To]
}

class DScalar(self: DValue[Double]) {

  def unary_- = new DValue[Double] {

    override def v: Double =
      -self.v

    override def dv(v: Double): Unit =
      self.dv(-v)
  }

  def -(other: DValue[Double]) = new LazyDValue[Double](0.0) {

    override def _v: Double =
      self.v - other.v

    override def _dv(v: Double): Unit = {
      self.dv(v)
      other.dv(-v)
    }
  }

  def +(other: DValue[Double]) = new LazyDValue[Double](0.0) {

    override def _v: Double =
      self.v + other.v

    override def _dv(v: Double): Unit = {
      self.dv(v)
      other.dv(v)
    }
  }

  def *(other: DValue[Double]) = new LazyDValue[Double](0.0) {

    override def _v: Double =
      self.v * other.v

    override def _dv(v: Double): Unit = {
      self.dv(v * other.v)
      other.dv(v * self.v)
    }
  }

  def /(other: DValue[Double]) = new LazyDValue[Double](0.0) {

    override def _v: Double =
      self.v / other.v

    override def _dv(v: Double): Unit = {
      self.dv(v / other.v)
      other.dv(-v * this.v / other.v)
    }
  }

}

class DVariable(var v: Double) extends DValue[Double] {

  var grad = 0.0

  override def dv(v: Double): Unit = {
    grad += v
  }

}

class DConstant(val v: Double) extends DValue[Double] {

  override def dv(v: Double): Unit = {}
}

object Functions {

  object log extends DFunction1[Double, Double] {

    def apply(x: Double): Double = scala.math.log(x)

    // returned value takes ownership of the reference passed on the stack
    def apply(x: DValue[Double]): DValue[Double] = new LazyDValue[Double](0.0) {

      override def _v: Double =
        scala.math.log(x.v)

      override def _dv(dx: Double): Unit = {
        x.dv(dx / x.v)
      }
    }
  }

  object exp extends DFunction1[Double, Double] {

    override def apply(x: Double): Double = scala.math.exp(x)

    override def apply(x: DValue[Double]): DValue[Double] = new LazyDValue[Double](0.0) {

      override def _v: Double =
        scala.math.exp(x.v)

      override def _dv(dx: Double): Unit = {
        x.dv(dx * v)
      }
    }

  }

  object sigmoid extends DFunction1[Double, Double] {

    import DValue._

    override def apply(x: Double): Double = 1.0 / (1.0 + math.exp(-x))

    override def apply(x: DValue[Double]): DValue[Double] =
      toConstant(1.0) / (exp(-x) + toConstant(1.0))
  }

  object pow extends DFunction2[Double, Double, Double] {

    def apply(base: Double, exp: Double): Double = scala.math.pow(base, exp)

    def apply(base: DValue[Double], exp: DValue[Double]) = new LazyDValue[Double](0.0) {

      override def _v: Double =
        scala.math.pow(base.v, exp.v)

      override def _dv(dx: Double): Unit = {
        val ev = exp.v
        base.dv(dx * ev * scala.math.pow(base.v, ev - 1))
        exp.dv(dx * scala.math.log(base.v) * v)
      }
    }
  }

}

object DValue {

  implicit def toConstant(value: Double) = new DConstant(value)

  implicit def toScalarOps(value: DValue[Double]): DScalar = {
    new DScalar(value)
  }

  implicit val scalarOrdering: Ordering[DValue[Double]] = Ordering.by(_.v)

  implicit val scalarNumeric: Fractional[DValue[Double]] = new Fractional[DValue[Double]] {

    override def plus(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) + y

    override def minus(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) - y

    override def times(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) * y

    override def div(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) / y

    override def negate(x: DValue[Double]): DValue[Double] = -toScalarOps(x)

    override def fromInt(x: Int): DValue[Double] = toConstant(x)

    override def toInt(x: DValue[Double]): Int = x.v.toInt

    override def toLong(x: DValue[Double]): Long = x.v.toLong

    override def toFloat(x: DValue[Double]): Float = x.v.toFloat

    override def toDouble(x: DValue[Double]): Double = x.v

    override def compare(x: DValue[Double], y: DValue[Double]): Int = {
      x.v.compareTo(y.v)
    }

  }
}
