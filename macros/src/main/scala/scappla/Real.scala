package scappla

import scala.language.experimental.macros

trait Real {

  def v: Double

  def dv(v: Double): Unit

  def buffer: Buffer =
    new Buffer(this)

  def const: Constant =
    new Constant(this.v)
}

class Buffer(upstream: Real) extends Real {

  private var refCount: Int = 1

  private var grad: Double =
    0.0

  override def v: Double =
    upstream.v

  override def dv(v: Double): Unit = {
    grad = grad + v
  }

  /**
   * basic refcounting; when a function needs a buffer instead of a plain Real
   * (i.e. when it has potentially more than one use of the value), it further defers
   * the backpropagation of the gradient.  When all references have completed, can the
   * gradient be propagated further backwards.
   */
  override def buffer = {
    refCount += 1
    this
  }

  def complete(): Unit = {
    refCount -= 1
    if (refCount == 0) {
      upstream.dv(grad)
    }
  }

  override def toString: String = s"Buf($upstream)"
}

class Constant(upstream: Double) extends Real {

  override def v: Double = upstream

  override def dv(v: Double): Unit = {}

  override def toString: String = {
    upstream match {
      case d : Double =>
        s"Const(${"%.4f".format(d)})"
      case _ =>
        s"Const($upstream)"
    }
  }
}

abstract class LazyReal(private var value: Double) extends Real {

  private var isset: Boolean = false

  final override def v: Double = {
    if (!isset) {
      value = _v
      isset = true
    }
    value
  }

  final override def dv(dx: Double): Unit = {
    _dv(dx)
    isset = false
  }

  protected def _v: Double

  protected def _dv(dx: Double): Unit

  override def toString: String = s"Lazy($value)"
}

trait DFunction1 extends (Double => Double) {

  def apply(in: Real): Real
}

trait DFunction2 extends ((Double, Double) => Double) {

  def apply(in1: Real, in2: Real): Real
}

case class DNeg(up: Real) extends Real {

  override def v: Double =
    -up.v

  override def dv(v: Double): Unit =
    up.dv(-v)

  override def toString: String = s"-$up"
}

case class DSub(from: Real, what: Real) extends LazyReal(0.0) {

  override def _v: Double =
    from.v - what.v

  override def _dv(v: Double): Unit = {
    from.dv(v)
    what.dv(-v)
  }

  override def toString: String = s"($from - $what)"
}

case class DAdd(a: Real, b: Real) extends LazyReal(0.0) {

  override def _v: Double =
    a.v + b.v

  override def _dv(v: Double): Unit = {
    a.dv(v)
    b.dv(v)
  }

  override def toString: String = s"($a + $b)"
}

case class DMul(a: Real, b: Real) extends LazyReal(0.0) {

  override def _v: Double =
    a.v * b.v

  override def _dv(v: Double): Unit = {
    a.dv(v * b.v)
    b.dv(v * a.v)
  }

  override def toString: String = s"($a * $b)"
}

case class DDiv(num: Real, denom: Real) extends LazyReal(0.0) {

  override def _v: Double =
    num.v / denom.v

  override def _dv(v: Double): Unit = {
    num.dv(v / denom.v)
    denom.dv(-v * this.v / denom.v)
  }

  override def toString: String = s"$num / $denom"
}

class DVariable(var v: Double) extends Real {

  var grad = 0.0

  override def dv(v: Double): Unit = {
    grad += v
  }

}

object Functions {

  object log extends DFunction1 {

    def apply(x: Double): Double = scala.math.log(x)

    // returned value takes ownership of the reference passed on the stack
    def apply(x: Real): Real = new LazyReal(0.0) {

      override def _v: Double =
        scala.math.log(x.v)

      override def _dv(dx: Double): Unit = {
        x.dv(dx / x.v)
      }

      override def toString: String = s"Log($x)"
    }
  }

  object exp extends DFunction1 {

    override def apply(x: Double): Double = scala.math.exp(x)

    override def apply(x: Real): Real = new LazyReal(0.0) {

      override def _v: Double =
        scala.math.exp(x.v)

      override def _dv(dx: Double): Unit = {
        x.dv(dx * v)
      }

      override def toString: String = s"Exp($x)"
    }

  }

  object sigmoid extends DFunction1 {

    import Real._

    override def apply(x: Double): Double = 1.0 / (1.0 + math.exp(-x))

    override def apply(x: Real): Real =
      Real(1.0) / (exp(-x) + Real(1.0))
  }

  object pow extends DFunction2 {

    def apply(base: Double, exp: Double): Double = scala.math.pow(base, exp)

    def apply(base: Real, exp: Real) = new LazyReal(0.0) {

      override def _v: Double =
        scala.math.pow(base.v, exp.v)

      override def _dv(dx: Double): Unit = {
        val ev = exp.v
        base.dv(dx * ev * scala.math.pow(base.v, ev - 1))
        exp.dv(dx * scala.math.log(base.v) * v)
      }

      override def toString: String = s"Pow($base, $exp)"
    }
  }

}

object Real {

  implicit def apply(value: Double): Real = new Constant(value)

  implicit val scalarOrdering: Ordering[Real] = Ordering.by(_.v)

  implicit val scalarNumeric: Fractional[Real] = new Fractional[Real] {

    override def plus(x: Real, y: Real): Real = DAdd(x, y)

    override def minus(x: Real, y: Real): Real = DSub(x, y)

    override def times(x: Real, y: Real): Real = DMul(x, y)

    override def div(x: Real, y: Real): Real = DDiv(x, y)

    override def negate(x: Real): Real = DNeg(x)

    override def fromInt(x: Int): Real = Real(x)

    override def toInt(x: Real): Int = x.v.toInt

    override def toLong(x: Real): Long = x.v.toLong

    override def toFloat(x: Real): Float = x.v.toFloat

    override def toDouble(x: Real): Double = x.v

    override def compare(x: Real, y: Real): Int = {
      x.v.compareTo(y.v)
    }

  }

  implicit def mkNumericOps(lhs: Real): scalarNumeric.FractionalOps =
    new scalarNumeric.FractionalOps(lhs)
}
