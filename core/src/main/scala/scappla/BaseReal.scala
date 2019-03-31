package scappla

import scala.language.experimental.macros

trait BaseReal extends Expr[Double] {

  override def buffer: RealBuffer = RealBuffer(this)

  override def const: RealConstant = RealConstant(v)
}

case class RealBuffer(upstream: Real)
    extends BaseReal with Buffered[Double] {

  private var refCount: Int = 1

  protected var grad: Double = 0.0

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
  override def buffer: RealBuffer = {
    refCount += 1
    this
  }

  def complete(): Unit = {
    refCount -= 1
    if (refCount == 0) {
      upstream.dv(grad)
      grad = 0.0
    }
  }

  override def toString: String = s"Buf($upstream)"
}

case class RealConstant(override val v: Double) extends Constant[Double](v) with BaseReal {

  override def toString: String = {
    s"Const(${"%.4f".format(v)})"
  }
}

abstract class LazyReal(private var value: Double) extends BaseReal {

  private var isset: Boolean = false
  private var completed = 0

  final override def v: Double = {
    if (!isset) {
      completed = 0
      value = _v
      isset = true
    }
    value
  }

  final override def dv(dx: Double): Unit = {
    completed += 1
    if (completed > 1) {
      println(s"  Completed ${completed} times")
    }
    _dv(dx)
    isset = false
  }

  protected def _v: Double

  protected def _dv(dx: Double): Unit

  override def toString: String = s"Lazy($value)"
}

case class DNeg(up: Real) extends BaseReal {

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

case class DDiv(numer: Real, denom: Real) extends LazyReal(0.0) {

  override def _v: Double =
    numer.v / denom.v

  override def _dv(v: Double): Unit = {
    numer.dv(v / denom.v)
    denom.dv(-v * this.v / denom.v)
  }

  override def toString: String = s"$numer / $denom"
}

object Real {

  implicit def apply(value: Double): BaseReal = RealConstant(value)

  implicit val scalarOrdering: Ordering[Real] = Ordering.by(_.v)
}
