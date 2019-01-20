package scappla

trait Expr[X] {

  def v: X

  def dv(v: X): Unit

  def unary_-()(implicit numE: Numeric[Expr[X]]): Expr[X] =
    numE.negate(this)

  def +(other: Expr[X])(implicit numE: Numeric[Expr[X]]): Expr[X] =
    numE.plus(this, other)

  def -(other: Expr[X])(implicit numE: Numeric[Expr[X]]): Expr[X] =
    numE.minus(this, other)

  def *(other: Expr[X])(implicit numE: Numeric[Expr[X]]): Expr[X] =
    numE.times(this, other)

  def /(other: Expr[X])(implicit numE: Fractional[Expr[X]]): Expr[X] =
    numE.div(this, other)

  def buffer: Buffered[X] =
    NoopBuffer(this)

  def const: Constant[X] =
    new Constant(this.v)
}

object Expr {

  implicit def fromDouble(value: Double) = Real(value)

  implicit def mkNumericOps[X](value: Expr[X])(implicit num: Fractional[Expr[X]]) =
    num.mkNumericOps(value)
}

trait Buffered[X] extends Expr[X] with Completeable

case class NoopBuffer[X](upstream: Expr[X]) extends Buffered[X] {

  override def v: X = upstream.v

  override def dv(v: X): Unit = upstream.dv(v)

  override def buffer: Buffered[X] = this

  override def complete(): Unit = {}
}

class Constant[X](val v: X) extends Buffered[X] {

  override def dv(v: X): Unit = {}

  override def buffer: Buffered[X] = this

  override def complete(): Unit = {}
}

trait ShapeOf[D, Shape] {

  def apply(data: D): Shape
}

trait DoubleShape extends ShapeOf[Double, DoubleShape]

object DoubleShape extends DoubleShape {

  override def apply(data: Double): DoubleShape = DoubleShape
}

object ShapeOf {

  implicit val doubleShape: ShapeOf[Double, DoubleShape] = DoubleShape
}

trait LiftedFractional[X, S] extends Fractional[Expr[X]] {

  def const(x: X): Expr[X]

  def fromInt(x: Int, shape: S): Expr[X]
}
