package scappla

trait Expr[X] {

  def v: X

  def dv(v: X): Unit

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
