package scappla

trait Differentiable[X] {

  def v: X

  def dv(v: X): Unit

  def buffer: Buffered[X]

  def const: Constant[X] =
    new Constant(this.v)
}

trait Buffered[X] extends Differentiable[X] with Completeable

class Constant[X](val v: X) extends Buffered[X] {

  override def dv(v: X): Unit = {}

  override def buffer: Buffered[X] = this

  override def complete(): Unit = {}
}
