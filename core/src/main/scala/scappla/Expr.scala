package scappla

import scappla.Functions._
import scappla.tensor._
import scappla.optimization.Optimizer

import scala.collection.mutable
import scala.collection.AbstractIterator
import scala.collection.generic.MutableMapFactory
import scala.collection.generic.CanBuildFrom

sealed trait Expr[X, S] {
  type Type = X
  type Shp = S

  def unary_-(implicit num: Numeric[Value[X]]) =
    Apply1(this, num.negate)

  def +(other: Expr[X, S])(implicit num: Numeric[Value[X]]) =
    Apply2(this, other, num.plus)

  def *(other: Expr[X, S])(implicit num: Numeric[Value[X]]) =
    Apply2(this, other, num.times)
}

case class ConstantExpr[X, S](value: Value[X]) extends Expr[X, S]

class Param[X, S](
    val initial: X,
    val base: BaseField[X, S],
    val expr: ValueField[X, S],
    val name: Option[String] = None
) extends Expr[X, S] {
  type S
}

object Param {

  def apply[X, S](
      initial: X,
      name: Option[String] = None
  )(implicit
      base: BaseField[X, S],
      expr: ValueField[X, S]
  ) = new Param[X, S](initial, base, expr, name)
}

case class Apply1[A, AS, X, S](in: Expr[A, AS], fn: Value[A] => Value[X]) extends Expr[X, S]

case class Apply2[A, AS, B, BS, C, CS](lhs: Expr[A, AS], rhs: Expr[B, BS], fn: (Value[A], Value[B]) => Value[C]) extends Expr[C, CS] {
  type LHS = A
  type RHS = B
}

object Expr {

//  implicit def toConstant[X, S](v: X): Expr[X, S] = ConstantExpr(new RealConstant(v))

  implicit def dataToConstant[X, S](v: X)(implicit bf: BaseField[X, S]): Expr[X, S] = ConstantExpr(Constant(v))

  implicit def valueToConstant[X, S](vdv: Value[X])(implicit bf: BaseField[X, S]): Expr[X, S] = ConstantExpr(vdv)

}

trait Interpreter {

  def eval[X, S](expr: Expr[X, S]): Value[X]

  def reset(): Unit
}

object NoopInterpreter extends Interpreter {

  // intentionally not implemented
  override def eval[X, S](expr: Expr[X, S]): Value[X] = ???

  override def reset(): Unit = {}
}

class OptimizingInterpreter(val opt: Optimizer) extends Interpreter {

  private val values: ReversibleLinkedHashMap[Any, Any] = new ReversibleLinkedHashMap[Any, Any]()

  private val params: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any]()

  @inline
  private def has[X, S](e: Expr[X, S]): Boolean = {
    values.contains(e)
  }

  @inline
  private def get[X, S](e: Expr[X, S]): Value[X] = {
    values(e).asInstanceOf[Value[X]]
  }

  @inline
  private def put[X, S](e: Expr[X, S], value: Value[X]) =
    values(e) = value

  override def eval[X, S](expr: Expr[X, S]): Value[X] = {
    if (!has(expr)) {
      val value: Value[X] = expr match {
        case cst: ConstantExpr[X, S] =>
          cst.value

        case param: Param[X, S] =>
          val p = if (params.contains(expr)) {
            params(expr).asInstanceOf[Value[X]]
          } else {
            val p = opt.param[X, S](param.initial, param.name)(param.base, param.expr)
            params(expr) = p
            p
          }
          p

        case app: Apply1[_, _, X, S] =>
          val upstream = eval(app.in)
          app.fn(upstream)

        case app: Apply2[_, _, _, _, X, S] =>
          val upA: Value[app.LHS] = eval(app.lhs)
          val upB: Value[app.RHS] = eval(app.rhs)
          app.fn(upA, upB)
      }
      put(expr, value.buffer)
    }
    get(expr)
  }

  override def reset(): Unit = {
    for { (_, value) <- values.reverse } {
      value.asInstanceOf[Completeable].complete()
    }
    values.clear()
  }
}

class ReversibleLinkedHashMap[A, B]() extends mutable.LinkedHashMap[A, B] with mutable.Map[A, B] with mutable.MapLike[A, B, ReversibleLinkedHashMap[A, B]] {

  def reverseIterator: Iterator[(A, B)] = new AbstractIterator[(A, B)] {
    private[this] var curr = lastEntry
    def hasNext = curr ne null
    def next() = {
      if (hasNext) {
        val res = (curr.key, curr.value)
        curr = curr.earlier
        res
      }
      else Iterator.empty.next()
    }
  }

  def reverse: Iterable[(A, B)] = new mutable.AbstractIterable[(A, B)] {
    def iterator = reverseIterator
  }

  override def empty = new ReversibleLinkedHashMap[A, B]()

  override def newBuilder = ReversibleLinkedHashMap.newBuilder[A, B]

  override def clone() = empty ++= repr

  override def -(key: A) = clone() -= key

}

object ReversibleLinkedHashMap extends MutableMapFactory[ReversibleLinkedHashMap] {
  implicit def cbf[A, B]: CanBuildFrom[Coll, (A, B), ReversibleLinkedHashMap[A, B]] = new MapCanBuildFrom[A, B]
  def empty[A, B] = new ReversibleLinkedHashMap[A, B]
}