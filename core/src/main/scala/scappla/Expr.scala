package scappla

import scappla.Functions._
import scappla.tensor._
import scappla.optimization.Optimizer

import scala.collection.mutable
import scala.collection.AbstractIterator
import scala.collection.generic.MutableMapFactory
import scala.collection.generic.CanBuildFrom
import scappla.tensor.Tensor.TensorField

sealed trait Expr[X, S] {
  type Type = X
  type Shp = S

  def unary_- =
    Apply1(this, { v: Value[X, S] => -v })

  def +(other: Expr[X, S]) =
    Apply2(this, other, {
      (thisv: Value[X, S], otherv: Value[X, S]) => thisv + otherv
    })

  def -(other: Expr[X, S]) =
    Apply2(this, other, {
      (thisv: Value[X, S], otherv: Value[X, S]) => thisv - otherv
    })

  def *(other: Expr[X, S]) =
    Apply2(this, other, {
      (thisv: Value[X, S], otherv: Value[X, S]) => thisv * otherv
    })

  def /(other: Expr[X, S]) =
    Apply2(this, other, {
      (thisv: Value[X, S], otherv: Value[X, S]) => thisv / otherv
    })
}

object Expr {

  implicit def apply(value: Double): Expr[Double, Unit] = ConstantExpr(Constant(value, ()))

  implicit def valueToConstant[X, S](vdv: Value[X, S])(implicit bf: BaseField[X, S]): Expr[X, S] = ConstantExpr(vdv)
}

case class ConstantExpr[X, S](value: Value[X, S]) extends Expr[X, S] {
  override val hashCode: Int = {
    value.v.hashCode()
  }
}

class Param[X, S](
    val initial: X,
    val shape: S,
    val base: BaseField[X, S],
    val name: Option[String] = None
) extends Expr[X, S] {
  type S
}

object Param {

  def apply(initial: Double) =
    new Param[Double, Unit](initial, (), implicitly[BaseField[Double, Unit]], None)

  def apply(initial: Double, name: String) =
    new Param[Double, Unit](initial, (), implicitly[BaseField[Double, Unit]], Some(name))

  def apply[X, S](initial: X, shape: S)(implicit base: BaseField[X, S]) =
    new Param[X, S](initial, shape, base, None)

  def apply[X, S](initial: X, shape: S, name: String)(implicit base: BaseField[X, S]) =
    new Param[X, S](initial, shape, base, Some(name))
}

case class Apply1[A, AS, X, S](in: Expr[A, AS], fn: Value[A, AS] => Value[X, S]) extends Expr[X, S]

case class Apply2[A, AS, B, BS, C, CS](lhs: Expr[A, AS], rhs: Expr[B, BS], fn: (Value[A, AS], Value[B, BS]) => Value[C, CS]) extends Expr[C, CS] {
  type LHS = A
  type RHS = B
}

trait Interpreter {

  def eval[X, S](expr: Expr[X, S]): Value[X, S]

  def reset(): Unit
}

object NoopInterpreter extends Interpreter {

  // intentionally not implemented
  override def eval[X, S](expr: Expr[X, S]): Value[X, S] = ???

  override def reset(): Unit = {}
}

/**
 * A default optimizer (Adam, SGD or another first-order algorithm) takes care of
 * handling parameters that do not have an optimizer assigned to them.
 *
 * Groups of parameters are optimized together, allowing the optimizer to leverage
 * higher-order gradient (approximate Hessian) information.
 */
class OptimizingInterpreter(
    val opt: Optimizer,
) extends Interpreter {

  private val values = mutable.HashMap[Any, Any]()

  private val params: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any]()

  @inline
  private def has[X, S](e: Expr[X, S]): Boolean = {
    values.contains(e)
  }

  @inline
  private def get[X, S](e: Expr[X, S]): Value[X, S] = {
    values(e).asInstanceOf[Value[X, S]]
  }

  @inline
  private def put[X, S](e: Expr[X, S], value: Value[X, S]) =
    values(e) = value

  override def eval[X, S](expr: Expr[X, S]): Value[X, S] = {
    expr match {
      case cst: ConstantExpr[X, S] =>
        cst.value
      case _ => if (!has(expr)) {
        val value: Value[X, S] = expr match {

          case param: Param[X, S] =>
            val p = if (params.contains(expr)) {
              params(expr).asInstanceOf[Value[X, S]]
            } else {
              val p = opt.param[X, S](
                param.initial,
                param.shape,
                param.name
              )(param.base)

              params(expr) = p
              p
            }
            p.buffer

          case app: Apply1[_, _, X, S] =>
            val upstream = eval(app.in)
            app.fn(upstream)

          case app: Apply2[_, _, _, _, X, S] =>
            val upA = eval(app.lhs)
            val upB = eval(app.rhs)
            app.fn(upA, upB)
        }
        put(expr, value)
      }
      get(expr)
    }
  }

  override def reset(): Unit = {
    for { (expr, value) <- values if expr.isInstanceOf[Param[_, _]] } {
        value.asInstanceOf[Completeable].complete()
    }
    opt.step()
    values.clear()
  }
}