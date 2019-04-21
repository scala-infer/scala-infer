package scappla

import scappla.Functions.{exp, log, sigmoid, sum}
import scappla.optimization.Optimizer

import scala.collection.mutable

sealed trait Expr[X, S] {
  type Type = X
  type Shp = S

  def +(other: Expr[X, S])(implicit num: Numeric[Value[X]]) =
    Add(this, other, num)

  def *(other: Expr[X, S])(implicit num: Numeric[Value[X]]) =
    Times(this, other, num)

  def unary_-(implicit num: Numeric[Value[X]]) =
    Neg(this, num)
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

case class Neg[X, S](a: Expr[X, S], base: Numeric[Value[X]]) extends Expr[X, S]

case class Add[X, S](a: Expr[X, S], b: Expr[X, S], base: Numeric[Value[X]]) extends Expr[X, S]

case class Times[X, S](a: Expr[X, S], b: Expr[X, S], base: Numeric[Value[X]]) extends Expr[X, S]

case class Log[X, S](a: Expr[X, S], valueFn: log.Apply[Value[X], Value[X]]) extends Expr[X, S]

case class Sigmoid[X, S](a: Expr[X, S], valueFn: sigmoid.Apply[Value[X], Value[X]]) extends Expr[X, S]

case class Exp[X, S](a: Expr[X, S], valueFn: exp.Apply[Value[X], Value[X]]) extends Expr[X, S]

case class Sum[X, S](a: Expr[X, S], valueFn: sum.Apply[Value[X], Value[Double]]) extends Expr[Double, Unit] {
  type SrcType = X
  type SrcShp = S
}

object Expr {

//  implicit def toConstant[X, S](v: X): Expr[X, S] = ConstantExpr(new RealConstant(v))

  implicit def dataToConstant[X, S](v: X)(implicit bf: BaseField[X, S]): Expr[X, S] = ConstantExpr(Constant(v))

  implicit def valueToConstant[X, S](vdv: Value[X])(implicit bf: BaseField[X, S]): Expr[X, S] = ConstantExpr(vdv)

  implicit def logApply[X, S](implicit logFn: log.Apply[Value[X], Value[X]]): log.Apply[Expr[X, S], Expr[X, S]] =
    new Functions.log.Apply[Expr[X, S], Expr[X, S]] {
      override def apply(x: Expr[X, S]): Expr[X, S] = Log(x, logFn)
    }

  implicit def sigmoidApply[X, S](implicit logFn: sigmoid.Apply[Value[X], Value[X]]): sigmoid.Apply[Expr[X, S], Expr[X, S]] =
    new Functions.sigmoid.Apply[Expr[X, S], Expr[X, S]] {
      override def apply(x: Expr[X, S]): Expr[X, S] = Sigmoid(x, logFn)
    }

  implicit def expApply[X, S](implicit logFn: exp.Apply[Value[X], Value[X]]): exp.Apply[Expr[X, S], Expr[X, S]] =
    new Functions.exp.Apply[Expr[X, S], Expr[X, S]] {
      override def apply(x: Expr[X, S]): Expr[X, S] = Exp(x, logFn)
    }

  implicit def sumApply[X, S](implicit valueFn: sum.Apply[Value[X], Value[Double]]): sum.Apply[Expr[X, S], Expr[Double, Unit]] =
    new Functions.sum.Apply[Expr[X, S], Expr[Double, Unit]] {
      override def apply(x: Expr[X, S]): Expr[Double, Unit] = Sum(x,valueFn)
    }

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

  private val values: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any]()

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

        case e: Neg[X, S] =>
          val upstream = eval(e.a)
          upstream.unary_-()(e.base)

        case e: Add[X, S] =>
          val upA = eval(e.a)
          val upB = eval(e.b)
          upA.+(upB)(e.base)

        case e: Times[X, S] =>
          val upA = eval(e.a)
          val upB = eval(e.b)
          upA.*(upB)(e.base)

        case e: Log[X, S] =>
          log.apply(eval(e.a))(e.valueFn)

        case e: Sigmoid[X, S] =>
          sigmoid.apply(eval(e.a))(e.valueFn)

        case e: Exp[X, S] =>
          exp.apply(eval(e.a))(e.valueFn)

        case e: Sum[_, _] =>
          val result = e.a match {
            case ex: Expr[e.SrcType, e.SrcShp] =>
              val ee: Value[e.SrcType] = eval(ex)
              sum.apply(ee)(e.valueFn)
          }
          result.asInstanceOf[Value[X]]

      }
      put(expr, value)
    }
    get(expr)
  }

  override def reset(): Unit = {
    values.clear()
  }
}
