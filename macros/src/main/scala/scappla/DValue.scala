package scappla

import scala.annotation.StaticAnnotation
import scala.language.experimental.macros
import scala.reflect.api.Trees
import scala.reflect.macros.whitebox
import scala.reflect.macros._

trait DValue[X] {

  def v: X

  def dv(v: X): Unit

  def complete(): Unit = {}
}

class DScalar(self: DValue[Double]) {

  def unary_- = new DValue[Double] {

    override def v: Double = -self.v

    override def dv(v: Double): Unit = self.dv(-v)
  }

  def +(other: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = self.v + other.v

    override def dv(v: Double): Unit = {
      grad += v
    }

    override def complete(): Unit = {
      self.dv(grad)
      other.dv(grad)
    }
  }

  def *(other: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = self.v * other.v

    override def dv(v: Double): Unit = {
      grad += v
    }

    override def complete(): Unit = {
      self.dv(grad * other.v)
      other.dv(grad * self.v)
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

object DValue {

  implicit def toConstant(value: Double) = new DConstant(value)

  implicit def toScalarOps(value: DValue[Double]): DScalar = {
    new DScalar(value)
  }

  // returned value takes ownership of the reference passed on the stack
  def log(x: DValue[Double]) = new DValue[Double] {

    override lazy val v: Double = scala.math.log(x.v)

    override def dv(dx: Double): Unit = {
      x.dv(dx / x.v)
    }
  }

}

class dvalue extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro DValueTransformer.impl
}

class DValueTransformer(val c: whitebox.Context) {

  import c.universe._

  def impl(annottees: c.Expr[Any]*): c.Expr[Any] = {
    val inputs = annottees.map(_.tree).toList
    /*
    for (input <- inputs) {
      println(showRaw(input))
      println()
    }
    */
    inputs.head match {
      case q"def $name($argName: $valType): $tpt = $body" =>
//      case (method: DefDef) :: _ =>
//        val DefDef(mods, name, List(), List(List(ValDef(_, argName, valType, _))), tpt, body) = method
        println(s"RAW: ${showRaw(body)}")
        val (_, valExpr) :: stmts = flattenBody(Seq(0), body)._1
        println(s"STMTS: ${stmts}")
        val result =
          q"""object $name {

                import _root_.scappla.DValue._

                class impl($argName: DValue[$valType]) extends DValue[$tpt] {

                  ..${stmts.reverse.map{case (vName, vExpr) => q"val $vName=$vExpr"}}

                  private val _v: DValue[$tpt] = $valExpr

                  val v: $tpt = _v.v

                  def dv(d: $tpt): Unit = _v.dv(d)

                  override def complete() = {
                    _v.complete()
                    ..${stmts.map { case (vName, _) => q"$vName.complete()" }}
                  }
                }

                def apply($argName: DValue[$valType]) = new impl($argName)
             }"""
        println(s"RESULT: ${showCode(result)}")
        //                  def v: $tpt = { ..$stmts }
        //        val tree = ClassDef(mods, termName.toTypeName, tparams, Template())
        //        c.Expr[Any](Block(List(), Literal(Constant(()))))
        c.Expr[Any](result)

      case _ =>
        (EmptyTree, inputs)
        c.Expr[Any](Literal(Constant(())))
    }
  }

  def flattenBody(ids: Seq[Int], funcBody: c.Tree): (List[(c.TermName, c.Tree)], Boolean) = {
    println(s"EXPANDING ${showRaw(funcBody)}")
    funcBody match {
      case q"$s.$method($o)" =>
        val (sbjStmts, sbjCom) = flattenBody(ids :+ 0, s)
        val (sbjVar, sbjDef) = if (sbjCom) {
          val varname = sbjStmts.head._1
          (Ident(varname), sbjStmts)
        } else {
          (s, List.empty)
        }
        val (objStmts, objCom) = flattenBody(ids :+ 1, o)
        val (objVar, objDef) = if (objCom) {
          val varname = objStmts.head._1
          (Ident(varname), objStmts)
        } else {
          (o, List.empty)
        }
        (
            List(
              (TermName("var$" + ids.mkString("$")), q"$sbjVar.$method($objVar)")
            ) ++ objDef ++ sbjDef,
            true
        )
      case _ =>
        (List((TermName("dummy"), funcBody)), false)
    }
  }

}
