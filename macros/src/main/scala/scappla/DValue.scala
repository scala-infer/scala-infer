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

  def log(x: Double): Double = scala.math.log(x)

  // returned value takes ownership of the reference passed on the stack
  def log(x: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = scala.math.log(x.v)

    override def dv(dx: Double): Unit = {
      grad += dx
    }

    override def complete(): Unit = {
      x.dv(grad / x.v)
    }
  }

  def pow(base: Double, exp: Double): Double = scala.math.pow(base, exp)

  def pow(base: DValue[Double], exp: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = scala.math.pow(base.v, exp.v)

    override def dv(dx: Double): Unit = {
      grad += dx
    }

    override def complete(): Unit = {
      base.dv(grad * exp.v * scala.math.pow(base.v, exp.v - 1))
      exp.dv(grad * scala.math.log(base.v) * v)
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
        val stmts = flattenBody(Seq(0), body)._1
        println(s"STMTS: ${stmts}")
        val result =
          q"""object $name {

                import _root_.scappla.DValue._

                class impl($argName: DValue[$valType]) extends DValue[$tpt] {

                  ..${stmts.reverse.map{case (vName, vExpr) => q"val $vName=$vExpr"}}

                  def v: $tpt = ${stmts.head._1}.v

                  def dv(d: $tpt): Unit = ${stmts.head._1}.dv(d)

                  override def complete() = {
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
    def expand(idx: Int, v: c.Tree) = {
      val (sbjStmts, sbjCom) = flattenBody(ids :+ idx, v)
      if (sbjCom) {
        val varname = sbjStmts.head._1
        (Ident(varname), sbjStmts)
      } else {
        (v, List.empty)
      }
    }
    funcBody match {
      case q"$s.$method($o)" =>
        val (sbjVar, sbjDef) = expand(0, s)
        val (objVar, objDef) = expand(1, o)
        (
            List(
              (TermName("var$" + ids.mkString("$")), q"$sbjVar.$method($objVar)")
            ) ++ objDef ++ sbjDef,
            true
        )
      case q"$fn($a, $b)" =>
        val (aVar, aDef) = expand(0, a)
        val (bVar, bDef) = expand(1, b)
        (
            List(
              (TermName("var$" + ids.mkString("$")), q"$fn($aVar, $bVar)")
            ) ++ aDef ++ bDef,
            true
        )
      case q"$fn($o)" =>
        val (objVar, objDef) = expand(0, o)
        (
            List(
              (TermName("var$" + ids.mkString("$")), q"$fn($objVar)")
            ) ++ objDef,
            true
        )
      case _ =>
        (List((TermName("dummy"), funcBody)), false)
    }
  }

}
