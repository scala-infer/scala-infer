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

trait DFunction1[From, To] {

  def apply(in: From): To

  def apply(in: DValue[From]): DValue[To]
}

trait DFunction2[From1, From2, To] {

  def apply(in1: From1, in2: From2): To

  def apply(in1: DValue[From1], in2: DValue[From2]): DValue[To]
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

object Functions {

  def autodiff[A, B](fn: A => B): DValue[A] => DValue[B] = macro FunctionMacros.autodiff[A, B]

  object log extends DFunction1[Double, Double] {

    def apply(x: Double): Double = scala.math.log(x)

    // returned value takes ownership of the reference passed on the stack
    def apply(x: DValue[Double]): DValue[Double] = new DValue[Double] {

      private var grad = 0.0

      override lazy val v: Double = scala.math.log(x.v)

      override def dv(dx: Double): Unit = {
        grad += dx
      }

      override def complete(): Unit = {
        x.dv(grad / x.v)
      }
    }
  }

  object pow extends DFunction2[Double, Double, Double] {

    def apply(base: Double, exp: Double): Double = scala.math.pow(base, exp)

    def apply(base: DValue[Double], exp: DValue[Double]) = new DValue[Double] {

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

}

object DValue {

  implicit def toConstant(value: Double) = new DConstant(value)

  implicit def toScalarOps(value: DValue[Double]): DScalar = {
    new DScalar(value)
  }
}

class FunctionMacros(val c: whitebox.Context) {

  import c.universe._

  def autodiff[A: WeakTypeTag, B: WeakTypeTag](fn: c.Expr[A => B]): c.Expr[DValue[A] => DValue[B]] = {
    fn.tree match {
      case q"($argName: $argType) => $body" =>
        //      case (method: DefDef) :: _ =>
        //        val DefDef(mods, name, List(), List(List(ValDef(_, argName, valType, _))), tpt, body) = method
        println(s"RAW: ${showRaw(body)}")
        val stmts = flattenBody(Seq(0), body)._1
        val tpt = implicitly[WeakTypeTag[B]]
        val valType = implicitly[WeakTypeTag[A]]
        println(s"STMTS: ${stmts}")
        val result =
          q"""($argName: DValue[$valType]) => new DValue[$tpt] {

                  import _root_.scappla.DValue._

                  ..${stmts.reverse.map{case (vName, vExpr) => q"private val $vName : DValue[Double] = $vExpr"}}

                  def v: $tpt = ${stmts.head._1}.v

                  def dv(d: $tpt): Unit = ${stmts.head._1}.dv(d)

                  override def complete() = {
                    ..${stmts.map { case (vName, _) => q"$vName.complete()" }}
                  }
             }"""
        println(s"RESULT: ${showCode(result, printTypes=true)}")
        //                  def v: $tpt = { ..$stmts }
        //        val tree = ClassDef(mods, termName.toTypeName, tparams, Template())
        //        c.Expr[Any](Block(List(), Literal(Constant(()))))
        c.Expr(result)

      case _ =>
        c.Expr(EmptyTree)
    }
  }

  def flattenBody(ids: Seq[Int], funcBody: c.Tree): (List[(c.TermName, c.Tree)], Boolean) = {
    println(s"EXPANDING ${showRaw(funcBody)}")

    def expand(idx: Int, v: c.Tree): (c.Tree, List[(c.TermName, c.Tree)]) = {
      val (sbjStmts, sbjCom) = flattenBody(ids :+ idx, v)
      if (sbjCom) {
        val varname = sbjStmts.head._1
        (Ident(varname), sbjStmts)
      } else {
        // the type of an argument may have changed (e.g. Double -> DValue[Double])
        val clone = v match {
          case Ident(TermName(name)) =>
            Ident(TermName(name))
          case _ => v
        }
        (clone, List.empty)
      }
    }

    def mkVar(ids: Seq[Int]) =
      TermName("var$" + ids.mkString("$"))

    funcBody match {
      case q"$s.$method($o)" =>
        println(s"METHOD: ${showRaw(method)}")

        val (sbjVar, sbjDef) = expand(0, s)
        val (objVar, objDef) = expand(1, o)
        (
            List(
              (mkVar(ids), q"$sbjVar.$method($objVar)")
            ) ++ objDef ++ sbjDef,
            true
        )
      case q"$fn($a, $b)" =>
        val (aVar, aDef) = expand(0, a)
        val (bVar, bDef) = expand(1, b)
        val Select(qualifier, name) = fn
        (
            List(
              (mkVar(ids), q"${Select(qualifier,name)}($aVar, $bVar)")
            ) ++ aDef ++ bDef,
            true
        )
      case q"$fn($o)" =>
        val (objVar, objDef) = expand(0, o)
        (
            List(
              (mkVar(ids), q"$fn($objVar)")
            ) ++ objDef,
            true
        )
      case q"{ ..$defs }" if (defs.size > 1) =>
        val stmts = defs.reverse.zipWithIndex.flatMap { case (stmt, idx) =>
            expand(idx, stmt)._2
        }
        println(s"  RAW last: ${showCode(defs.last, printTypes = true)}")
//        val tpt = defs.last.tpe
        val tpt = funcBody.tpe
        (
            List((mkVar(ids),
                q"""new DValue[$tpt] {

                  ..${stmts.reverse.map{
                    case (vName, vExpr) =>
                      q"private val $vName : DValue[Double]=$vExpr"
                    }
                  }

                  def v: $tpt = ${stmts.head._1}.v

                  def dv(d: $tpt): Unit = ${stmts.head._1}.dv(d)

                  override def complete() = {
                    ..${stmts.map { case (vName, _) => q"$vName.complete()" }}
                  }
             }""")),
            true
        )
      case q"val $tname : $tpt = $expr" =>
        println(s"  TYPEOF($tname) := $tpt")
        val (objVar, objDef) = expand(0, expr)
        val defHead :: defTail = objDef
        (
            List(
              (tname, q"${defHead._2}")
            ) ++ defTail,
            true
        )
      case _ =>
        (List((mkVar(ids), funcBody)), false)
    }
  }

}
