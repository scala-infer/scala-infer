package scappla

import scala.reflect.macros.blackbox

class Macros(val c: blackbox.Context) {

  import c.universe._

  def autodiff[A: WeakTypeTag, B: WeakTypeTag](fn: c.Expr[A => B]): c.Expr[DFunction1[A, B]] = {
    fn.tree match {
      case q"($argName: $argType) => $body" =>
        //      case (method: DefDef) :: _ =>
        //        val DefDef(mods, name, List(), List(List(ValDef(_, argName, valType, _))), tpt, body) = method
        println(s"RAW: ${showRaw(body)}")
        val stmts = flattenBody(Seq(0), body)._1
        val tpt = implicitly[WeakTypeTag[B]]
        val valType = implicitly[WeakTypeTag[A]]
        println(s"STMTS: ${stmts}")

        /*
                        def apply($argName: $valType) = ${new Transformer {
                    override def transform(tree: c.universe.Tree): c.universe.Tree = {
                      super.transform(tree) match {
                        case Ident(TermName(tname)) =>
                          println(s"  TRANSFORMING ${tname}")
                          Ident(TermName(tname))
                        case t @ _ => t
                      }
                    }
                  }.transform(body)}
                     */
        val result =
          q"""new DFunction1[$valType, $tpt] {

                def apply($argName: $valType) = ${c.parse(showCode(body))}

                def apply($argName: DValue[$valType]) = new DValue[$tpt] {

                  import _root_.scappla.DValue._

                  ..${stmts.reverse.map {
                    case (vName, vExpr) =>
                      q"private val $vName = $vExpr.buffer"
                    }
                  }

                  def v: $tpt = ${stmts.head._1}.v

                  def dv(d: $tpt): Unit = {
                    ${stmts.head._1}.dv(d)
                    ..${stmts.map { case (vName, _) => q"$vName.complete()" }}
                  }
               }
             }"""
        println(s"RESULT: ${showRaw(result, printOwners=true, printPositions=true, printTypes = true)}")
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
                      q"private val $vName = $vExpr.buffer"
                    }
                  }

                  def v: $tpt = ${stmts.head._1}.v

                  def dv(d: $tpt): Unit = {
                    ${stmts.head._1}.dv(d)
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

  /*
  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[Distribution[X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]
    fn.tree match {
      case q"() => $body" =>
        val result = q"""new Distribution[$xType]{}"""
        c.Expr(result)
      case _ =>
        c.Expr(EmptyTree)
    }
  }
  */
}
