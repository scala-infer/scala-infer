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
        val dvalBody = flattenBody(body)
        val tpt = implicitly[WeakTypeTag[B]]
        val valType = implicitly[WeakTypeTag[A]]

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

                import _root_.scappla.DValue._

                def apply($argName: $valType) = ${c.parse(showCode(body))}

                def apply($argName: DValue[$valType]) = $dvalBody
             }"""
//        println(s"RESULT: ${showCode(result)}")
//        println(s"RESULT: ${showRaw(result, printOwners=true, printPositions=true, printTypes = true)}")
        //                  def v: $tpt = { ..$stmts }
        //        val tree = ClassDef(mods, termName.toTypeName, tparams, Template())
        //        c.Expr[Any](Block(List(), Literal(Constant(()))))
        c.Expr(result)

      case _ =>
        c.Expr(EmptyTree)
    }
  }

  def flattenBody(funcBody: c.Tree): c.Tree = {
//    println(s"EXPANDING ${showRaw(funcBody)}")


    def expand(v: c.Tree): c.Tree = {
        // the type of an argument may have changed (e.g. Double -> DValue[Double])
        v match {
          case Ident(TermName(name)) =>
//            println(s"TERM: ${name}")
            Ident(TermName(name))

          case q"$s.$method($o)" =>
//            println(s"MATCHING ${showRaw(v)}")
//            println(s"METHOD: ${showRaw(method)}")
//            println(s" S: ${showRaw(s)}")
//            println(s" O: ${showRaw(o)}")

            val s_e = expand(s)
            val o_e = expand(o)
            q"$s_e.$method($o_e)"

          case q"$fn($a, $b)" =>
//            println(s"FUNCTION: ${showRaw(fn)}")
            val Select(qualifier, name) = fn

            val a_e = expand(a)
            val b_e = expand(b)
            q"${Select(qualifier, name)}($a_e, $b_e)"

          case q"$fn($o)" =>
//            println(s"FUNCTION: ${showRaw(fn)}")
            val o_e = expand(o)
            q"$fn($o_e)"

          case _ =>
//            println(s"UNMATCHED: ${showRaw(v)}")
            v
        }
    }

    def mkVar(ids: Seq[Int]) =
      TermName("var$" + ids.mkString("$"))

    funcBody match {

      case q"{ ..$defs }" if defs.size > 1 =>
        val stmts = defs.reverse.flatMap {

          case q"val $tname : $tpt = $expr" =>
//            println(s"  TYPEOF($tname) := $tpt")
            val expr_e = expand(expr)
            Some((tname, expr_e))

          case _ => None
        }

        //        println(s"  RAW last: ${showCode(defs.last, printTypes = true)}")
        //        val tpt = defs.last.tpe
        val tpt = funcBody.tpe
        q"""new DValue[$tpt] {

            import _root_.scappla.DValue._

                  ..${
          stmts.reverse.map {
            case (vName, vExpr) =>
              q"private val $vName = $vExpr.buffer"
          }
        }

                  def v: $tpt = ${stmts.head._1}.v

                  def dv(d: $tpt): Unit = {
                    ${stmts.head._1}.dv(d)
                    ..${stmts.map { case (vName, _) => q"$vName.complete()" }}
                  }
             }"""

      case _ =>
        expand(funcBody)
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
