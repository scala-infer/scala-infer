package scappla

import scala.reflect.macros.blackbox

class Macros(val c: blackbox.Context) {

  import c.universe._

  def autodiff[A: WeakTypeTag, B: WeakTypeTag](fn: c.Expr[A => B]): c.Expr[DFunction1[A, B]] = {
    fn.tree match {
      case q"($argName: $argType) => $body" =>
        //      case (method: DefDef) :: _ =>
        //        val DefDef(mods, name, List(), List(List(ValDef(_, argName, valType, _))), tpt, body) = method
//        println(s"RAW: ${showRaw(body)}")
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
  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[Model[X]] = {
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

  def infer1[Y: WeakTypeTag, X: WeakTypeTag](fn: c.Expr[Y => X]): c.Expr[scappla.Model1[Y, X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]
    val yType = implicitly[WeakTypeTag[Y]]

    def mkVar(ids: Seq[Int]) =
      TermName("var$" + ids.mkString("$"))

    def expand(v: c.Tree): c.Tree = {
      // the type of an argument may have changed (e.g. Double -> DValue[Double])
      v match {
        case Ident(TermName(name)) =>
          Ident(TermName(name))

        case q"$s.$method($o)" =>
          val s_e = expand(s)
          val o_e = expand(o)
          q"$s_e.$method($o_e)"

        case q"$fn($a, $b)" =>
          val Select(qualifier, name) = fn

          val a_e = expand(a)
          val b_e = expand(b)
          q"${Select(qualifier, name)}($a_e, $b_e)"

        case q"$fn($o)" =>
          val o_e = expand(o)
          q"$fn($o_e)"

        case q"if ($cond) { $t } else { $a }" =>
          q"""if ($cond) { println("YES!"); $t } else { $a }"""

        case _ =>
          v
      }
    }

    def flattenBody(body: c.Tree, argName: c.TermName): c.Tree = {

      def cleanup(v: c.Tree): c.Tree = {
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

            val s_e = cleanup(s)
            val o_e = cleanup(o)
            q"$s_e.$method($o_e)"

          case q"$fn($a, $b)" =>
//            println(s"FUNCTION: ${showRaw(fn)}")
            val Select(qualifier, name) = fn

            val a_e = cleanup(a)
            val b_e = cleanup(b)
            q"${Select(qualifier, name)}($a_e, $b_e)"

          case q"$fn($o)" =>
//            println(s"FUNCTION: ${showRaw(fn)}")
            val o_e = cleanup(o)
            q"$fn($o_e)"

          case _ =>
//            println(s"UNMATCHED: ${showRaw(v)}")
            v
        }
      }


      val mapping = scala.collection.mutable.HashMap[c.TermName, c.Tree]()

      case class Var(model: c.Tree, guide: c.Tree)

      body match {
        case q"{ ..$defs }" if defs.size >= 1 =>
          val guides = scala.collection.mutable.HashMap[c.Tree, Var]()
          val resultTree = defs.last match {
            case q"if ($c) { $t } else { $a }" =>
              val newT = t match {
                case q"scappla.this.`package`.sample[$tDist]($prior, $posterior)" =>
                  println(s"MATCHING ${showRaw(tDist.tpe)}")
                  val guide = q"scappla.BBVIGuide[$tDist]($posterior)"
                  guides += q"t" -> Var(prior, guide)
                  q"t.sample($prior)"
              }
              val newA = a match {
                case q"scappla.this.`package`.sample[$aDist]($prior, $posterior)" =>
                  println(s"MATCHING ${showRaw(aDist.tpe)}")
                  val guide = q"scappla.BBVIGuide[$aDist]($posterior)"
                  guides += q"a" -> Var(prior, guide)
                  q"a.sample($prior)"
              }
              val newC = cleanup(c)
              q"if ($newC) { $newT } else { $newA }"
          }
          val stmts = (for { (name, v) <- guides.toSeq }
                       yield {
                         val b = q"private val $name = ${v.guide}"
                         b match {
                           case Block(List(valDef), _) => valDef
                         }
                       }
          ) :+ q"""def sample(${mkVar(Seq(0))}: Variable[$yType]): Variable[$xType] = {

               val ${argName} = ${mkVar(Seq(0))}.get
               val result = $resultTree
               ${mkVar(Seq(0))}.addVariable(result.modelScore, result.guideScore)

               println("ARGUMENT: " + $argName)
               println(s"RESULT: $${result}")

               result
             }"""
          println(s"RAW CODE: ${showRaw(stmts.head)}")
          q"""new Model1[$yType, $xType] { ..$stmts }"""

            /*
          ..${
      stmts.reverse.map {
      case (vName, vExpr) =>
      q"private val $vName = $vExpr.buffer"
      }
      }
      */

      }
    }

    fn.tree match {
      case q"($argName: $argType) => $body" =>

        val newBody = flattenBody(body, argName)
        println("INFERRING")
        println(showCode(newBody))
        c.Expr(newBody)

      case _ =>
        c.Expr(EmptyTree)
    }
  }

}
