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
        val dvalBody = evalAutoDiff(body)
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

  def evalAutoDiff(funcBody: c.Tree): c.Tree = {
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

  def cleanup(casts: collection.Map[String, c.Tree])(v: c.Tree): c.Tree = {
    val cleaner = cleanup(casts) _

    // the type of an argument may have changed (e.g. Double -> DValue[Double])
    v match {
      case Ident(TermName(name)) =>
            println(s"TERM: ${name}")
        casts.get(name)
          .getOrElse({
            println(s"   NOT FOUND")
            Ident(TermName(name))
          })

      case q"$s.$method($o)" =>
//            println(s"MATCHING ${showRaw(v)}")
            println(s"METHOD: ${showRaw(v)}")
//            println(s" S: ${showRaw(s)}")
//            println(s" O: ${showRaw(o)}")

        val s_e = cleaner(s)
        val o_e = cleaner(o)
        q"$s_e.$method($o_e)"

      case q"$fn($a, $b)" =>
            println(s"FUNCTION: ${showRaw(fn)}")
//        val Select(qualifier, name) = fn

        val a_e = cleaner(a)
        val b_e = cleaner(b)
 //       q"${Select(qualifier, name)}($a_e, $b_e)"
        q"$fn($a_e, $b_e)"

      case q"$fn($o)" =>
            println(s"FUNCTION: ${showRaw(fn)}")
        val o_e = cleaner(o)
        q"$fn($o_e)"

      case q"$a match { case ..$cases }" =>
        val a_e = cleanup(Map.empty)(a)
        val cases_e = cases.map{ c => cleaner(c) }
        q"$a_e match { case ..$cases_e }"

      case _ =>
            println(s"UNMATCHED: ${showRaw(v)}")
        v
    }
  }

  def mkVar(ids: Seq[Int]) =
    TermName("var$" + ids.mkString("$"))

  class RVVisitor {

    private val vars = scala.collection.mutable.HashMap[TermName, TermName]()
    private val obs = scala.collection.mutable.HashMap[TermName, Seq[TermName]]()
    val guides = scala.collection.mutable.HashMap[TermName, c.Tree]()

    def visit(tree: c.Tree): Seq[c.Tree] = {
      tree match {

        case q"val $tname : $tpt = scappla.this.`package`.sample[$tDist]($prior, $posterior)" =>
          val guideVar = TermName(c.freshName())
          val tVarName = TermName(c.freshName())
          println(s"MATCHING ${showRaw(tDist.tpe)}")
          val guide = q"scappla.BBVIGuide[$tDist]($posterior)"
          guides += guideVar -> guide
          vars += tname -> tVarName
          println(s"  CASTING ${showRaw(q"$tname")}")
          Seq(
            q"val ${tVarName} = ${guideVar}.sample($prior)",
            q"val $tname: $tpt = ${tVarName}.get"
          )

        case q"val $tname : $tpt = scappla.this.`package`.sample[$tDist]($model)" =>
          val tVarName = TermName(c.freshName())
          vars += tname -> tVarName
          println(s"MATCHING ${showRaw(tDist.tpe)}")
          Seq(
            q"val ${tVarName} = ${cleaner(model)}.sample()",
            q"val $tname: $tpt = ${tVarName}.get"
          )

        // observe needs to produce (resultVar, dependentVars..)
        case q"scappla.this.`package`.observe[$tDist]($oDist, $o)" =>
          val obName = TermName(c.freshName())
          val result = q"val $obName : scappla.Observation = scappla.observeImpl[$tDist](${cleaner(oDist)}, $o)"
          // FIXME: determine actual dependencies from oDist expression
          obs += obName -> vars.keySet.toSeq
          println(s" OBSERVE ${showRaw(result)}")
          result +: vars.values.toSeq.map { v =>
            q"$v.addObservation($obName.score)"
          }

        case q"val $tname : $tpt = $expr" =>
          Seq(
            q"val ${tname} = ${cleaner(expr)}"
          )

        // need to ensure that last expression is an RV
        case _ => Seq(cleaner(tree))
      }

    }

    def build(last: TermName, rType: c.Type): c.Tree = {
      val lastVar = vars(last)
      q"""new Variable[$rType] {

          import scappla.DValue._

          val get: $rType = ${last}

          val modelScore: Score = {
              ${(obs.keys.map { t =>
                  q"$t.score"
                } ++ vars.values.map { t =>
                  q"$t.modelScore"
                }).foldLeft(q"DValue.toConstant(0.0)") {
                  (cur, o) => q"DAdd($cur, $o)"
                }
              }
          }

          val guideScore: Score = {
              ${vars.values.map { t =>
                  q"$t.guideScore"
                }.foldLeft(q"DValue.toConstant(0.0)") {
                  (cur, o) => q"DAdd($cur, $o)"
                }
              }
          }

          def addObservation(score: Score) =
            ${lastVar}.addObservation(score)

          def addVariable(modelScore: Score, guideScore: Score) =
            ${lastVar}.addVariable(modelScore, guideScore)

          def complete() = {
              ..${(obs.keys ++ vars.values.toSeq.reverse).map { t =>
                  q"$t.complete()"
                }
              }
          }
      }"""
    }

    def cleaner(v: c.Tree): c.Tree = {
      // the type of an argument may have changed (e.g. Double -> DValue[Double])
      v match {
        case Ident(TermName(name)) =>
          val tname = TermName(name)
          println(s"TERM: ${name}")
          Ident(vars.get(tname)
              .getOrElse({
                println(s"   NOT FOUND")
                tname })
          )

        case q"$s.$method($o)" =>
          //            println(s"MATCHING ${showRaw(v)}")
          println(s"METHOD: ${showRaw(v)}")
          //            println(s" S: ${showRaw(s)}")
          //            println(s" O: ${showRaw(o)}")

          val s_e = cleaner(s)
          val o_e = cleaner(o)
          q"$s_e.$method($o_e)"

        case q"$fn($a, $b)" =>
          println(s"FUNCTION: ${showRaw(fn)}")
          //        val Select(qualifier, name) = fn

          val a_e = cleaner(a)
          val b_e = cleaner(b)
          //       q"${Select(qualifier, name)}($a_e, $b_e)"
          q"$fn($a_e, $b_e)"

        case q"$fn($o)" =>
          println(s"FUNCTION: ${showRaw(fn)}")
          val o_e = cleaner(o)
          q"$fn($o_e)"

        case q"$a match { case ..$cases }" =>
          val a_e = cleanup(Map.empty)(a)
          val cases_e = cases.map{ c => cleaner(c) }
          q"$a_e match { case ..$cases_e }"

        case _ =>
          println(s"UNMATCHED: ${showRaw(v)}")
          v
      }
    }

  }

  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[scappla.Model[X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]

    def flattenBody(body: c.Tree): c.Tree = {
      body match {
        case q"{ ..$stmts }" =>
          val visitor = new RVVisitor()
          val setup :+ last = stmts
          val (prelim, lastName) = last match {
            case Ident(TermName(name)) =>
              (setup, TermName(name))
            case _ =>
              val lastName = TermName(c.freshName())
              (setup :+ q"val ${lastName} = ${last}", lastName)
          }
          val newSetup = prelim.flatMap(t => visitor.visit(t))
          val newLast = visitor.build(lastName, body.tpe)

          /*
          val casts = scala.collection.mutable.HashMap[String, c.Tree]()
          val cleaner = cleanup(casts) _
          val newVars = for { stmt <- stmts } yield {
            stmt match {

              case q"val $tname : $tpt = scappla.this.`package`.sample[$tDist]($prior, $posterior)" =>
                val guideVar = Ident(c.freshName())
                val tVarName = Ident(c.freshName())
                println(s"MATCHING ${showRaw(tDist.tpe)}")
                val guide = q"scappla.BBVIGuide[$tDist]($posterior)"
                casts += tname.decoded -> tVarName
                println(s"  CASTING ${showRaw(q"$tname")}")
                Seq(
                  q"val ${guideVar} = ${guide}" match {
                    case Block(List(valDef), _) => valDef
                  },
                  q"val ${tVarName} = ${guideVar}.sample($prior)" match {
                    case Block(List(valDef), _) => valDef
                  },
                  q"val $tname: $tpt = ${tVarName}.get"
                )

              case q"val $tname : $tpt = scappla.this.`package`.sample[$tDist]($model)" =>
                val tVarName = Ident(c.freshName())
                casts += tname.decoded -> tVarName
                println(s"MATCHING ${showRaw(tDist.tpe)}")
                Seq(
                  q"val ${tVarName} = ${cleaner(model)}.sample()" match {
                    case Block(List(valDef), _) => valDef
                  },
                  q"val $tname: $tpt = ${tVarName}.get"
                )

              case q"scappla.this.`package`.observe[$tDist]($oDist, $o)" =>
                val obName = Ident(c.freshName())
                val result = q"val $obName : scappla.Observation = scappla.observeImpl[$tDist](${cleaner(oDist)}, $o)" match {
                      case Block(List(valDef), _) => valDef
                  }
                println(s" OBSERVE ${showRaw(result)}")
                Seq(result)

              case q"val $tname : $tpt = $expr" =>
                Seq(
                  q"val ${cleaner(Ident(tname))} = ${cleaner(expr)}" match {
                    case Block(List(valDef), _) => valDef
                  }
                )

              case _ => Seq(cleaner(stmt))
            }
          }
          */
          q"""new Model[$xType] {

                ..${visitor.guides.map { case (guideVar, guide) =>
                    q"val ${guideVar} = ${guide}"
                }}

                def sample() = {
                  ..${newSetup :+ newLast}
                }
              }"""
          // q"""new Model[$xType] { def sample() = null }"""
      }
    }

    val newBody = flattenBody(fn.tree)
    println("INFERRING")
    println(showCode(newBody))
    c.Expr[scappla.Model[X]](newBody)
  }

  def infer1[Y: WeakTypeTag, X: WeakTypeTag](fn: c.Expr[Y => X]): c.Expr[scappla.Model1[Y, X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]
    val yType = implicitly[WeakTypeTag[Y]]

    def flattenBody(body: c.Tree, argName: c.TermName): c.Tree = {

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
              val newC = cleanup(Map.empty)(c)
              q"if ($newC) { $newT } else { $newA }"
          }
          val stmts = (
            for { (name, v) <- guides.toSeq } yield {
              val b = q"private val $name = ${v.guide}"
              b match {
                case Block(List(valDef), _) => valDef
              }
            }
          ) :+ q"""def sample(${mkVar(Seq(0))}: Variable[$yType]): Variable[$xType] = {

               val ${argName} = ${mkVar(Seq(0))}.get
               val result = $resultTree
               ${mkVar(Seq(0))}.addVariable(result.modelScore, result.guideScore)

               result
             }"""
          println(s"RAW CODE: ${showRaw(stmts.head)}")
          q"""new Model1[$yType, $xType] { ..$stmts }"""
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
