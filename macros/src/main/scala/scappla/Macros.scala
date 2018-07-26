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

  def mkVar(ids: Seq[Int]) =
    TermName("var$" + ids.mkString("$"))

  class RVVisitor(body: c.Tree, args: Map[TermName, TermName] = Map.empty) {

    println(s"VISITING ${showCode(body)}")

    private val vars = scala.collection.mutable.HashMap[TermName, Set[TermName]]()
    for { (arg, v) <- args } vars += arg -> Set(v)

    private val obs = scala.collection.mutable.HashSet[TermName]()
    val guides = scala.collection.mutable.HashMap[TermName, c.Tree]()

    val stmts = body match {
      case q"{..$stmts }" =>
        val setup :+ last = stmts
        val (prelim, lastName) = last match {
          case Ident(TermName(name)) =>
            (setup, TermName(name))
          case _ =>
            val lastName = TermName(c.freshName())
            (setup :+ q"val ${lastName} = ${last}", lastName)
        }
        val newSetup = prelim.flatMap(toANF).flatMap(t => visit(t))
        val newLast = build(lastName, body.tpe)
        newSetup :+ newLast
    }

    def toANF(tree: c.Tree): Seq[c.Tree] = {

      def expand(expr: c.Tree, doVar: Boolean = false)(fn: c.Tree => Seq[c.Tree]): Seq[c.Tree] = {
        expr match {
          case Ident(TermName(name)) =>
            fn(Ident(TermName(name)))

          case Literal(value) =>
            println(s"   LITERAL ${showRaw(expr)}")
            fn(expr)

          case _ =>
            val varName = TermName(c.freshName())
            val stmts :+ last = toANF(expr)
            if (doVar || stmts.size > 0) {
              (stmts :+ q"val $varName = $last") ++ fn(Ident(varName))
            } else {
              fn(last)
            }
        }
      }

      def reIdentPat(expr: c.Tree): c.Tree = {
        println(s"   REMAPPING ${showRaw(expr)}")
        expr match {
          case pq"$tName @ $pat" =>
            println(s"     TERM ${showRaw(tName)}")
            val TermName(name) = tName
            pq"${TermName(name)} @ $pat"

          case pq"$ref(..$pats)" =>
            pq"$ref(..${pats.map{pat => reIdentPat(pat)}})"

          case _ =>
            expr
        }
      }

      println(s"MAMTCHINIG ${showRaw(tree)}")
      tree match {
        case q"{ ..$stmts }" if stmts.size > 1 =>
         stmts.flatMap(t => toANF(t))

        case q"if ($cond) $tExpr else $fExpr" =>
          expand(cond, true) { condName =>
            Seq(q"if ($condName) ${inl(toANF(tExpr))} else ${inl(toANF(fExpr))}")
          }

        case q"$expr match { case ..$cases }" =>
          val mappedCases = cases.map { c =>
            val cq"$when => $result" = c
            cq"${reIdentPat(when)} => ${inl(toANF(result))}"
          }
          expand(expr, true) { matchName =>
            Seq(q"$matchName match { case ..${mappedCases} }")
          }

        case q"$s.$method($o)" =>
          expand(s) { sName =>
            expand(o) { oName =>
              Seq(q"$sName.$method($oName)")
            }
          }

        /*
        case q"$fn($a, $b)" =>
          expand(a) { aName =>
            expand(b) { bName =>
              Seq(q"$fn($aName, ${bName})")
            }
          }
         */

        case q"$fn($o)" =>
          expand(o, true) { oName =>
            Seq(q"$fn(${oName})")
          }

        case q"$fn(..$args)" =>
          println(s"  FN MATCH ${showRaw(fn)}")
          val res = args.map { arg =>
            expand(arg) { aName =>
              Seq(aName)
            }
          }
          val defs = res.flatMap { _.dropRight(1) }
          val newArgs = res.map { _.last }
          defs :+ q"$fn(..$newArgs)"

        case q"$mods val $tname : $tpt = $expr" =>
          println(s"    VALDEFF ${showRaw(tname)}")
          val TermName(name) = tname
          val stmts :+ last = toANF(expr)
          stmts :+ q"$mods val ${TermName(name)}: $tpt = $last"

        case Ident(TermName(name)) =>
          Seq(Ident(TermName(name)))

        case q"(..$args) => $fn" =>
          Seq(q"""(..${args.map { arg => arg match {
                         case ValDef(mods, TermName(name), tpt, expr) =>
                           println(s"    FN VALDEF ${showRaw(arg)}")
                           val stmts = toANF(expr)
                           ValDef(mods, TermName(name), tpt, q"{..$stmts}")

                         case _ =>
                           println(s"    FN ARG ${showRaw(arg)}")
                           arg
                     } }}) => {
                ..${expand(fn) { fName => Seq(fName) }}
              }
            """)

        case q"$expr: $tpt" =>
          expand(expr) { eName =>
            Seq(q"$eName: $tpt")
          }

        case q"$expr.$tname" =>
          expand(expr) { eName =>
            Seq(q"$eName.$tname")
          }

        case _ =>
          println(s"  SKIPPING ${showRaw(tree)}")
          Seq(tree)

      }
    }

    def inl(exprs: Seq[c.Tree]) = {
      if (exprs.size == 1) {
        exprs.head
      } else {
        q"{ ..$exprs }"
      }
    }

    def visit(tree: c.Tree): Seq[c.Tree] = {
      tree match {

        case q"val $tname : $tpt = scappla.this.`package`.sample[$tDist]($prior, $posterior)" =>
          val guideVar = TermName(c.freshName())
          val tVarName = TermName(c.freshName())
          val TermName(name) = tname
          println(s"MATCHING ${showRaw(tDist.tpe)} (${showCode(tDist)})")
          val guide = tDist.tpe match {
            case tpe if tpe =:= typeOf[DValue[Double]] =>
              q"scappla.ReparamGuide($posterior)"

            case _ =>
              q"scappla.BBVIGuide[$tDist]($posterior)"
          }
          guides += guideVar -> guide
          vars += tname -> Set(tVarName)
          println(s"  CASTING ${showRaw(q"$tname")}")
          Seq(
            q"val ${tVarName} = ${guideVar}.sample($prior)",
            q"val ${TermName(name)}: $tpt = ${tVarName}.get"
          )

        case q"val $tname : $tpt = scappla.this.`package`.sample[$tDist]($model)" =>
          val tVarName = TermName(c.freshName())
          vars += tname -> Set(tVarName)
          var Ident(TermName(modelTName)) = model
          val deps = vars.getOrElse(TermName(modelTName), Set.empty)
          println(s"MATCHING ${showRaw(model)}")
          val Ident(TermName(varName)) = model
          println(s"     DEPS: ${vars(TermName(varName))}")
          Seq(
            q"""val ${tVarName} = ${model}
               .withDeps(new Dependencies(..${deps}))
               .sample()""",
            q"val $tname: $tpt = ${tVarName}.get"
          )

        // observe needs to produce (resultVar, dependentVars..)
        case q"scappla.this.`package`.observe[$tDist]($oDist, $o)" =>
          val obName = TermName(c.freshName())
          val result = q"val $obName : scappla.Observation = scappla.observeImpl[$tDist](${oDist}, $o)"
          obs += obName
          val deps = findVars(oDist)
          // println(s"  VARIABLES in oDist: ${deps} => ${vars(deps.head) }")
          println(s" OBSERVE ${showRaw(result)}")
          result +: deps.flatMap { d =>
            vars.getOrElse(d, Set.empty)
          }.toSeq.map { v =>
            q"$v.addObservation($obName.score)"
          }

        case q"val $tname = if ($cond) { $tExpr } else { $fExpr }" =>
          tExpr match {
            case q"scappla.this.`package`.sample[$tDist]($tPrior, $tPosterior)" =>
              val tGuideVar = TermName(c.freshName())
              val tGuide = q"scappla.BBVIGuide[$tDist]($tPosterior)"
              guides += tGuideVar -> tGuide
              fExpr match {
                case q"scappla.this.`package`.sample[$fDist]($fPrior, $fPosterior)" =>
                  val fGuideVar = TermName(c.freshName())
                  val fGuide = q"scappla.BBVIGuide[$fDist]($fPosterior)"
                  guides += fGuideVar -> fGuide

                  val resVar = TermName(c.freshName())
                  vars += tname -> Set(resVar)
                  Seq(
                    q"""val $resVar = if ($cond)
                        $tGuideVar.sample($tPrior)
                     else
                        $fGuideVar.sample($fPrior)
                     """,
                    q"val $tname = $resVar.get"
                  ) ++ vars.values.toSet.flatten.filterNot(_ == resVar).map { varName =>
                    q"$varName.addVariable($resVar.modelScore, $resVar.guideScore)"
                  }
              }
            case _ =>
              Seq(
                q"val $tname = if ($cond) $tExpr else $fExpr"
              )
          }

        case q"val $tname = $expr" =>
          println(s"   VALDEF ${showRaw(expr.tpe)} : ${showRaw(expr)}")
          val exprVars = findVars(expr)
          val deps = exprVars.flatMap { ev =>
            vars.getOrElse(ev, Set.empty)
          }
          if (deps.nonEmpty) {
            vars += tname -> deps
          }
          Seq(
            q"val ${tname} = ${expr}"
          )

        // need to ensure that last expression is an RV
        case _ => Seq(tree)
      }

    }

    def findVars(tree: c.Tree): Set[TermName] = {
      tree match {
        case Ident(TermName(name)) =>
          Set(TermName(name))

        case q"$fn(..$args)" =>
          args.toSet.flatMap(arg => findVars(arg))

        case q"$a match { case ..$cases }" =>
          findVars(a)

        case _ => Set.empty
      }
    }

    def build(last: TermName, rType: c.Type): c.Tree = {
      val lastVars = vars(last)
      q"""new Variable[$rType] {

          import scappla.DValue._

          val get: $rType = ${last}

          val modelScore: Score = {
              ${(obs.map { t =>
                  q"$t.score": c.Tree
                } ++ vars.values.toSet.flatten.toSeq.map { t =>
                  q"$t.modelScore" : c.Tree
                }).reduceOption { (a, b) => q"DAdd($a, $b)" }
                .getOrElse(q"DValue.toConstant(0.0)")
              }
          }

          val guideScore: Score = {
              ${vars.values.toSet.flatten.toSeq.map { t =>
                  q"$t.guideScore": c.Tree
                }.reduceOption { (a, b) => q"DAdd($a, $b)" }
                .getOrElse(q"DValue.toConstant(0.0)")
              }
          }

          def addObservation(score: Score) = {
             ..${lastVars.toSeq.map { lv => q"$lv.addObservation(score)" }}
          }

          def addVariable(modelScore: Score, guideScore: Score) = {
             ..${lastVars.toSeq.map { lv => q"$lv.addVariable(modelScore, guideScore)" }}
          }

          def complete() = {
              ..${(obs.toSeq ++ vars.filterNot {
                    case (k, _) => args.contains(k)
                  }.values.toSet.flatten.toSeq.reverse)
                    .map { t =>
                      q"$t.complete()"
                    }
              }
          }
      }"""
    }

  }

  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[scappla.Model[X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]

    val visitor = new RVVisitor(fn.tree)
    val newBody =
      q"""new Model[$xType] {

        ..${visitor.guides.map {
            case (guideVar, guide) =>
              q"val ${guideVar} = ${guide}"
          }}

        def sample() = {
          ..${visitor.stmts}
        }
      }"""
    println("INFERRING")
    println(showCode(newBody))
    c.Expr[scappla.Model[X]](newBody)
  }

  def infer1[Y: WeakTypeTag, X: WeakTypeTag](fn: c.Expr[Y => X]): c.Expr[scappla.Model1[Y, X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]
    val yType = implicitly[WeakTypeTag[Y]]

    fn.tree match {
      case q"($argName: $argType) => $body" =>
        val varArgName = TermName("_upstreamArg")

        val visitor = new RVVisitor(body, Map(argName -> varArgName))
        val newBody =
          q"""new Model1[$yType, $xType] {

          ..${visitor.guides.map {
              case (guideVar, guide) =>
                q"val ${guideVar} = ${guide}"
            }}

          def apply($argName: $argType) = new Model[$xType] {

            private var $varArgName: Variable[_] = null

            override def withDeps(deps: Variable[_]): Model[$xType] = {
              $varArgName = deps
              this
            }

            def sample() = {
              ..${visitor.stmts}
            }
          }
        }"""
        println("INFERRING")
        println(showCode(newBody))
        c.Expr(newBody)

      case _ =>
        c.Expr(EmptyTree)
    }
  }

}
