package scappla

import scala.collection.mutable
import scala.reflect.api.{Names, Trees}
import scala.reflect.macros.blackbox

class Macros(val c: blackbox.Context) {


  import c.universe._
  import c.internal._
  import decorators._

  /**
    * Want to be able to express forEach((entry => ... observe(entry), data), as well as
    * val sprinkler = (rain : Boolean) => ...
    * val rain = sample(...)
    * val sprinkled = sprinkler(rain)
    *
    * Transform a function A => B to Variable[A] => Variable[B]
    * (and A => B => C to Variable[A] => Variable[B] => Variable[C], etcetera)
    *
    * Track dependencies between variables.  This allows us to reconstruct the Bayesian Network
    * of nodes.
    * Identifiers that resolve to Random Variables will cause those RVs to be included in the
    * dependencies of the variable that's being constructed.
    *
    * In A Normal Form representation, all arguments to a function are trivial, i.e. just
    * an identifier.
    */

  case class RichTree(
      // return type of the stack frame
      tree: Tree,
      // resolved references - Variable
      vars: Set[TermName]
  )

  object RichTree {

    def apply(tree: Tree): RichTree = RichTree(tree, Set.empty)

    def join(tree: Tree, a: RichTree, b: RichTree): RichTree =
      RichTree(tree, a.vars ++ b.vars)

    class Builder(bound: Set[TermName]) {

      // locally created Variables
      private val vars = mutable.Set.empty[TermName]

      def ref(variable: TermName): Builder = {
        if (bound.contains(variable)) {
          vars += variable
        }
        this
      }

      def build(tree: Tree): RichTree = {
        RichTree(
          tree,
          vars.toSet
        )
      }
    }
  }

  case class TreeContext(bound: Set[TermName]) {

    def builder = new RichTree.Builder(bound)
  }

/*
  case class VariableAggregator(
      tpe: Type,
      result: TermName,
      // locally created variables, with their dependencies
      vars: Map[TermName, Set[TermName]],
      // observations
      obs: Set[TermName]
  )
*/

  class Scope(known: Map[TermName, Set[TermName]]) {

    private val refs: mutable.HashMap[TermName, Set[TermName]] = mutable.HashMap.empty

    def isDefined(v: TermName): Boolean = {
      known.contains(v) || refs.contains(v)
    }

    def reference(v: TermName): RichTree = {
      if (isDefined(v)) {
        RichTree(Ident(v), refs(v))
      } else {
        RichTree(Ident(v), Set.empty)
      }
    }

    def declare(v: TermName, deps: Set[TermName]): Scope = {
      refs += v -> deps
      this
    }

    def push(): Scope = {
      new Scope(known ++ refs.toMap)
    }

  }

  class GuideAggregator {

    private val guides = new scala.collection.mutable.ListBuffer[Tree]()

    def define(guide: Tree): GuideAggregator = {
      guides += guide
      this
    }

    def build(): Seq[Tree] = guides.toSeq
  }

  class VariableAggregator {

    private val vars: mutable.ListBuffer[TermName] = new mutable.ListBuffer()

    private val obs: mutable.Set[TermName] = mutable.Set.empty

    def variable(v: TermName): VariableAggregator = {
      vars += v
      this
    }

    def observation(o: TermName): VariableAggregator = {
      obs += o
      this
    }

    def build(result: RichTree): RichTree = {
      val tree = q"""new Variable[${result.tree.tpe}] {

          import scappla.Real._

          val get: ${result.tree.tpe} = ${result.tree}

          val modelScore: Score = {${
        (obs.map { t =>
          q"$t.score": Tree
        } ++ vars.map { t =>
          q"$t.modelScore": Tree
        }).reduceOption { (a, b) => q"DAdd($a, $b)" }
            .getOrElse(q"Real.apply(0.0)")
      }}

          val guideScore: Score = {${
        vars.map { t =>
          q"$t.guideScore": Tree
        }.reduceOption { (a, b) => q"DAdd($a, $b)" }
            .getOrElse(q"Real.apply(0.0)")
      }}

          def addObservation(score: Score) = {..${
        result.vars.toSeq.map { lv => q"$lv.addObservation(score)" }
      }}

          def addVariable(modelScore: Score, guideScore: Score) = {..${
        result.vars.toSeq.map { lv => q"$lv.addVariable(modelScore, guideScore)" }
      }}

          def complete() = {..${
        (obs.toSeq ++ vars.reverse).map { t =>
          q"$t.complete()"
        }
      }}
      }"""
      RichTree(tree, Set.empty)
    }
  }

  class BlockVisitor(scope: Scope, guides: GuideAggregator) {

    private val builder = new VariableAggregator()

    def visitBlockStmts(stmts: Seq[Tree]): Seq[RichTree] = {
      val setup :+ last = stmts
      val richSetup = setup.flatMap(t => visitStmt(t))
      richSetup ++ visitExpr(last) { lastRt =>
        Seq(builder.build(lastRt))
      }
    }

    def visitExpr(expr: Tree, doVar: Boolean = false)(fn: RichTree => Seq[RichTree]): Seq[RichTree] = {
      expr match {
        case Ident(TermName(name)) =>
          val tname = TermName(name)
          fn(scope.reference(tname))

        case Literal(value) =>
          println(s"   LITERAL ${showRaw(expr)}")
          fn(RichTree(expr))

        case q"{ ..$stmts }" =>
          val visitor = new BlockVisitor(scope.push(), guides)
          val newStmts = visitor.visitBlockStmts(stmts)

          // reduce list of variables to those known in the current scope
          val vars = newStmts.flatMap(_.vars).toSet.filter(scope.isDefined)

          val varName = c.freshName()
          builder.variable(varName)

          val varExpr = q"val $varName = { ..${newStmts.map(_.tree)}}"
          Seq(
            RichTree(varExpr, vars),
            RichTree(q"$varName.get", Set(varName))
          )

        case q"if ($cond) $tExpr else $fExpr" =>
          visitExpr(cond, true) { condRef =>
            println(s"  MATCHING IF ELSE $condRef")
            val richTrue = toExpr(visitExpr(tExpr)(Seq(_)))
            val richFalse = toExpr(visitExpr(fExpr)(Seq(_)))
            val resultTree =
              q"""if (${condRef.tree})
                    ${richTrue.tree}
                  else
                    ${richFalse.tree}"""
            fn(
              RichTree.join(
                resultTree,
                RichTree.join(resultTree, richTrue, richFalse),
                condRef
              )
            )
          }

        case q"$expr match { case ..$cases }" =>
          val mappedCases = cases.map { c =>
            val cq"$when => $result" = c
            val richResult = toExpr(visitExpr(result)(Seq(_)))
            (richResult, cq"${reIdentPat(when)} => ${richResult.tree}")
          }
          visitExpr(expr, true) { matchName =>
            val result = q"${matchName.tree} match { case ..${mappedCases.map{ _._2 }}}"
            val richResult = mappedCases.map { _._1 }.reduce(RichTree.join(result, _, _))
            fn(richResult)
          }

        case q"scappla.this.`package`.sample[$tDist]($prior, $posterior)" =>
          visitExpr(prior, doVar = true) { priorName =>
            visitExpr(posterior, doVar = true) { posteriorName =>
              val guideVar = TermName(c.freshName())
              val tVarName = TermName(c.freshName())
//              println(s"MATCHING ${showRaw(tDist.tpe)} (${showCode(tDist)})")
              val guide = tDist.tpe match {
                case tpe if tpe =:= typeOf[Real] =>
                  q"scappla.ReparamGuide(${posteriorName.tree})"
                case _ =>
                  q"scappla.BBVIGuide[$tDist](${posteriorName.tree})"
              }
              guides.define(q"val ${guideVar} = ${guide}")
              builder.variable(tVarName)
              scope.declare(tVarName, (priorName.vars ++ posteriorName.vars) + tVarName)
              Seq(
                RichTree(
                  q"val $tVarName = $guideVar.sample(${priorName.tree})",
                  priorName.vars
                )
              ) ++ fn(scope.reference(tVarName))
            }
          }

        case q"(..$bargs) => $body" =>
          val newArgs: Seq[(TermName, TermName, Tree, RichTree)] = bargs.map {
            case q"$mods val $arg: $tpt = $argExpr" =>
              println(s"    FN VALDEF ${showRaw(arg)}")
              val TermName(tname) = arg
              val varName = TermName(c.freshName(tname))
              val inlExpr = toExpr(visitExpr(argExpr)(Seq(_)))
              (
                  TermName(tname),
                  varName,
                  q"$mods val $varName: ${tq"Variable[$tpt]"} = ${inlExpr.tree}",
                  inlExpr
              )
          }

          val newScope = scope.push()
          newArgs.map { _._1 }.foreach(arg => newScope.declare(arg, Set(arg)))

          val visitor = new BlockVisitor(newScope, guides)
          val argDecls = newArgs.map { arg =>
            RichTree(q"val ${arg._1} = ${arg._2}.get", Set(arg._2))
          }
          val q"{..$stmts}" = body
          val newStmts = visitor.visitBlockStmts(stmts)
          val newBody = q"{..${newStmts.map { _.tree }}}"
          val newVars = newStmts.flatMap { _.vars }.toSet.filter(scope.isDefined)

          val varName = TermName(c.freshName())
          scope.declare(varName, newVars)

          val newDefTree = q"val $varName = (..${newArgs.map(_._3)}) => $newBody"
          Seq(RichTree.join(
                newDefTree,
                RichTree(newDefTree, newVars),
                newArgs.map {_._4}.reduce(RichTree.join(newDefTree, _, _))
              )) ++ fn(scope.reference(varName))

/*
        case _ =>
          val varName = TermName(c.freshName())
          val stmts :+ last = visitStmt(expr)
          if (doVar || stmts.nonEmpty) {
            if (expr.tpe =:= definitions.UnitTpe) {
//              println(s"   LAST EXPRESSION IS EMPTY ${showCode(last.tree, printTypes = true)}")
              (stmts :+ last) ++ fn(EmptyTree)
            } else {
              (stmts :+ q"val $varName = $last") ++ fn(Ident(varName))
            }
          } else {
            fn(last)
          }
*/
      }
    }

    def visitStmt(expr: Tree): Seq[RichTree] = {
      expr match {
        // observe needs to produce (resultVar, dependentVars..)
        case q"scappla.this.`package`.observe[$tDist]($oDist, $o)" =>
          visitExpr(oDist, true) { richDistName =>
            val obName = TermName(c.freshName())
            val resultTree = q"val $obName : scappla.Observation = scappla.observeImpl[$tDist](${richDistName.tree}, $o)"

            builder.observation(obName)

            // println(s"  VARIABLES in oDist: ${deps} => ${vars(deps.head) }")
            println(s" OBSERVE ${showRaw(resultTree)}")

            RichTree(resultTree, richDistName.vars) +:
                richDistName.vars.toSeq.map { v =>
                  RichTree(q"$v.addObservation($obName.score)")
                }
          }

        case q"val $tname : $tpt = $rhs" =>
          // FIXME: just pass tname to toANF?
          val Ident(TermName(name)) = tname
//          println(s"  RECURRING into ${showCode(rhs)}")
          visitExpr(rhs) { exprName =>
            scope.declare(TermName(name), exprName.vars)
            Seq(
              RichTree(q"val $tname = ${exprName.tree}", exprName.vars)
            )
          }

        case _ if expr.tpe =:= definitions.UnitTpe =>
          visitExpr(expr)(Seq(_))
      }

    }

    def reIdentPat(expr: Tree): Tree = {
      println(s"   REMAPPING ${showRaw(expr)}")
      expr match {
        case pq"$tName @ $pat" =>
          println(s"     TERM ${showRaw(tName)}")
          val TermName(name) = tName
          pq"${TermName(name)} @ $pat"

        case pq"$ref(..$pats)" =>
          pq"$ref(..${pats.map { pat => reIdentPat(pat) }})"

        case _ =>
          expr
      }
    }

    def toExpr(exprs: Seq[RichTree]): RichTree = {
      if (exprs.size == 1) {
        exprs.head
      } else {
        RichTree(
          q"{ ..${exprs.map(_.tree)} }",
          exprs.flatMap(_.vars).toSet
        )
      }
    }

  }

  /*
    class RVVisitor(body: c.Tree, args: Map[TermName, TermName] = Map.empty) {

      println(s"VISITING ${showCode(body, printTypes = true)}")

      private val vars = scala.collection.mutable.HashMap[TermName, Set[TermName]]()
      for {(arg, v) <- args} vars += arg -> Set(v)

      private val obs = scala.collection.mutable.HashSet[TermName]()
      val guides = scala.collection.mutable.HashMap[TermName, c.Tree]()

      var varStack: List[TermName] = Nil

      val (stmts, lastName) = body match {
        case q"{..$stmts }" =>
          val setup :+ last = stmts
          println(s"LAST [${last.tpe}]: ${showCode(last)}")
          val (prelim, lastName) = last match {
            case Ident(TermName(name)) =>
              (setup, TermName(name))
            case _ =>
              val lastName = TermName(c.freshName())
              (setup :+ q"val ${lastName} = ${last}", lastName)
          }
          val newSetup = prelim.flatMap(toANF)
          println(s"VARS: ${vars}")
          val newLast = build(lastName, body.tpe)
          (newSetup :+ newLast, lastName)
      }

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
            if (doVar || stmts.nonEmpty) {
              if (expr.tpe =:= definitions.UnitTpe) {
                println(s"   LAST EXPRESSION IS EMPTY ${showCode(last, printTypes = true)}")
                (stmts :+ last) ++ fn(EmptyTree)
              } else {
                (stmts :+ q"val $varName = $last") ++ fn(Ident(varName))
              }
            } else {
              fn(last)
            }
        }
      }


      def toANF(tree: c.Tree): Seq[c.Tree] = {

        println(s"MAMTCHINIG ${showCode(tree, printTypes = true)}")
        tree match {

          case q"$s.$method($o)" =>
            expand(s) { sName =>
              expand(o) { oName =>
                Seq(q"$sName.$method($oName)")
              }
            }


          case q"$fn($o)" =>
            println(s"  FN MATCH (1 arg) ${showRaw(tree, printTypes = true, printIds = true)}")
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
            val defs = res.flatMap {
              _.dropRight(1)
            }
            val newArgs = res.map {
              _.last
            }
            defs :+ q"$fn(..$newArgs)"

          case q"val $tname = if ($cond) { $tExpr } else { $fExpr }" =>

          case q"($arg: $aty) => $fnBody" =>
            println(s"     FN TYPE: [${fnBody.tpe}]")
            // val fnRV = new RVVisitor(fnBody)
            // val newBody = visit(fnBody)
            val res =
            q"""new Function1[${aty}, ${fnBody.tpe}] with Completeable {

                var score : Real = Real.apply(0.0)
                var completions : List[Completeable] = Nil

                def apply($arg: $aty) = ${inl(toANF(fnBody))}

                def complete(): Unit = {
                    completions.foreach(_.complete())
                }
            }"""
            println(s"   RESULT: ${showCode(res)}")

            val exprVars = findVars(res)
            val deps = exprVars.flatMap { ev =>
              vars.getOrElse(ev, Set.empty)
            }
            val tname = varStack.head
            vars += tname -> deps
            Seq(
              res
            )
  //                case q"$mods val $tname : $tpt = $expr" =>
  //                  println(s"    VALDEFF ${showRaw(tname)}")
  //                  val TermName(name) = tname
  //                  val stmts :+ last = toANF(expr)
  //                  stmts :+ q"$mods val ${TermName(name)}: $tpt = $last"

          case Ident(TermName(name)) =>
            Seq(Ident(TermName(name)))

          case q"$expr: $tpt" =>
            expand(expr) { eName =>
              Seq(q"$eName: $tpt")
            }

  //        case q"$expr.$tname" =>
  //          expand(expr) { eName =>
  //            Seq(q"$eName.$tname")
  //          }

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

            import scappla.Real._

            val get: $rType = ${last}

            val modelScore: Score = {
                ${
          (obs.map { t =>
            q"$t.score": c.Tree
          } ++ vars.values.toSet.flatten.toSeq.map { t =>
            q"$t.modelScore": c.Tree
          }).reduceOption { (a, b) => q"DAdd($a, $b)" }
              .getOrElse(q"Real.apply(0.0)")
        }
            }

            val guideScore: Score = {
                ${
          vars.values.toSet.flatten.toSeq.map { t =>
            q"$t.guideScore": c.Tree
          }.reduceOption { (a, b) => q"DAdd($a, $b)" }
              .getOrElse(q"Real.apply(0.0)")
        }
            }

            def addObservation(score: Score) = {
               ..${lastVars.toSeq.map { lv => q"$lv.addObservation(score)" }}
            }

            def addVariable(modelScore: Score, guideScore: Score) = {
               ..${lastVars.toSeq.map { lv => q"$lv.addVariable(modelScore, guideScore)" }}
            }

            def complete() = {
                ..${
          (obs.toSeq ++ topoSortVars(vars).filterNot {
            case k => args.contains(k)
          }.reverse).map { t =>
            q"$t.complete()"
          }
        }
            }
        }"""
      }
    }
  */

  /*
  def topoSortVars(vars: Map[TermName, Set[TermName]]): Seq[TermName] = {
    val visited = scala.collection.mutable.ListBuffer[TermName]()

    def visitVar(tname: TermName): Unit = {
      if (vars.contains(tname) && !visited.contains(tname)) {
        visited += tname
        for {
          dep <- vars(tname)
        } {
          visitVar(dep)
        }
      }
    }

    for {
      dep <- vars.keys
    } {
      visitVar(dep)
    }

    visited
  }
  */

  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[scappla.Model[X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]

    val guides = new GuideAggregator()
    val q"{..$stmts}" = fn.tree
    val visitor = new BlockVisitor(new Scope(Map.empty), guides)
    val newStmts = visitor.visitBlockStmts(stmts)
    val newBody =
      q"""new Model[$xType] {

        ..${guides.build()}

        def sample() = {
          ..${newStmts.map{_.tree}}
        }
      }"""
    println("INFERRING")
    println(showCode(newBody))
    c.Expr[scappla.Model[X]](newBody)
  }

}
