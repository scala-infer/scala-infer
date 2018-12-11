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
      vars: Set[TermName],
      // wrapper to conform to original type
      // Rewriting function
      //   f: A => B
      // has type
      //   f': Variable[A] => Variable[B]
      // the wrapper then again has type A => B
      wrapper: Option[Tree] = None
  ) {

    def map(fn: Tree => Tree): RichTree = {
      RichTree(fn(tree), vars)
    }
  }

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

  class Scope(known: Map[TermName, RichTree]) {

    private val args: mutable.Set[TermName] = mutable.Set.empty
    private val refs: mutable.HashMap[TermName, RichTree] = mutable.HashMap.empty

    def isPreDefined(v: TermName): Boolean = {
      known.contains(v) || args.contains(v)
    }

    def isDefined(v: TermName): Boolean = {
      known.contains(v) || refs.contains(v)
    }

    def reference(v: TermName): RichTree = {
      if (isDefined(v)) {
        if (refs.contains(v)) {
          refs(v)
        } else {
          known(v)
        }
      } else {
        RichTree(Ident(v), Set.empty)
      }
    }

    def argument(name: c.universe.TermName, tree: RichTree): Scope = {
      args += name
      declare(name, tree)
    }

    def declare(v: TermName, deps: RichTree): Scope = {
      refs += v -> deps
      this
    }

    def push(): Scope = {
      new Scope(known ++ refs.toMap)
    }

  }

  class VariableAggregator {

    private val vars: mutable.ListBuffer[TermName] = new mutable.ListBuffer()

    private val obs: mutable.Set[TermName] = mutable.Set.empty

    private val fns: mutable.Set[TermName] = mutable.Set.empty

    private val completeable: mutable.ListBuffer[Tree] = new mutable.ListBuffer()

    def variable(v: TermName): VariableAggregator = {
      vars += v
      completeable += q"$v.node"
      this
    }

    def observation(o: TermName): VariableAggregator = {
      obs += o
      completeable += q"$o"
      this
    }

    def function(o: TermName): VariableAggregator = {
      fns += o
      completeable += q"$o"
      this
    }

    def buffer(o: TermName): VariableAggregator = {
      completeable += q"$o"
      this
    }

    def build(scope: Scope, result: RichTree, tpe: Type): RichTree = {
      val tree = q"""Variable[${tpe.widen}](${result.tree}, new BayesNode {

          import scappla.Real._

          val modelScore: Score = {${
        (obs.map { t =>
          q"$t.score": Tree
        } ++ vars.map { t =>
          q"$t.node.modelScore": Tree
        }).reduceOption { (a, b) => q"DAdd($a, $b)" }
            .getOrElse(q"Real.apply(0.0)")
      }}

          val guideScore: Score = {${
        vars.map { t =>
          q"$t.node.guideScore": Tree
        }.reduceOption { (a, b) => q"DAdd($a, $b)" }
            .getOrElse(q"Real.apply(0.0)")
      }}

          def addObservation(score: Score) = {..${
        result.vars.filter(!scope.isPreDefined(_)).toSeq.map { lv =>
          q"$lv.node.addObservation(score)"
        }
      }}

          def addVariable(modelScore: Score, guideScore: Score) = {..${
        result.vars.filter(!scope.isPreDefined(_)).toSeq.map {lv =>
          q"$lv.node.addVariable(modelScore, guideScore)"
        }
      }}

          def complete() = {..${
        completeable.reverse.map { c =>
          q"$c.complete()"
        }
      }}
      })"""
      RichTree(tree, Set.empty)
    }
  }

  class BlockVisitor(scope: Scope) {

    private val builder = new VariableAggregator()

    def visitBlockStmts(stmts: Seq[Tree]): Seq[RichTree] = {
      println(s"VISITINGN BLOCK ${showCode(q"{..$stmts}")}")
      val setup :+ last = stmts
      val richSetup = setup.flatMap(t => visitStmt(t))
      println(s"   TPE: ${last.tpe}")
      richSetup ++ (if (last.tpe =:= definitions.UnitTpe) {
        visitStmt(last) :+ builder.build(scope, RichTree(EmptyTree), definitions.UnitTpe)
      } else {
        visitExpr(last) { lastRt =>
          Seq(builder.build(scope, lastRt, last.tpe))
        }
      })
    }

    def visitApply(expr: Tree)(fn: RichTree => Seq[RichTree]): Seq[RichTree] = {
      expr match {
        case Ident(TermName(name)) if expr.tpe <:< typeOf[Function1[_, _]] =>
          println(s"   REFERNECE FN ${showRaw(expr, printTypes = true)}")
          val tname = TermName(name)
          if (scope.isDefined(TermName(name))) {
            fn(scope.reference(tname))
          } else {
            val varFn = c.freshName()
            val typeArgs = expr.tpe.widen.typeArgs
            println(s"n args: ${typeArgs.size}")
            val richFn = RichTree(
              q"""val $varFn : Variable[${TypeTree(typeArgs(0))}] => Variable[${TypeTree(typeArgs(1))}] =
                      in => Variable(${TermName(name)}(in.get), in.node)""")
            scope.declare(varFn, RichTree(q"$varFn", Set.empty))
            Seq(richFn) ++ fn(scope.reference(varFn))
          }
      }
    }

    def visitExpr(expr: Tree, doVar: Boolean = false)(fn: RichTree => Seq[RichTree]): Seq[RichTree] = {
      expr match {
        case Ident(TermName(name)) =>
          println(s"   REFERNECE ${showRaw(expr, printTypes = true)}")
          val tname = TermName(name)
          fn(scope.reference(tname))

        case Literal(value) =>
          println(s"   LITERAL ${showRaw(expr)}")
          fn(RichTree(expr))

        case q"{ ..$stmts }" if stmts.size > 1 =>
          val visitor = new BlockVisitor(scope.push())
          val newStmts = visitor.visitBlockStmts(stmts)

          // reduce list of variables to those known in the current scope
          val vars = newStmts.flatMap(_.vars).toSet.filter(scope.isDefined)

          val varName = TermName(c.freshName())
          builder.variable(varName)

          val varDef = q"{ ..${newStmts.map(_.tree)}}"
          val varExpr = q"val $varName = $varDef"
          Seq(RichTree(varExpr, vars)) ++
              (if (expr.tpe =:= definitions.UnitTpe) {
                Seq.empty
              } else {
                Seq(RichTree(q"$varName.get", Set(varName)))
              })

        case q"if ($cond) $tExpr else $fExpr" =>
          visitExpr(cond, true) { condRef =>
            println(s"  MATCHING IF ELSE $condRef")

            val ifVar = TermName(c.freshName())
            builder.variable(ifVar)

            val richTrue = visitSubExpr(tExpr)
            val richFalse = visitSubExpr(fExpr)
            val resultTree = q"$ifVar.get"
            Seq(RichTree(
              q"""val $ifVar = if (${condRef.tree})
                    ${richTrue.tree}
                  else
                    ${richFalse.tree}""",
              Set(ifVar)
            )) ++ condRef.vars.map { cv =>
              RichTree(q"$cv.node.addVariable($ifVar.node.modelScore, $ifVar.node.guideScore)")
            } ++ fn(
              RichTree.join(
                resultTree,
                RichTree.join(resultTree, richTrue, richFalse),
                RichTree(condRef.tree, condRef.vars + ifVar)
              )
            )
          }

        case q"$expr match { case ..$cases }" =>
          val mappedCases = cases.map { c =>
            val cq"$when => $result" = c
            val richResult = visitSubExpr(result)
            (richResult, cq"${reIdentPat(when)} => ${richResult.tree}")
          }
          visitExpr(expr, true) { matchName =>
            val matchVar = TermName(c.freshName())
            builder.variable(matchVar)
            val resultTree = q"$matchVar.get"
            val result = q"${matchName.tree} match { case ..${mappedCases.map{ _._2 }}}"
            val richResult = mappedCases.map { _._1 }.reduce(RichTree.join(result, _, _))
            Seq(RichTree(
              q"val $matchVar = $result",
              Set(matchVar)
            )) ++ fn(
              RichTree.join(resultTree, matchName, richResult)
            )
          }

        case q"scappla.this.`package`.sample[$tDist]($prior, $guide)" =>
          visitExpr(prior, doVar = true) { priorName =>
            visitExpr(guide, doVar = true) { guideName =>
              val tVarName = TermName(c.freshName())
//              println(s"MATCHING ${showRaw(tDist.tpe)} (${showCode(tDist)})")
              builder.variable(tVarName)
              scope.declare(tVarName, RichTree(q"$tVarName", (priorName.vars ++ guideName.vars) + tVarName))
              val ref = scope.reference(tVarName).map(tree => q"$tree.get")
              Seq(
                RichTree(
                  q"val $tVarName = ${guideName.tree}.sample(${priorName.tree})",
                  priorName.vars ++ guideName.vars
                )
              ) ++ fn(ref)
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

          val visitor = new BlockVisitor(newScope)
          val argDecls = newArgs.map { arg =>
            RichTree(q"val ${arg._1} = ${arg._2}.get", Set(arg._2))
          }
          newArgs.foreach(arg => {
            newScope.argument(arg._2, RichTree(q"${arg._2}", Set(arg._2)))
            newScope.declare(arg._1, RichTree(q"${arg._1}", Set(arg._2)))
          })

          val q"{..$stmts}" = body
          val newStmts = argDecls ++ visitor.visitBlockStmts(stmts)
          val newBody = q"{..${newStmts.map { _.tree }}}"
          val newVars = newStmts.flatMap { _.vars }.toSet.filter(scope.isDefined)

          val varName = TermName(c.freshName())
          val tpes: Seq[Type] = bargs.map {
            case q"$mods val $arg: $tpt = $argExpr" =>
              tpt.tpe
          }
          val wrapper =
            q"""new Function1[..${tpes :+ body.tpe}] with Completeable {

                var nodes : List[BayesNode] = Nil

              def apply(..${
              bargs.map {
                case q"$mods val $arg: $tpt = $argExpr" =>
                  val inlExpr = toExpr(visitExpr(argExpr)(Seq(_)))
                  q"$mods val $arg: $tpt = ${inlExpr.tree}"
              }
            }): ${body.tpe} = {${
              q"""val result = $varName(..${
                bargs.map {
                  case q"$mods val $arg: $tpt = $argExpr" =>
                    q"Variable($arg, ConstantNode)"
                }
              })
              val value = result.get
              nodes = result.node :: nodes
              value
              """
            }}

                def complete(): Unit = {
                  nodes.reverse.foreach(_.complete())
                }
              }"""
          scope.declare(varName, RichTree(q"$varName", newVars, Some(wrapper)))

          val newDefTree = q"val $varName = (..${newArgs.map(_._3)}) => $newBody"
          Seq(RichTree.join(
                newDefTree,
                RichTree(newDefTree, newVars),
                newArgs.map {_._4}.reduce(RichTree.join(newDefTree, _, _))
              )) ++ fn(scope.reference(varName))

        /*
         * local f: A => B
         *   has been transformed to
         *     f: Variable[A] => Variable[B]
         *     - sampling & observations allowed
         *
         * free  f: A => B
         *   is still
         *     f: A => B
         *     - no sampling, no observations
         *
         * can we unify these?
         *
         * In scope: Map[ f => enriched(f) ] ?
         * in RichTree itself?
         *
         * can we invoke functions with their dependencies passed along?
         */
        case q"$f.apply($o)" =>
          f match {
            case Ident(TermName(fname))
              if scope.isDefined(fname) && scope.reference(fname).wrapper.isDefined =>
              visitExpr(f) { richFn =>
                visitExpr(o) { richO =>
                  val varResult = TermName(c.freshName())
                  builder.variable(varResult)
                  val result = q"$varResult.get"
                  val nodes = richO.vars.map { t => q"$t.node" }
                  Seq(RichTree(
                    q"val $varResult = ${richFn.tree}.apply(scappla.Variable(${richO.tree}, new scappla.Dependencies(Seq(..$nodes))))"
                  )) ++ fn(
                    RichTree(result, richFn.vars ++ richO.vars + varResult)
                  )
                }
              }
            case _ =>
              visitExpr(f) { richFn =>
                visitExpr(o) { richO =>
                  val result = q"${richFn.tree}.apply(${richO.tree})"
                  fn(RichTree.join(result, richFn, richO))
                }
              }
          }

          /*
        case q"$f($o)" =>
          visitExpr(f) { richFn =>
            visitExpr(o) { richO =>
              val result = q"${richFn.tree}(${richO.tree})"
              fn(RichTree.join(result, richFn, richO))
            }
          }
          */

        case q"$f.$m[..$tpts](...$mArgs)" =>
          println(s"  FN MATCH (TYPE: ${expr.tpe}) ${showCode(expr)}")
          val mRes = (mArgs: List[List[Tree]]).map { args =>
            args.map { arg =>
              visitExpr(arg) { aName =>
                if (aName.wrapper.isEmpty) {
                  Seq(aName)
                } else {
                  val fnName = TermName(c.freshName())
                  builder.function(fnName)
                  Seq(
                    RichTree(q"val $fnName = ${aName.wrapper.get}"),
                    RichTree(q"$fnName")
                  )
                }
              }
            }
          }
          val defs = mRes.flatMap { res =>
            res.flatMap {
              _.dropRight(1)
            }
          }
          val newArgs = mRes.map { res =>
            res.map {
              _.last.tree
            }
          }
          visitExpr(f) { richFn =>
            val result = q"${richFn.tree}.$m[..$tpts](...$newArgs)"
            defs ++ fn(RichTree.join(
              result,
              richFn,
              mRes.flatten.flatten.reduceOption(
                RichTree.join(result, _, _)
              ).getOrElse(RichTree(EmptyTree))
            ))
          }

        case q"$subject.this" =>
          println(s"  MATCH THIS ${showCode(expr)}")
          fn(RichTree(expr))

        case q"$subject.$method" =>
          val expanded = method match {
            case TermName("apply") =>
              visitApply(subject) _
            case _ =>
              visitExpr(subject) _
          }
          expanded { richSubject =>
            fn(richSubject.map { t => q"$t.$method" })
          }

          /*
        case q"$subj[..$tpts]" if tpts.nonEmpty =>
          visitExpr(subj) { richSubject =>
            fn(richSubject.map { t => q"$t[..$tpts]"})
          }
          */

        case EmptyTree =>
          fn(RichTree(EmptyTree))

        case q"$v: $tpt" =>
          visitExpr(v) { richV =>
            fn(RichTree(q"${richV.tree}: $tpt", richV.vars))
          }

        case _ =>
          throw new RuntimeException(s"no match for ${showCode(expr)}")

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
            visitExpr(o) { richO =>
              val obName = TermName(c.freshName())
              val resultTree = q"val $obName : scappla.Observation = scappla.observeImpl[$tDist](${richDistName.tree}, ${richO.tree})"

              builder.observation(obName)

              // println(s"  VARIABLES in oDist: ${deps} => ${vars(deps.head) }")
              println(s" OBSERVE ${showRaw(resultTree)}")

              RichTree(resultTree, richDistName.vars) +:
                  richDistName.vars.toSeq.map { v =>
                    RichTree(q"$v.node.addObservation($obName.score)")
                  }
            }
          }

        case q"$mods val $tname : $tpt = $rhs" =>
          // FIXME: just pass tname to toANF?
          val TermName(name) = tname
//          println(s"  RECURRING into ${showCode(rhs)}")
          visitExpr(rhs) { exprName =>
              val fullExpr = if (rhs.tpe <:< typeOf[scappla.Expr[_]]) {
                println(s"  DIFFERENTIABLE ${showCode(rhs)}")
                builder.buffer(TermName(name))
                exprName.map { t => q"$t.buffer"}
              } else {
                exprName
              }
            scope.declare(TermName(name), fullExpr.copy(tree = q"${TermName(name)}"))
            Seq(
              fullExpr.map { t => q"val ${TermName(name)} = $t" }
            )
          }

        case _ if expr.tpe =:= definitions.UnitTpe =>
          visitExpr(expr)(Seq(_))
      }

    }

    def visitSubExpr(tExpr: Tree) = {
      val newScope = scope.push()
      val subVisitor = new BlockVisitor(newScope)
      val newSubStmts = if (tExpr.tpe =:= definitions.UnitTpe) {
        subVisitor.visitStmt(tExpr) :+
            subVisitor.builder.build(newScope, RichTree(EmptyTree), definitions.UnitTpe)
      } else {
        subVisitor.visitExpr(tExpr) { rtLast =>
          Seq(subVisitor.builder.build(newScope, rtLast, tExpr.tpe))
        }
      }
      toExpr(newSubStmts)
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

  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[scappla.Model[X]] = {
    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]

    val q"{..$stmts}" = fn.tree
    val visitor = new BlockVisitor(new Scope(Map.empty))
    val newStmts = visitor.visitBlockStmts(stmts)
    val newBody =
      q"""new Model[$xType] {

        def sample() = {
          ..${newStmts.map{_.tree}}
        }
      }"""
    println("INFERRING")
    println(showCode(newBody))
    c.Expr[scappla.Model[X]](newBody)
  }

}
