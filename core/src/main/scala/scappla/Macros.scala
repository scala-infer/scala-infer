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
      wrapper: Option[TypeName] = None,

      // arguments that the tree depends on
      // only available when the tree is a function
      args: Set[Int] = Set.empty
  ) {

    val isFn = {
      wrapper.isDefined
    }

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

  case class RichBlock(
    setup: Seq[Tree],
    result: RichTree
  ) {

    def flatMap(fn: RichTree => RichBlock) = map(fn)

    def map(fn: RichTree => RichBlock) = {
      val block = fn(result)
      RichBlock(
        setup ++ block.setup,
        block.result
      )
    }

  }

  object RichBlock {
    def apply(stmts: Seq[RichTree]): RichBlock = {
      RichBlock(
        stmts.dropRight(1).map { _.tree },
        stmts.last
      )
    }
  }

  case class RichArg(
    mods: Modifiers,
    tpe: Tree,
    origName: TermName,
    oldArgDecl: Tree
  )


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

  class Scope(known: Map[Name, RichTree]) {

    private val refs: mutable.HashMap[Name, RichTree] = mutable.HashMap.empty

    def isDefined(v: Name): Boolean = {
      known.contains(v) || refs.contains(v)
    }

    def isDeclared(v: Name): Boolean = {
      refs.contains(v)
    }

    def reference(v: Name): RichTree = {
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

    private val completeable: mutable.ListBuffer[Tree] = new mutable.ListBuffer()

    private var built = false

    def variable(v: TermName): VariableAggregator = {
      if (built) {
        throw new RuntimeException("Builder already complete - cannot add variable")
      }
      vars += v
      completeable += q"$v.node"
      this
    }

    def observation(o: TermName): VariableAggregator = {
      if (built) {
        throw new RuntimeException("Builder already complete - cannot add observation")
      }
      obs += o
      completeable += q"$o"
      this
    }

    def buffer(o: TermName): VariableAggregator = {
      if (built) {
        throw new RuntimeException("Builder already complete - cannot add buffer")
      }
      completeable += q"$o"
      this
    }

    override def toString(): String = {
      s"  VARS: ${vars.mkString(",")}; OBS: ${obs.mkString(",")}"
    }

    def build(scope: Scope, result: RichTree, tpe: Type): RichTree = {
      built = true
      // println(s"    BUILDING $this (${scope.refs}) - ${result.vars.filter(scope.isDeclared)}")
      val tree = if (obs.isEmpty
          && vars.isEmpty
          && completeable.isEmpty
          && result.vars.forall(v => !scope.isDeclared(v))
      ) {
        q"""scappla.Variable[${tpe.widen}](${result.tree}, scappla.ConstantNode)"""
      } else if (obs.isEmpty
          && vars.size == 1
          && completeable.size == 1
      ) {
        q"""scappla.Variable[${tpe.widen}](${result.tree}, ${vars.head}.node)"""
      } else {
        q"""scappla.Variable[${tpe.widen}](${result.tree}, new scappla.BayesNode {

          val modelScore = {${
          (obs.map { t =>
            q"$t.score": Tree
          } ++ vars.map { t =>
            q"$t.node.modelScore": Tree
          }).reduceOption { (a, b) => q"$a.+($b)" }
              .getOrElse(q"scappla.Value.apply(0.0)")
          }}.buffer

          val guideScore = {${
          vars.map { t =>
            q"$t.node.guideScore": Tree
          }.reduceOption { (a, b) => q"$a.+($b)" }
              .getOrElse(q"scappla.Value.apply(0.0)")
          }}.buffer

          def addObservation(score: scappla.Score) = {..${
          result.vars.filter(scope.isDeclared).toSeq.map { lv =>
            q"$lv.node.addObservation(score)"
          }
          }}

          def addVariable(modelScore: scappla.Score, guideScore: scappla.Score) = {..${
          result.vars.filter(scope.isDeclared).toSeq.map { lv =>
            q"$lv.node.addVariable(modelScore, guideScore)"
          }
          }}

          def complete() = {..${
          Seq(
            q"modelScore.complete()",
            q"guideScore.complete()"
          ) ++
          completeable.reverse.map { c =>
            q"$c.complete()"
          }
          }}
        })"""
      }
      RichTree(tree, result.vars.filter(v => !scope.isDeclared(v)))
    }
  }

  class BlockVisitor(scope: Scope) {

    private val builder = new VariableAggregator()

    def visitBlockStmts(stmts: Seq[Tree]): RichBlock = {
//      println(s"VISITINGN BLOCK ${showCode(q"{..$stmts}")}")
      val setup :+ last = stmts
      val richSetup = setup.flatMap(t => visitStmt(t))
//      println(s"   TPE: ${last.tpe}")
      if (last.tpe =:= definitions.UnitTpe) {
        val lastSetup = visitStmt(last)
        val lastVar = builder.build(scope, RichTree(EmptyTree), definitions.UnitTpe)
        RichBlock(
          richSetup ++ lastSetup,
          lastVar
        )
      } else {
        val lastBlock = visitExpr(last) 
        val lastExpr = builder.build(scope, lastBlock.result, last.tpe)
        RichBlock(
          richSetup ++ lastBlock.setup,
          lastExpr
        )
      }
    }

    def visitApply(expr: Tree): RichBlock = {
      expr match {
        case Ident(TermName(name)) if expr.tpe <:< typeOf[Function1[_, _]] =>
//          println(s"   REFERNECE FN ${showRaw(expr, printTypes = true)}")
          val tname = TermName(name)
          if (scope.isDefined(TermName(name))) {
            val ref = scope.reference(tname)
            RichBlock(Seq.empty, ref)
          } else {
            val varFn = TermName(c.freshName())
            val typeArgs = expr.tpe.widen.typeArgs
//            println(s"n args: ${typeArgs.size}")
            val richFn =
              q"""val $varFn : ${TypeTree(typeArgs(0))} => scappla.Variable[${TypeTree(typeArgs(1))}] =
                      in => scappla.Variable(${TermName(name)}(in), scappla.ConstantNode)"""
            scope.declare(varFn, RichTree(q"$varFn", Set.empty))
            RichBlock(
              Seq(richFn),
              RichTree(q"$varFn", Set.empty)
            )
          }
      }
    }

    def visitExpr(expr: Tree, resultVarName: Option[String] = None): RichBlock = {
      expr match {
        case Ident(TermName(name)) =>
//          println(s"   REFERNECE ${showRaw(expr, printTypes = true)}")
          val tname = TermName(name)
          val ref = scope.reference(tname)
          RichBlock(Seq.empty, ref)

        case Literal(value) =>
//          println(s"   LITERAL ${showRaw(expr)}")
          val ref = RichTree(expr)
          RichBlock(Seq.empty, ref)

        case q"{ ..$stmts }" if stmts.size > 1 =>
          // println(s"   BLOCK ${showCode(expr)}")
          val visitor = new BlockVisitor(scope.push())
          val newStmts = visitor.visitBlockStmts(stmts)

          // reduce list of variables to those known in the current scope
          val vars: Set[TermName] = newStmts.result.vars.toSet.filter(scope.isDefined)

          val varDef = q"{ ..${newStmts.setup :+ newStmts.result.tree}}"
          /*if (expr.tpe =:= definitions.UnitTpe) {
            RichBlock(
              Seq(varDef),
              RichTree(EmptyTree, vars)
            )
          } else { */
            val varName = TermName(c.freshName())
            builder.variable(varName)
            scope.declare(varName, RichTree(EmptyTree))

            val varExpr = q"val $varName = $varDef"
            RichBlock(
              Seq(varExpr),
              RichTree(q"$varName.get", vars)
            )
          // }

        case q"if ($cond) $tExpr else $fExpr" =>
          for {
            condRef <- visitExpr(cond)
          } yield {
//            println(s"  MATCHING IF ELSE $condRef")

            val ifVar = TermName(c.freshName())
            builder.variable(ifVar)
            scope.declare(ifVar, RichTree(EmptyTree))

            val richTrue = visitSubExpr(tExpr)
            // println(s"  IF TRUE $condRef: ${richTrue.vars}")
            val richFalse = visitSubExpr(fExpr)
            // println(s"  IF FALSE $condRef: ${richTrue.vars}")

            // branch and add the scores to the variables sampled in the current scope
            // The scores of variables and observations in the branches are added
            // to the variables defined in the current scope.
            val setup: Seq[Tree] = (
                q"""val $ifVar = if (${condRef.tree})
                    ${richTrue.tree}
                  else
                    ${richFalse.tree}"""
              ) +: (condRef.vars ++ richTrue.vars ++ richFalse.vars)
                  .filter(scope.isDeclared)
                  .toSeq.map { cv =>
                q"$cv.node.addVariable($ifVar.node.modelScore, $ifVar.node.guideScore)"
              }

            RichBlock(
              setup,
              RichTree(
                q"$ifVar.get",
                (condRef.vars ++ richTrue.vars ++ richFalse.vars)
                  .filter(rv => !scope.isDeclared(rv)) +
                  ifVar
              )
            )
          }

        case q"$expr match { case ..$cases }" =>
          val mappedCases = cases.map { c =>
            val cq"$when => $result" = c
            val richResult = visitSubExpr(result)
            // println(s"   CASE (${richResult.vars}): ${showCode(richResult.tree)}")
            (richResult.vars, cq"${reIdentPat(when)} => ${richResult.tree}")
          }

          val matchName = visitExpr(expr)
          val matchVar = TermName(c.freshName())
          val caseVars = mappedCases.map { _._1 }.flatten
          builder.variable(matchVar)
          scope.declare(matchVar, RichTree(EmptyTree))

          // val fnResult = fn(RichTree(q"$matchVar.get"))
          RichBlock(
            (matchName.setup :+ 
              q"""val $matchVar = ${matchName.result.tree} match {
                  case ..${mappedCases.map{ _._2 }}
                }""") ++ (matchName.result.vars ++ caseVars)
                  .filter(scope.isDeclared)
                  .toSeq.map { cv =>
              q"$cv.node.addVariable($matchVar.node.modelScore, $matchVar.node.guideScore)"
            },
            RichTree(
              q"$matchVar.get",
              (matchName.result.vars ++ caseVars) + matchVar
            )
          )

        case q"scappla.`package`.sample[$tDist]($prior, $guide)" =>
          for {
            priorName <- visitExpr(prior)
            guideName <- visitExpr(guide)
          } yield {
            val tVarName = TermName(c.freshName())
            // println(s"  SAMPLE MATCHING ${showRaw(tDist.tpe)} (${showCode(tDist)})")
            val interpreter = scope.reference(TermName("interpreter"))
            builder.variable(tVarName)
            scope.declare(tVarName, RichTree(EmptyTree))

            RichBlock(
              Seq(q"val $tVarName = ${guideName.tree}.sample(${interpreter.tree}, ${priorName.tree})") ++
              (priorName.vars ++ guideName.vars)
                  .toSeq.filter(scope.isDeclared).map { rv =>
                q"$rv.node.addVariable($tVarName.node.modelScore, $tVarName.node.guideScore)"
              },
              RichTree(q"$tVarName.get", Set(tVarName))
            )
          }

        case q"(..$bargs) => $body" =>
          val varName = TermName(c.freshName())
          visitFunction(varName, bargs, body)

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
          // println(s"  FN APPLY (TYPE: ${expr.tpe}) ${showCode(expr)}")
          f match {
            case Ident(fname)
                if scope.isDefined(fname) && scope.reference(fname).isFn =>
              for {
                richO <- visitExpr(o)
              } yield {
                // println(s"APPLYING FN ${fname}")
                val richFn = scope.reference(fname)
                val varResult = TermName(c.freshName())
                builder.variable(varResult)
                scope.declare(varResult, RichTree(EmptyTree))

                val vars = richFn.vars ++ richO.vars
                val argVars = Seq(richO.vars)
                RichBlock(
                  q"val $varResult = ${richFn.tree}.apply(${richO.tree})" +:
                  vars.filter(scope.isDeclared).toSeq.map { rv =>
                    q"$rv.node.addVariable($varResult.node.modelScore, $varResult.node.guideScore)"
                  },
                  RichTree(
                    q"$varResult.get",
                    richFn.args.flatMap { i => argVars(i) } + varResult
                  )
                )
              }

            case _ =>
              for {
                richFn <- visitExpr(f)
                richO <- visitExpr(o)
              } yield {
                // println(s"APPLYING UNKNOWN FN ${f}")
                RichBlock(
                  Seq.empty,
                  RichTree(
                    q"${richFn.tree}.apply(${richO.tree})",
                    richFn.vars ++ richO.vars
                  )
                )
              }
          }

        case q"$f.$m[..$tpts](...$mArgs)" if mArgs.size > 0 =>
          // println(s"  FN MATCH (TYPE: ${expr.tpe}) ${showCode(expr)}")
          val accumulator = TermName(c.freshName())
          var needAccumulator = false
          val mRes = (mArgs: List[List[Tree]]).map { args =>
            args.map { arg =>
              val RichBlock(setup, aName) = visitExpr(arg)
              if (!aName.isFn) {
                RichBlock(setup, aName)
              } else {
                val fnName = TermName(c.freshName())
                needAccumulator = true
                RichBlock(
                  setup :+ q"val $fnName = new ${aName.wrapper.get}($accumulator)",
                  RichTree(q"$fnName", Set.empty)
                )
              }
            }
          }

          for {
            richFn <- visitExpr(f)
          } yield {
            val defs = mRes.flatMap { res =>
              res.flatMap { _.setup }
            }
            val newArgs = mRes.map { res =>
              res.map { _.result.tree }
            }
            val vars = richFn.vars ++ mRes.flatten.flatMap { _.result.vars }

            if (!needAccumulator) {
              RichBlock(
                defs.toSeq,
                RichTree(
                  q"${richFn.tree}.$m[..$tpts](...$newArgs)",
                  vars
                )
              )
            } else {
              val resultName = TermName(c.freshName())
              val accumulatorVar = TermName(c.freshName())
              builder.variable(accumulatorVar)
              scope.declare(accumulatorVar, RichTree(EmptyTree))

              RichBlock(
                (q"val $accumulator = new scappla.Accumulator()" +: defs.toSeq :+
                q"val $resultName = ${richFn.tree}.$m[..$tpts](...$newArgs)" :+ 
                q"val $accumulatorVar = $accumulator.toVariable($resultName)") ++
                vars.filter(scope.isDeclared).toSeq.map { rv =>
                  q"$rv.node.addVariable($accumulatorVar.node.modelScore, $accumulatorVar.node.guideScore)"
                },
                RichTree(
                  q"$resultName",
                  vars + accumulatorVar
                )
              )
            }
          }

        case q"$f(..$args)" =>
          // println(s"  FN VAL (TYPE: ${expr.tpe}) ${showCode(expr)}")
          f match {
            case Ident(fname)
                if scope.isDefined(fname) && scope.reference(fname).isFn =>
              for {
                richFn <- visitExpr(f)
              } yield {
                // println(s"APPLYING KNOWN METHOD ${fname}")
                val newArgs = args.map { visitExpr(_) }

                val varResult = TermName(c.freshName())
                builder.variable(varResult)
                scope.declare(varResult, RichTree(EmptyTree))

                val argVars = newArgs.map { _.result.vars }.flatten.toSet
                RichBlock(
                  newArgs.map { _.setup }.flatten ++ Seq(
                    q"val $varResult = ${richFn.tree}(..${newArgs.map { _.result.tree }})"
                  ) ++ newArgs.flatMap { _.result.vars }
                    .filter{ scope.isDeclared }
                    .map { v =>
                      q"$v.node.addVariable($varResult.node.modelScore, $varResult.node.guideScore)"
                    },
                  RichTree(
                    q"$varResult.get",
                    richFn.vars ++ newArgs.flatMap { _.result.vars } ++ Set(varResult)
                  )
                )
              }

            case _ =>
              // println(s"APPLYING UNNOWN FUNCTION ${showRaw(f)}")
              for {
                richFn <- visitExpr(f)
              } yield {
                val newArgs = args.map { visitExpr(_) }
                RichBlock(
                  newArgs.map { _.setup }.flatten,
                  RichTree(
                    q"${richFn.tree}(..${newArgs.map { _.result.tree }})",
                    richFn.vars ++ newArgs.flatMap { _.result.vars }
                  )
                )
              }
          }

        case q"$subject.this" =>
          // println(s"  MATCH THIS ${showCode(expr)}")
          RichBlock(Seq.empty, RichTree(expr))

        case q"$subject.$method[..$tpts]" =>
          // println(s"   EXPANDING  ${showCode(expr)}")
          method match {
            case TermName("apply") =>
              val richSubject = visitApply(subject)
              RichBlock(
                richSubject.setup,
                RichTree(
                  q"${richSubject.result.tree}.$method[..$tpts]",
                  richSubject.result.vars
                )
              )
            case _ =>
              val richSubject = visitExpr(subject)
              RichBlock(
                richSubject.setup,
                RichTree(
                  q"${richSubject.result.tree}.$method[..$tpts]",
                  richSubject.result.vars
                )
              )
          }

          /*
        case q"$subj[..$tpts]" if tpts.nonEmpty =>
          visitExpr(subj) { richSubject =>
            fn(richSubject.map { t => q"$t[..$tpts]"})
          }
          */

        case EmptyTree =>
          RichBlock(Seq.empty, RichTree(EmptyTree))

        case q"$v: $tpt" =>
          val richV = visitExpr(v)
          // println(s"   TYPED ${v} : ${showRaw(tpt)}")
          val newtree = tpt match {
            case tq"$base @$annot" => 
              val newtpt = tq"${richV.result.tree} @$annot"
              q"${richV.result.tree}: $newtpt"
            case _ => 
              q"${richV.result.tree}: $tpt"
          }
          RichBlock(
            richV.setup,
            RichTree(newtree, richV.result.vars)
          )

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

    def visitStmt(expr: Tree): Seq[Tree] = {
      expr match {
        // observe needs to produce (resultVar, dependentVars..)
        case q"scappla.`package`.observe[$tDist]($oDist, $o)" =>
          (for {
            richDistName <- visitExpr(oDist)
            richO <- visitExpr(o)
          } yield {
            // Don't know what to do in this case - does not appear to be an actual observation
            /*
            if (richO.vars.nonEmpty) {
              throw new IllegalArgumentException("Observation cannot be dependent on random variables")
            }
            */

            val obName = TermName(c.freshName())
            val interpreter = scope.reference(TermName("interpreter"))
            val resultTree = q"""
              val $obName : scappla.Observation =
                scappla.observeImpl[$tDist](${interpreter.tree}, ${richDistName.tree}, ${richO.tree})
            """

            // add the observation score to the model score of the current scope
            builder.observation(obName)

            // println(s"  VARIABLES in oDist: ${deps} => ${vars(deps.head) }")
  //              println(s" OBSERVE ${showRaw(resultTree)}")

            RichBlock(
              resultTree +:
              richDistName.vars.filter(scope.isDeclared).toSeq.map { v =>
                q"$v.node.addObservation($obName.score)"
              },
              RichTree(EmptyTree)
            )
          }).setup

        // allow function name to be used when expanding rhs
        // full definition (with expanded tree) follows later
        // NOTE: no type parameterization or multiple argument lists
        case q"$mods def $tname(..$bargs): $rtpt = $body" =>
          val TermName(name) = tname
          val newName = TypeName(c.freshName(name))
          scope.declare(TermName(name), RichTree(q"${TermName(name)}", Set.empty, Some(newName)))
          visitMethod(TermName(name), bargs, body)

        // rewrite non-function assignments
        case q"$mods val $tname : $tpt = $rhs" =>
          (for {
            exprName <- visitExpr(rhs)
          } yield {
            val TermName(name) = tname
            val fullExpr = if (rhs.tpe <:< typeOf[scappla.Value[_, _]]) {
              builder.buffer(TermName(name))
              exprName.map { t => q"$t.buffer"}
            } else {
              exprName
            }
            scope.declare(TermName(name), fullExpr.copy(tree = q"${TermName(name)}"))
            RichBlock(
              Seq(q"val ${TermName(name)} = ${fullExpr.tree}"),
              RichTree(EmptyTree)
            )
          }).setup

        case _ if expr.tpe =:= definitions.UnitTpe =>
          val rt = visitExpr(expr)
          rt.setup :+ rt.result.tree
      }

    }

    def visitMethod(varName: TermName, bargs: Seq[Tree], body: Tree): Seq[Tree] = {
      val newArgs = parseArgs(bargs)

      val argScope = scope.push()
      newArgs.foreach { arg =>
        argScope.declare(arg.origName, RichTree(q"${arg.origName}", Set(arg.origName)))
      }
      val newScope = argScope.push()

      val visitor = new BlockVisitor(newScope)

      val q"{..$stmts}" = body
      val block = visitor.visitBlockStmts(stmts)
      val newBody = q"{..${block.setup :+ block.result.tree}}"

      val (wrapperName, wrapperDef) = newFnWrapper(varName, newArgs, body.tpe)
      // println(s"WRAPPER: ${showCode(wrapper)}")
      scope.declare(
        varName,
        RichTree(
          q"$varName",
          block.result.vars.filter(v => !argScope.isDeclared(v)),
          Some(wrapperName),
          newArgs.zipWithIndex.filter {
            case (arg, index) => block.result.vars.contains(arg.origName)
          }.map { _._2 }.toSet
        )
      )

      Seq(
        q"def $varName(..${newArgs.map(_.oldArgDecl)}): scappla.Variable[${body.tpe}] = $newBody"
      )
    }

    def visitFunction(varName: TermName, bargs: Seq[Tree], body: Tree): RichBlock = {
      val newArgs = parseArgs(bargs)

      val argScope = scope.push()
      newArgs.foreach { arg =>
        argScope.declare(arg.origName, RichTree(q"${arg.origName}", Set(arg.origName)))
      }
      val newScope = argScope.push()

      val q"{..$stmts}" = body
      val visitor = new BlockVisitor(newScope)
      val block = visitor.visitBlockStmts(stmts)
      val newBody = q"{..${block.setup :+ block.result.tree}}"

      val argVarTpes = newArgs.map { _.tpe }
      val fnTpe = treesToFunctionDef(argVarTpes :+ tq"scappla.Variable[${body.tpe}]")
      // println(s"FN TPE: ${fnTpe}")

      val (wrapperName, wrapperDef) = newFnWrapper(varName, newArgs, body.tpe)
      // println(s"WRAPPER: ${showCode(wrapper)}")
      scope.declare(
        varName,
        RichTree(
          q"$varName",
          block.result.vars.filter(v => !argScope.isDeclared(v)),
          Some(wrapperName),
          newArgs.zipWithIndex.filter {
            case (arg, index) => block.result.vars.contains(arg.origName)
          }.map { _._2 }.toSet
        )
      )

      val newDefTree = q"val $varName : $fnTpe = (..${newArgs.map(_.oldArgDecl)}) => $newBody"
      // println(s"NEW DEF: ${showCode(newDefTree)}")
      RichBlock(
        Seq(
          newDefTree,
          wrapperDef,
        ),
        scope.reference(varName)
      )
    }

    def newFnWrapper(varName: TermName, newArgs: Seq[RichArg], bodyTpe: Type): (TypeName, Tree) = {
      val tpes: Seq[Tree] = newArgs.map { _.tpe }
      val argTpes = tpes :+ tq"${bodyTpe}"
      val fnWrapperTpe = treesToFunctionDef(argTpes)
      val TermName(tName) = varName
      val newName = TypeName(c.freshName(tName))
      (
        newName,
        q"""class ${newName}(accumulator: scappla.Accumulator) extends $fnWrapperTpe {

            def apply(..${newArgs.map { arg =>
                q"${arg.mods} val ${arg.origName}: ${arg.tpe}"
            }}): ${bodyTpe} = {${
              q"""val result = $varName(..${
                newArgs.map { _.origName }
              })
              accumulator.add(result.node)
              result.get
              """
            }}

            }"""
      )
    }

    def fnWrapper(varName: TermName, newArgs: Seq[RichArg], bodyTpe: Type): (TypeName, Tree) = {
      val tpes: Seq[Tree] = newArgs.map { _.tpe }
      val argTpes = tpes :+ tq"${bodyTpe}"
      val fnWrapperTpe = treesToFunctionDef(argTpes)
      val TermName(tName) = varName
      val newName = TypeName(c.freshName(tName))
      (
        newName,
        q"""class ${newName}(accumulator: scappla.Accumulator) extends $fnWrapperTpe {

            def apply(..${newArgs.map { arg =>
                q"${arg.mods} val ${arg.origName}: ${arg.tpe}"
            }}): ${bodyTpe} = {${
              q"""val result = $varName(..${
                newArgs.map { arg =>
                  q"${arg.origName}"
                }
              })
              accumulator.add(result.node)
              result.get
              """
            }}

            }"""
      )
    }


    def parseArgs(bargs: Seq[Tree]): Seq[RichArg] = {
      bargs.map {
        case q"$mods val $arg: $tpt = $argExpr" =>
          // println(s"    FN VALDEF ${showRaw(arg)}")
          val TermName(tname) = arg
          val newArgName = TermName(c.freshName(tname))
          val inlExpr = toExpr(visitExpr(argExpr), scope)
          RichArg(
            mods,
            tpt,
            TermName(tname),
            q"$mods val $arg: $tpt"
          )
      }
    }

    def treesToFunctionDef(argTpes: Seq[Tree]): Tree = {
      argTpes.size match {
        case 1 => tq"Function0[..$argTpes]"
        case 2 => tq"Function1[..$argTpes]"
        case 3 => tq"Function2[..$argTpes]"
        case 4 => tq"Function3[..$argTpes]"
        case 5 => tq"Function4[..$argTpes]"
        case 6 => tq"Function5[..$argTpes]"
        case 7 => tq"Function6[..$argTpes]"
        case 8 => tq"Function7[..$argTpes]"
        case 9 => tq"Function8[..$argTpes]"
        case 10 => tq"Function9[..$argTpes]"
        case 11 => tq"Function10[..$argTpes]"
        case 12 => tq"Function11[..$argTpes]"
        case 13 => tq"Function12[..$argTpes]"
        case 14 => tq"Function13[..$argTpes]"
        case 15 => tq"Function14[..$argTpes]"
        case 16 => tq"Function15[..$argTpes]"
        case 17 => tq"Function16[..$argTpes]"
        case 18 => tq"Function17[..$argTpes]"
        case 19 => tq"Function18[..$argTpes]"
        case 20 => tq"Function19[..$argTpes]"
        case 21 => tq"Function20[..$argTpes]"
        case 22 => tq"Function21[..$argTpes]"
        case 23 => tq"Function22[..$argTpes]"
      }
    }

    def visitSubExpr(tExpr: Tree): RichTree = {
      // println(s"VISITING SUB EXPR ${showCode(tExpr)} (${tExpr.tpe})")
      val newScope = scope.push()
      val subVisitor = new BlockVisitor(newScope)
      val newSubStmts = if (tExpr.tpe =:= definitions.UnitTpe) {
        // println(s"   UNIT TPE EXPR (${tExpr.tpe})")
        val stmts = subVisitor.visitStmt(tExpr)
        // println(s" (UNIT TPE) BUILDER: ${subVisitor.builder}")
        RichBlock(
          stmts,
          subVisitor.builder.build(newScope, RichTree(EmptyTree), definitions.UnitTpe)
        )
      } else {
        // println(s"   NON UNIT TPE EXPR (${tExpr.tpe})")
        val rtLast = subVisitor.visitExpr(tExpr)
        // println(s"LAST EXPR IN SUB EXPR: ${showCode(rtLast.result.tree)}")
        // println(s" (NUNIT TPE) BUILDER: ${subVisitor.builder}")
        RichBlock(
          rtLast.setup,
          subVisitor.builder.build(newScope, rtLast.result, tExpr.tpe)
        )
      }
      val result = toExpr(newSubStmts, scope)
      // println(s"RESULT FROM SUB EXPR: ${showCode(result.tree)} (${result.vars})")
      result
    }

    def reIdentPat(expr: Tree): Tree = {
//      println(s"   REMAPPING ${showRaw(expr)}")
      expr match {
        case pq"$tName @ $pat" =>
//          println(s"     TERM ${showRaw(tName)}")
          val TermName(name) = tName
          pq"${TermName(name)} @ $pat"

        case pq"$ref(..$pats)" =>
          pq"$ref(..${pats.map { pat => reIdentPat(pat) }})"

        case _ =>
          expr
      }
    }

    def toExpr(block: RichBlock, scope: Scope): RichTree = {
      val exprs = block.setup
      if (exprs.size == 0) {
        val result = block.result
        RichTree(
          result.tree,
          result.vars.filter(scope.isDefined)
        )
      } else {
//        println(s"TO EXPR VARS ${exprs.flatMap(_.vars).map{ _.toString}.mkString(",")}")
        RichTree(
          q"{ ..${exprs :+ block.result.tree} }",
          block.result.vars.filter(scope.isDefined)
        )
      }
    }

  }

  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[scappla.Model[X]] = {
//    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]

    val q"{..$stmts}" = fn.tree
    val visitor = new BlockVisitor(new Scope(Map(TermName("interpreter") -> RichTree(q"interpreter"))))
    val newStmts = visitor.visitBlockStmts(stmts)
    val newBody =
      q"""new scappla.Model[$xType] {

        def sample(interpreter: scappla.Interpreter) = {
          val scappla.Variable(value, node) = {
            ..${newStmts.setup :+ newStmts.result.tree}
          }
          node.complete()
          value
        }
      }"""
//    println("INFERRING")
//    println(showCode(newBody))
    c.Expr[scappla.Model[X]](newBody)
  }

}