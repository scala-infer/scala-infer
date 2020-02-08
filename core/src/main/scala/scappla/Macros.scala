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
      wrapper: Option[TypeName] = None
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
  )

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
    newName: TermName,
    newArgDecl: Tree
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

  class Scope(known: Map[TermName, RichTree]) {

    private val args: mutable.Set[TermName] = mutable.Set.empty
    private val refs: mutable.HashMap[TermName, RichTree] = mutable.HashMap.empty
    private val referenced: mutable.Set[TermName] = mutable.Set.empty

    def isPreDefined(v: TermName): Boolean = {
      known.contains(v) || args.contains(v)
    }

    def isDefined(v: TermName): Boolean = {
      known.contains(v) || refs.contains(v)
    }

    def reference(v: TermName): RichTree = {
      referenced += v
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
      completeable += q"$o"
      this
    }

    def buffer(o: TermName): VariableAggregator = {
      completeable += q"$o"
      this
    }

    def build(scope: Scope, result: RichTree, tpe: Type): RichTree = {
      val tree = if (obs.isEmpty
          && vars.isEmpty
          && completeable.isEmpty
          && result.vars.forall(scope.isPreDefined)
      ) {
        q"""scappla.Variable[${tpe.widen}](${result.tree}, scappla.ConstantNode)"""
      } else {
        q"""scappla.Variable[${tpe.widen}](${result.tree}, new scappla.BayesNode {

          val modelScore: scappla.Score = {${
          (obs.map { t =>
            q"$t.score": Tree
          } ++ vars.map { t =>
            q"$t.node.modelScore": Tree
          }).reduceOption { (a, b) => q"$a.+($b)" }
              .getOrElse(q"scappla.Value.apply(0.0)")
          }}

          val guideScore: scappla.Score = {${
          vars.map { t =>
            q"$t.node.guideScore": Tree
          }.reduceOption { (a, b) => q"$a.+($b)" }
              .getOrElse(q"scappla.Value.apply(0.0)")
          }}

          def addObservation(score: scappla.Score) = {..${
          result.vars.filter(!scope.isPreDefined(_)).toSeq.map { lv =>
            q"$lv.node.addObservation(score)"
          }
          }}

          def addVariable(modelScore: scappla.Score, guideScore: scappla.Score) = {..${
          result.vars.filter(!scope.isPreDefined(_)).toSeq.map { lv =>
            q"$lv.node.addVariable(modelScore, guideScore)"
          }
          }}

          def complete() = {..${
          completeable.reverse.map { c =>
            q"$c.complete()"
          }
          }}
        })"""
      }
      RichTree(tree, Set.empty)
    }
  }

  class BlockVisitor(scope: Scope) {

    private val builder = new VariableAggregator()

    def visitBlockStmts(stmts: Seq[Tree]): RichBlock = {
//      println(s"VISITINGN BLOCK ${showCode(q"{..$stmts}")}")
      val setup :+ last = stmts
      val richSetup = setup.flatMap(t => visitStmt(t))
//      println(s"   TPE: ${last.tpe}")
      (if (last.tpe =:= definitions.UnitTpe) {
        val lastVar = builder.build(scope, RichTree(EmptyTree), definitions.UnitTpe)
        RichBlock(
          richSetup ++ visitStmt(last),
          lastVar
        )
      } else {
        val lastBlock = visitExpr(last) 
        val lastExpr = builder.build(scope, lastBlock.result, last.tpe)
        RichBlock(
          richSetup ++ lastBlock.setup,
          lastExpr
        )
      })
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
            val varFn = c.freshName()
            val typeArgs = expr.tpe.widen.typeArgs
//            println(s"n args: ${typeArgs.size}")
            val richFn =
              q"""val $varFn : scappla.Variable[${TypeTree(typeArgs(0))}] => scappla.Variable[${TypeTree(typeArgs(1))}] =
                      in => scappla.Variable(${TermName(name)}(in.get), in.node)"""
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
          val vars = newStmts.result.vars.toSet.filter(scope.isDefined)

          val varName = TermName(c.freshName())
          builder.variable(varName)
          scope.declare(varName, RichTree(EmptyTree))

          val varDef = q"{ ..${newStmts.setup :+ newStmts.result.tree}}"
          if (expr.tpe =:= definitions.UnitTpe) {
            RichBlock(
              Seq(varDef),
              RichTree(EmptyTree)
            )
          } else {
            val varExpr = q"val $varName = $varDef"
            RichBlock(
              Seq(varExpr),
              RichTree(q"$varName.get", vars)
            )
          }

        case q"if ($cond) $tExpr else $fExpr" =>
          val condRef = visitExpr(cond) 
//            println(s"  MATCHING IF ELSE $condRef")

          val ifVar = TermName(c.freshName())
          builder.variable(ifVar)
          scope.declare(ifVar, RichTree(EmptyTree))

          val richTrue = visitSubExpr(tExpr)
          // println(s"  IF TRUE $condRef: ${richTrue.vars}")
          val richFalse = visitSubExpr(fExpr)
          // println(s"  IF FALSE $condRef: ${richTrue.vars}")
          val resultTree = q"$ifVar.get"
          val setup: Seq[Tree] = (
            condRef.setup ++ Seq(
              q"""val $ifVar = if (${condRef.result.tree})
                  ${richTrue.tree}
                else
                  ${richFalse.tree}"""
            ) ++ condRef.result.vars.toSeq.map { cv =>
              q"$cv.node.addVariable($ifVar.node.modelScore, $ifVar.node.guideScore)"
            })
          RichBlock(
            setup,
            RichTree(
              q"$ifVar.get",
              condRef.result.vars ++ richTrue.vars ++ richFalse.vars
            )
          )

        case q"$expr match { case ..$cases }" =>
          val mappedCases = cases.map { c =>
            val cq"$when => $result" = c
            val richResult = visitSubExpr(result)
            (richResult, cq"${reIdentPat(when)} => ${richResult.tree}")
          }

          val matchName = visitExpr(expr)
          val matchVar = TermName(c.freshName())
          val caseVars = mappedCases.map { _._1.vars }.flatten
          builder.variable(matchVar)
          scope.declare(matchVar, RichTree(EmptyTree))

          // val fnResult = fn(RichTree(q"$matchVar.get"))
          RichBlock(
            matchName.setup :+ 
              q"""val $matchVar = ${matchName.result.tree} match {
                  case ..${mappedCases.map{ _._2 }}
                }"""
            ,
            RichTree(q"$matchVar.get", matchName.result.vars ++ caseVars)
          )

        case q"scappla.`package`.sample[$tDist]($prior, $guide)" =>
          val priorName = visitExpr(prior)
          val guideName = visitExpr(guide)

          val tVarName = TermName(c.freshName())
          // println(s"  SAMPLE MATCHING ${showRaw(tDist.tpe)} (${showCode(tDist)})")
          val interpreter = scope.reference(TermName("interpreter"))
          builder.variable(tVarName)
          scope.declare(tVarName, RichTree(EmptyTree))

          RichBlock(
            guideName.setup ++ priorName.setup :+
              q"val $tVarName = ${guideName.result.tree}.sample(${interpreter.tree}, ${priorName.result.tree})",
            RichTree(q"$tVarName.get", Set(tVarName))
          )

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
          f match {
            case Ident(TermName(fname))
                if scope.isDefined(fname) && scope.reference(fname).isFn =>
              // println(s"APPLYING FN ${fname}")
              val richFn = visitExpr(f)
              val richO = visitExpr(o)
              val varResult = TermName(c.freshName())
              builder.variable(varResult)
              scope.declare(varResult, RichTree(EmptyTree))

              val nodes = richO.result.vars.map { t => q"$t.node" }
              RichBlock(
                richFn.setup ++ richO.setup :+
                  q"val $varResult = ${richFn.result.tree}.apply(scappla.Variable(${richO.result.tree}, new scappla.Dependencies(Seq(..$nodes))))",
                RichTree(q"$varResult.get", richFn.result.vars ++ richO.result.vars + varResult)
              )

            case _ =>
              val richFn = visitExpr(f)
              val richO = visitExpr(o)
              RichBlock(
                richFn.setup ++ richO.setup,
                RichTree(
                  q"${richFn.result.tree}.apply(${richO.result.tree})",
                  richFn.result.vars ++ richO.result.vars
                )
              )
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
          // println(s"  FN MATCH (TYPE: ${expr.tpe}) ${showCode(expr)}")
          val invoVar = TermName(c.freshName())
          var needInvoVar = false
          val mRes = (mArgs: List[List[Tree]]).map { args =>
            args.map { arg =>
              val RichBlock(setup, aName) = visitExpr(arg)
              if (!aName.isFn) {
                RichBlock(setup, aName)
              } else {
                val fnName = TermName(c.freshName())
                needInvoVar = true
                RichBlock(
                  setup :+ q"val $fnName = new ${aName.wrapper.get}($invoVar)",
                  RichTree(q"$fnName", Set.empty)
                )
              }
            }
          }

          val invoDefs = if (needInvoVar) {
            val invoDef = q"val $invoVar = new scappla.Invocations()"
            builder.function(invoVar)
            Seq(invoDef)
          } else Seq.empty
          val defs = invoDefs ++ mRes.flatMap { res =>
            res.flatMap { _.setup }
          }
          val newArgs = mRes.map { res =>
            res.map { _.result.tree }
          }

          val richFn = visitExpr(f)
          RichBlock(
            richFn.setup,
            RichTree(
              q"${richFn.result.tree}.$m[..$tpts](...$newArgs)",
              richFn.result.vars ++ mRes.flatten.flatMap { _.result.vars }
            )
          )

        case q"$f(..$args)" =>
          f match {
            case Ident(TermName(fname)) if scope.isDefined(fname) && scope.reference(fname).isFn =>
              // println(s"APPLYING KNOWN METHOD ${fname}")
              val newArgs = args.map { o =>
                val argBlock = visitExpr(o)
                val nodes = argBlock.result.vars.map { t => q"$t.node" }
                (
                  argBlock.setup,
                  q"scappla.Variable(${argBlock.result.tree}, new scappla.Dependencies(Seq(..$nodes)))",
                  argBlock.result.vars
                )
              }

              val richFn = visitExpr(f)
              val varResult = TermName(c.freshName())
              builder.variable(varResult)
              scope.declare(varResult, RichTree(EmptyTree))

              val argVars = newArgs.map { _._3 }.flatten.toSet
              RichBlock(
                richFn.setup ++ newArgs.map { _._1 }.flatten ++ Seq(
                  q"val $varResult = ${richFn.result.tree}(..${newArgs.map { _._2 }})"
                ),
                RichTree(
                  q"$varResult.get",
                  richFn.result.vars ++ newArgs.flatMap { _._3 }
                )
              )

            case _ =>
              // println(s"APPLYING UNNOWN FUNCTION ${showRaw(f)}")
              val newArgs = args.map { o =>
                val argBlock = visitExpr(o)
                (
                  argBlock.setup,
                  argBlock.result.tree,
                  argBlock.result.vars
                )
              }

              val richFn = visitExpr(f)
              RichBlock(
                richFn.setup ++ newArgs.map { _._1 }.flatten,
                RichTree(
                  q"${richFn.result.tree}(..${newArgs.map { _._2 }})",
                  richFn.result.vars ++ newArgs.flatMap { _._3 }
                )
              )
          }

        case q"$subject.this" =>
          // println(s"  MATCH THIS ${showCode(expr)}")
          RichBlock(Seq.empty, RichTree(expr))

        case q"$subject.$method" =>
          method match {
            case TermName("apply") =>
              val richSubject = visitApply(subject)
              RichBlock(
                richSubject.setup,
                RichTree(
                  q"${richSubject.result.tree}.$method",
                  richSubject.result.vars
                )
              )
            case _ =>
              val richSubject = visitExpr(subject)
              RichBlock(
                richSubject.setup,
                RichTree(
                  q"${richSubject.result.tree}.$method",
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
          RichBlock(
            richV.setup,
            RichTree(q"${richV.result.tree}: $tpt", richV.result.vars)
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
          val richDistName = visitExpr(oDist)
          val richO = visitExpr(o)
          val obName = TermName(c.freshName())
          val interpreter = scope.reference(TermName("interpreter"))
          val resultTree = q"""
            val $obName : scappla.Observation =
              scappla.observeImpl[$tDist](${interpreter.tree}, ${richDistName.result.tree}, ${richO.result.tree})
          """

          builder.observation(obName)

          // println(s"  VARIABLES in oDist: ${deps} => ${vars(deps.head) }")
//              println(s" OBSERVE ${showRaw(resultTree)}")

          (richDistName.setup ++ richO.setup :+ resultTree) ++
            richDistName.result.vars.toSeq.map { v =>
              q"$v.node.addObservation($obName.score)"
            }

        // allow function name to be used when expanding rhs
        // full definition (with expanded tree) follows later
        // NOTE: no type parameterization or multiple argument lists
        case q"$mods def $tname(..$bargs): $rtpt = $body" =>
          val TermName(name) = tname
          val newName = TypeName(c.freshName(name))
          scope.declare(TermName(name), RichTree(q"${TermName(name)}", Set.empty, Some(newName)))
          Seq(visitMethod(TermName(name), bargs, body))

        // rewrite non-function assignments
        case q"$mods val $tname : $tpt = $rhs" =>
          val TermName(name) = tname
          val exprName = visitExpr(rhs)
          val fullExpr = if (rhs.tpe <:< typeOf[scappla.Value[_, _]]) {
            builder.buffer(TermName(name))
            exprName.result.map { t => q"$t.buffer"}
          } else {
            exprName.result
          }
          scope.declare(TermName(name), fullExpr.copy(tree = q"${TermName(name)}"))
          exprName.setup :+ q"val ${TermName(name)} = ${fullExpr.tree}"

        case _ if expr.tpe =:= definitions.UnitTpe =>
          val rt = visitExpr(expr)
          rt.setup
      }

    }

    def visitMethod(varName: TermName, bargs: Seq[Tree], body: Tree): Tree = {
      val newArgs = parseArgs(bargs)

      val newScope = scope.push()

      val visitor = new BlockVisitor(newScope)
      val argDecls = newArgs.map { arg =>
        q"val ${arg.origName} = ${arg.newName}.get"
      }
      newArgs.foreach { arg =>
        newScope.argument(arg.newName, RichTree(q"${arg.newName}", Set(arg.newName)))
        newScope.declare(arg.origName, RichTree(q"${arg.origName}", Set(arg.newName)))
      }

      val q"{..$stmts}" = body
      val block = visitor.visitBlockStmts(stmts)
      val newStmts = argDecls ++ block.setup
      val newBody = q"{..${newStmts :+ block.result.tree}}"

      q"def $varName(..${newArgs.map(_.newArgDecl)}): scappla.Variable[${body.tpe}] = $newBody"
    }

    def visitFunction(varName: TermName, bargs: Seq[Tree], body: Tree): RichBlock = {
      val newArgs = parseArgs(bargs)

      val newScope = scope.push()

      val visitor = new BlockVisitor(newScope)
      val argDecls = newArgs.map { arg =>
        if (arg.tpe.tpe <:< typeOf[Value[_, _]]) {
          // println(s"  FOUND VALUE ARG ${arg.origName.decoded}")
          visitor.builder.buffer(arg.origName)
          q"val ${arg.origName} = ${arg.newName}.get.buffer"
        } else {
          q"val ${arg.origName} = ${arg.newName}.get"
        }
      }
      newArgs.foreach { arg =>
        newScope.argument(arg.newName, RichTree(q"${arg.newName}", Set(arg.newName)))
        newScope.declare(arg.origName, RichTree(q"${arg.origName}", Set(arg.newName)))
      }

      val q"{..$stmts}" = body
      val block = visitor.visitBlockStmts(stmts)
      val newStmts = argDecls ++ block.setup :+ block.result.tree
      val newBody = q"{..${newStmts}}"

      // TODO: do something with the free variables of the function (newScope.referenced)
      // The log-probability, resulting from invocation, should be added to those variables
      // Is this something to do when the function goes out of scope in the completion phase?

      // val newVars = newStmts.flatMap { _.vars }.toSet.filter(scope.isDefined)
      // println(s"NEW VARS: ${newVars}")

      val argVarTpes = newArgs.map {arg => tq"scappla.Variable[${arg.tpe}]"}
      val fnTpe = treesToFunctionDef(argVarTpes :+ tq"scappla.Variable[${body.tpe}]")
      // println(s"FN TPE: ${fnTpe}")

      val (wrapperName, wrapperDef) = fnWrapper(varName, newArgs, body.tpe)
      // println(s"WRAPPER: ${showCode(wrapper)}")
      scope.declare(varName, RichTree(q"$varName", block.result.vars, Some(wrapperName)))

      val newDefTree = q"val $varName : $fnTpe = (..${newArgs.map(_.newArgDecl)}) => $newBody"
      // println(s"NEW DEF: ${showCode(newDefTree)}")
      RichBlock(
        Seq(
          newDefTree,
          wrapperDef,
        ),
        scope.reference(varName)
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
        q"""class ${newName}(invocations: scappla.Invocations) extends $fnWrapperTpe {

            def apply(..${newArgs.map { arg =>
                q"${arg.mods} val ${arg.origName}: ${arg.tpe}"
            }}): ${bodyTpe} = {${
              q"""val result = $varName(..${
                newArgs.map { arg =>
                  q"scappla.Variable(${arg.origName}, scappla.ConstantNode)"
                }
              })
              invocations.add(result.node)
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
            newArgName,
            q"$mods val $newArgName: ${tq"scappla.Variable[$tpt]"} = ${inlExpr.tree}"
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
        val result = subVisitor.builder.build(newScope, RichTree(EmptyTree), definitions.UnitTpe)
        RichBlock(
          subVisitor.visitStmt(tExpr),
          result
        )
      } else {
        // println(s"   NON UNIT TPE EXPR (${tExpr.tpe})")
        val rtLast = subVisitor.visitExpr(tExpr)
        // println(s"LAST EXPR IN SUB EXPR: ${showCode(rtLast.tree)}")
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