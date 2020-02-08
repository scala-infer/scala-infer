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
    stmts: Seq[Tree],
    vars: Set[TermName]
  )

  object RichBlock {
    def apply(stmts: Seq[RichTree]): RichBlock = {
      RichBlock(
        stmts.map { _.tree },
        stmts.flatMap { _.vars }.toSet
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
          richSetup ++
          visitStmt(last) :+
          lastVar.tree,
          lastVar.vars
        )
      } else {
        RichBlock(
          richSetup ++
          visitExpr(last) { lastRt =>
            RichBlock(Seq(builder.build(scope, lastRt, last.tpe)))
          }.stmts,
          Set.empty
        )
      })
    }

    def visitApply(expr: Tree)(fn: RichTree => RichBlock): RichBlock = {
      expr match {
        case Ident(TermName(name)) if expr.tpe <:< typeOf[Function1[_, _]] =>
//          println(s"   REFERNECE FN ${showRaw(expr, printTypes = true)}")
          val tname = TermName(name)
          if (scope.isDefined(TermName(name))) {
            fn(scope.reference(tname))
          } else {
            val varFn = c.freshName()
            val typeArgs = expr.tpe.widen.typeArgs
//            println(s"n args: ${typeArgs.size}")
            val richFn =
              q"""val $varFn : scappla.Variable[${TypeTree(typeArgs(0))}] => scappla.Variable[${TypeTree(typeArgs(1))}] =
                      in => scappla.Variable(${TermName(name)}(in.get), in.node)"""
            scope.declare(varFn, RichTree(q"$varFn", Set.empty))
            RichBlock(
              Seq(richFn) ++ fn(scope.reference(varFn)).stmts,
              Set.empty
            )
          }
      }
    }

    def visitExpr(expr: Tree, resultVarName: Option[String] = None)(fn: RichTree => RichBlock): RichBlock = {
      expr match {
        case Ident(TermName(name)) =>
//          println(s"   REFERNECE ${showRaw(expr, printTypes = true)}")
          val tname = TermName(name)
          fn(scope.reference(tname))

        case Literal(value) =>
//          println(s"   LITERAL ${showRaw(expr)}")
          fn(RichTree(expr))

        case q"{ ..$stmts }" if stmts.size > 1 =>
          // println(s"   BLOCK ${showCode(expr)}")
          val visitor = new BlockVisitor(scope.push())
          val newStmts = visitor.visitBlockStmts(stmts)

          // reduce list of variables to those known in the current scope
          val vars = newStmts.vars.toSet.filter(scope.isDefined)

          val varName = TermName(c.freshName())
          builder.variable(varName)
          scope.declare(varName, RichTree(EmptyTree))

          val varDef = q"{ ..${newStmts.stmts}}"
          val varExpr = q"val $varName = $varDef"
          val fnResult = if (expr.tpe =:= definitions.UnitTpe) {
            fn(RichTree(EmptyTree, vars))
          } else {
            fn(RichTree(q"$varName.get", vars))
          }
          RichBlock(
            Seq(varExpr) ++ fnResult.stmts,
            fnResult.vars
          )

        case q"if ($cond) $tExpr else $fExpr" =>
          visitExpr(cond) { condRef =>
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
              q"""val $ifVar = if (${condRef.tree})
                    ${richTrue.tree}
                  else
                    ${richFalse.tree}"""
              +: condRef.vars.toSeq.map { cv =>
                q"$cv.node.addVariable($ifVar.node.modelScore, $ifVar.node.guideScore)"
              })
            val fnResult = fn(RichTree(
              resultTree,
              condRef.vars ++ richTrue.vars ++ richFalse.vars
            ))
            RichBlock(
              setup ++ fnResult.stmts,
              fnResult.vars
            )
          }

        case q"$expr match { case ..$cases }" =>
          val mappedCases = cases.map { c =>
            val cq"$when => $result" = c
            val richResult = visitSubExpr(result)
            (richResult, cq"${reIdentPat(when)} => ${richResult.tree}")
          }

          visitExpr(expr) { matchName =>
            val matchVar = TermName(c.freshName())
            val caseVars = mappedCases.map { _._1.vars }.flatten
            builder.variable(matchVar)
            scope.declare(matchVar, RichTree(EmptyTree))

            val fnResult = fn(RichTree(q"$matchVar.get"))
            RichBlock(
              Seq(
                q"""val $matchVar = ${matchName.tree} match {
                    case ..${mappedCases.map{ _._2 }}
                  }""")
              ++ fnResult.stmts,
              matchName.vars ++ caseVars ++ fnResult.vars
            )
          }

        case q"scappla.`package`.sample[$tDist]($prior, $guide)" =>
          visitExpr(prior) { priorName =>
            visitExpr(guide) { guideName =>
              val tVarName = TermName(c.freshName())
              // println(s"  SAMPLE MATCHING ${showRaw(tDist.tpe)} (${showCode(tDist)})")
              builder.variable(tVarName)
              val interpreter = scope.reference(TermName("interpreter"))
              val ref = RichTree(q"$tVarName", Set(tVarName))
              scope.declare(tVarName, ref)

              val fnResult = fn(ref.map { tree => q"$tree.get" })
              RichBlock(
                q"val $tVarName = ${guideName.tree}.sample(${interpreter.tree}, ${priorName.tree})"
                +: fnResult.stmts,
                fnResult.vars
              )
            }
          }

        case q"(..$bargs) => $body" =>
          val varName = TermName(c.freshName())
          visitFunction(varName, bargs, body)(fn)

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
              visitExpr(f) { richFn =>
                visitExpr(o) { richO =>
                  val varResult = TermName(c.freshName())
                  builder.variable(varResult)
                  scope.declare(varResult, RichTree(EmptyTree))

                  val nodes = richO.vars.map { t => q"$t.node" }
                  val fnResult = fn(RichTree(
                    q"$varResult.get",
                    richFn.vars ++ richO.vars + varResult
                  ))
                  RichBlock(
                    q"val $varResult = ${richFn.tree}.apply(scappla.Variable(${richO.tree}, new scappla.Dependencies(Seq(..$nodes))))"
                    +: fnResult.stmts,
                    fnResult.vars
                  )
                }
              }
            case _ =>
              visitExpr(f) { richFn =>
                visitExpr(o) { richO =>
                  fn(RichTree(
                    q"${richFn.tree}.apply(${richO.tree})",
                    richFn.vars ++ richO.vars
                  ))
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
          // println(s"  FN MATCH (TYPE: ${expr.tpe}) ${showCode(expr)}")
          val invoVar = TermName(c.freshName())
          var needInvoVar = false
          val mRes = (mArgs: List[List[Tree]]).map { args =>
            args.map { arg =>
              visitExpr(arg) { aName =>
                if (!aName.isFn) {
                  RichBlock(Seq(aName.tree), aName.vars)
                } else {
                  val fnName = TermName(c.freshName())
                  needInvoVar = true
                  RichBlock(
                    Seq(
                      q"val $fnName = new ${aName.wrapper.get}($invoVar)",
                      q"$fnName"
                    ),
                    Set.empty
                  )
                }
              }
            }
          }

          val invoDefs = if (needInvoVar) {
            val invoDef = q"val $invoVar = new scappla.Invocations()"
            builder.function(invoVar)
            Seq(invoDef)
          } else Seq.empty
          val defs = invoDefs ++ mRes.flatMap { res =>
            res.flatMap { _.stmts.dropRight(1) }
          }
          val newArgs = mRes.map { res =>
            res.map { _.stmts.last.tree }
          }

          visitExpr(f) { richFn =>
            val fnResult = fn(RichTree(
              q"${richFn.tree}.$m[..$tpts](...$newArgs)",
              richFn.vars ++ mRes.flatten.flatMap { _.vars }
            ))
            RichBlock(
              defs ++ fnResult.stmts,
              fnResult.vars
            )
          }

        case q"$f(..$args)" =>
          f match {
            case Ident(TermName(fname)) if scope.isDefined(fname) && scope.reference(fname).isFn =>
              // println(s"APPLYING KNOWN METHOD ${fname}")
              val newArgs = args.map { o =>
                val argBlock = visitExpr(o) { richO =>
                  RichBlock(Seq(richO))
                }
                val nodes = argBlock.vars.map { t => q"$t.node" }
                (
                  argBlock.stmts.dropRight(1),
                  q"scappla.Variable(${argBlock.stmts.last}, new scappla.Dependencies(Seq(..$nodes)))",
                  argBlock.vars
                )
              }

              visitExpr(f) { richFn =>
                  val varResult = TermName(c.freshName())
                  builder.variable(varResult)
                  val argVars = newArgs.map { _._3 }.flatten.toSet
                  scope.declare(varResult, RichTree(EmptyTree))
                  val fnResult = fn(RichTree(
                    q"$varResult.get",
                    richFn.vars ++ newArgs.flatMap { _._3 }
                  ))
                  RichBlock(
                    newArgs.map { _._1 }.flatten ++ Seq(
                      q"val $varResult = ${richFn.tree}(..${newArgs.map { _._2 }})"
                    ) ++ fnResult.stmts,
                    fnResult.vars
                  )
              }

            case _ =>
              // println(s"APPLYING UNNOWN FUNCTION ${showRaw(f)}")
              val newArgs = args.map { o =>
                val argBlock = visitExpr(o) { richO =>
                  RichBlock(Seq(richO))
                }
                (
                  argBlock.stmts.dropRight(1),
                  argBlock.stmts.last,
                  argBlock.vars
                )
              }

              visitExpr(f) { richFn =>
                val result = q"${richFn.tree}(..${newArgs.map { _._2 }})"
                val fnResult = fn(RichTree(
                  result,
                  richFn.vars ++ newArgs.flatMap { _._3 }
                ))
                RichBlock(
                  newArgs.map { _._1 }.flatten ++ fnResult.stmts,
                  fnResult.vars
                )
              }
          }

        case q"$subject.this" =>
          // println(s"  MATCH THIS ${showCode(expr)}")
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

    def visitStmt(expr: Tree): Seq[Tree] = {
      expr match {
        // observe needs to produce (resultVar, dependentVars..)
        case q"scappla.`package`.observe[$tDist]($oDist, $o)" =>
          visitExpr(oDist) { richDistName =>
            visitExpr(o) { richO =>
              val obName = TermName(c.freshName())
              val interpreter = scope.reference(TermName("interpreter"))
              val resultTree = q"val $obName : scappla.Observation = scappla.observeImpl[$tDist](${interpreter.tree}, ${richDistName.tree}, ${richO.tree})"

              builder.observation(obName)

              // println(s"  VARIABLES in oDist: ${deps} => ${vars(deps.head) }")
//              println(s" OBSERVE ${showRaw(resultTree)}")

              RichBlock(
                RichTree(resultTree, richDistName.vars) +:
                  richDistName.vars.toSeq.map { v =>
                    RichTree(q"$v.node.addObservation($obName.score)")
                  }
              )
            }
          }.stmts

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
          visitExpr(rhs) { exprName =>
            val fullExpr = if (rhs.tpe <:< typeOf[scappla.Value[_, _]]) {
              builder.buffer(TermName(name))
              exprName.map { t => q"$t.buffer"}
            } else {
              exprName
            }
            scope.declare(TermName(name), fullExpr.copy(tree = q"${TermName(name)}"))
            RichBlock(
              Seq(fullExpr.map { t => q"val ${TermName(name)} = $t" })
            )
          }.stmts

        case _ if expr.tpe =:= definitions.UnitTpe =>
          visitExpr(expr)(rt => RichBlock(Seq(rt))).stmts
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
      val newStmts = argDecls ++ visitor.visitBlockStmts(stmts).stmts
      val newBody = q"{..${newStmts}}"

      q"def $varName(..${newArgs.map(_.newArgDecl)}): scappla.Variable[${body.tpe}] = $newBody"
    }

    def visitFunction(varName: TermName, bargs: Seq[Tree], body: Tree)(fn: RichTree => RichBlock): RichBlock = {
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
      val newStmts = argDecls ++ block.stmts
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
      scope.declare(varName, RichTree(q"$varName", block.vars, Some(wrapperName)))

      val newDefTree = q"val $varName : $fnTpe = (..${newArgs.map(_.newArgDecl)}) => $newBody"
      // println(s"NEW DEF: ${showCode(newDefTree)}")
      val fnResult = fn(scope.reference(varName))
      RichBlock(
        Seq(
          newDefTree,
          wrapperDef
        ) ++ fnResult.stmts,
        fnResult.vars
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
          val inlExpr = toExpr(visitExpr(argExpr)(rt => RichBlock(Seq(rt))), scope)
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
          subVisitor.visitStmt(tExpr) :+ result.tree,
          result.vars
        )
      } else {
        // println(s"   NON UNIT TPE EXPR (${tExpr.tpe})")
        subVisitor.visitExpr(tExpr) { rtLast =>
          // println(s"LAST EXPR IN SUB EXPR: ${showCode(rtLast.tree)}")
          RichBlock(
            Seq(subVisitor.builder.build(newScope, rtLast, tExpr.tpe))
          )
        }
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
      val exprs = block.stmts
      if (exprs.size == 1) {
        val head = exprs.head
        RichTree(
          head.tree,
          block.vars.filter(scope.isDefined)
        )
      } else {
//        println(s"TO EXPR VARS ${exprs.flatMap(_.vars).map{ _.toString}.mkString(",")}")
        RichTree(
          q"{ ..${exprs.map(_.tree)} }",
          block.vars.filter(scope.isDefined)
        )
      }
    }

  }

  def infer[X: WeakTypeTag](fn: c.Expr[X]): c.Expr[scappla.Model[X]] = {
//    println(s"  INFER: ${showRaw(fn)}")
    val xType = implicitly[WeakTypeTag[X]]

    val q"{..$stmts}" = fn.tree
    val visitor = new BlockVisitor(new Scope(Map(TermName("interpreter") -> RichTree(q"interpreter"))))
    val newStmts = visitor.visitBlockStmts(stmts).stmts
    val newBody =
      q"""new scappla.Model[$xType] {

        def sample(interpreter: scappla.Interpreter) = {
          val scappla.Variable(value, node) = {
            ..${newStmts.map{_.tree}}
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