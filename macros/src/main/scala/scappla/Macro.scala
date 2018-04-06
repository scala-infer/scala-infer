package scappla

import scala.annotation.StaticAnnotation
import scala.collection.mutable
import scala.language.experimental.macros
import scala.reflect.api.Trees
import scala.reflect.macros.whitebox.Context

trait DFunction1[X, Y] extends (X => Y) {

  // backwards propagate difference dy to dx
  def grad(x: X, dy: Y): X
}

trait DFunction2[@specialized(Double) X, @specialized(Double) Y, @specialized(Double) Z] extends ((X, Y) => Z) {

  // backwards propagate difference dy to dx
  def grad1(x: X, y: Y, dz: Z): X

  def grad2(x: X, y: Y, dz: Z): Y
}

object Functions {

  object cos extends DFunction1[Double, Double] {

    override def apply(x: Double): Double =
      scala.math.cos(x)

    override def grad(x: Double, dy: Double): Double =
      -dy * scala.math.sin(x)
  }

  object log extends DFunction1[Double, Double] {

    override def apply(x: Double): Double =
      scala.math.log(x)

    override def grad(x: Double, dy: Double): Double =
      dy / x
  }

  object exp extends DFunction1[Double, Double] {

    override def apply(x: Double): Double =
      scala.math.exp(x)

    override def grad(x: Double, dy: Double): Double =
      dy * scala.math.exp(x)
  }

  object pow extends DFunction2[Double, Double, Double] {

    override def apply(base: Double, exp: Double): Double =
      scala.math.pow(base, exp)

    override def grad1(base: Double, exp: Double, dz: Double): Double =
      dz * exp * scala.math.pow(base, exp - 1)

    override def grad2(base: Double, exp: Double, dz: Double): Double =
      dz * scala.math.log(base) * scala.math.pow(base, exp)
  }
}

class differentiate extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro Macro.impl
}

class Macro(val c: Context) {

  import c.universe._

  def impl(annottees: c.Expr[Any]*): c.Expr[Any] = {
    val inputs = annottees.map(_.tree).toList
    /*
    for (input <- inputs) {
      println(showRaw(input))
      println()
    }
    */
    inputs match {
      case (method: DefDef) :: _ =>
        val DefDef(mods, name, List(), List(List(ValDef(_, argName, valType, _))), tpt, body) = method
        val result = ModuleDef(
          Modifiers(),
          name,
          Template(
            List(AppliedTypeTree(Select(Ident(TermName("scappla")), TypeName("DFunction1")), List(valType, tpt))),
            noSelfType,
            List(
              DefDef(
                Modifiers(),
                termNames.CONSTRUCTOR,
                List(),
                List(List()),
                TypeTree(),
                Block(List(Apply(Select(Super(This(typeNames.EMPTY), typeNames.EMPTY), termNames.CONSTRUCTOR), List())), Literal(Constant(())))
              ),
              DefDef(
                Modifiers(),
                TermName("apply"),
                List(),
                List(List(ValDef(Modifiers(), argName, valType, EmptyTree))),
                tpt,
                body
              ),
              DefDef(
                Modifiers(),
                TermName("grad"),
                List(),
                List(List(
                  ValDef(Modifiers(), argName, valType, EmptyTree),
                  ValDef(Modifiers(), TermName("dy"), tpt, EmptyTree)
                )),
                valType,
                backward_impl(argName, valType, body, tpt)
              )
            )
          )
        )
//        println(s"RESULT: ${showCode(result)}")
//        val tree = ClassDef(mods, termName.toTypeName, tparams, Template())
//        c.Expr[Any](Block(List(), Literal(Constant(()))))
        c.Expr[Any](result)

      case _ =>
        (EmptyTree, inputs)
        c.Expr[Any](Literal(Constant(())))
    }
  }

  def backward_impl(name: TermName, valType: Tree, funcBody: Tree, resultType: Tree): c.Tree = {

    val TermName(strName) = name

    val transformer = new Transformer {

      val transforms = mutable.ListBuffer.empty[(String, c.Tree)]

      override def transform(tree: c.Tree) = {
        tree match {
          case ValDef(_, TermName(valName), valTpe, valDefBody) =>
            val res = super.transform(tree)
            if (res.exists {
              case Ident(term) if term == name => true
              case _ => false
            }) {
              val dValName = TermName(s"_d_${valName}")
              transforms.prepend(
                (
                    valName,
                    ValDef(
                      Modifiers(),
                      dValName,
                      TypeTree(),
                      calcGradient(valDefBody, strName).toTree
                    )
                )
              )
            }
            res
          case _ =>
            super.transform(tree)
        }
      }
    }

    val newBody = funcBody match {
      case Apply(_, _) =>
        q"dy * ${calcGradient(funcBody, strName).toTree}"
      case Block(_, _) =>
        val Block(newStats, newExpr) = transformer.transform(
          c.untypecheck(funcBody)
        )
        Block(
          newStats ++ transformer.transforms.map(_._2).toList,
          Apply(
            Select(Ident(TermName("dy")), TermName("$times")),
            List(transformer.transforms.map(_._1).foldLeft(calcGradient(newExpr, strName)) {
              case (acc, valName) =>
                Add(acc, Multiply(Variable(s"_d_${valName}"), calcGradient(newExpr, valName)))
            }.toTree)
          )
        )
    }

    println(s"Result: ${showCode(newBody)}")
    newBody
  }

  def calcGradient(funcBody: c.Tree, strName: String) = {
    // topologically sort components - leaf components first, top last
    val allComponents = mutable.ListBuffer.empty[Component]

    def visitComponent(component: Component): Unit = {
      if (!allComponents.contains(component)) {
        component match {
          case Negate(arg) =>
            visitComponent(arg)
          case Invert(arg) =>
            visitComponent(arg)
          case Op1(_, arg) =>
            visitComponent(arg)

          case Add(first, second) =>
            visitComponent(first)
            visitComponent(second)
          case Multiply(first, second) =>
            visitComponent(first)
            visitComponent(second)
          case Op2(_, base, power) =>
            visitComponent(base)
            visitComponent(power)

          case Op3(_, _, _, _) => ??? // Not supported - cannot take gradients of it

          case Variable(_) =>
          case DoubleConstant(_) =>
        }
        allComponents += component
      }
    }

    val gradients = mutable.HashMap
        .empty[Component, mutable.ListBuffer[Component]]
        .withDefault(_ => mutable.ListBuffer.empty[Component])

    def addGradients(backward: (Component, Component)): Unit = {
      val (key, value) = backward
      val newList = gradients(key) :+ value
      gradients.put(key, newList)
    }

    // collect components and bootstrap the gradient
    val topComponents = extractComponents(funcBody)
    for {
      component <- topComponents
    } {
      visitComponent(component)
      addGradients((component, DoubleConstant(1.0)))
    }

    // back propagate the gradients
    for {
      component <- allComponents.reverse
    } {
      val grads = gradients(component)
      val dout = grads.reduce(Add)
      for {
        grad <- component.backward(dout)
      } {
        addGradients(grad)
      }
    }

    gradients(Variable(strName))
        .reduceOption(Add)
        .getOrElse(DoubleConstant(0.0))
  }

  def extractComponents(tree: Trees#Tree): List[Component] = {
    import c.universe._
    tree match {
      case q"$nextTree + $arg" =>
        getComponent(arg) :: extractComponents(nextTree)
      case q"$nextTree - $arg" =>
        Negate(getComponent(arg)) :: extractComponents(nextTree)
      case somethingElse => getComponent(somethingElse) :: Nil
    }
  }

  def getComponent(tree: Trees#Tree): Component = {
    import c.universe._
    tree match {
      case Ident(TermName(x)) => Variable(x)
      case Literal(Constant(a)) => DoubleConstant(a.toString.toDouble)
      case q"-$x" => Negate(getComponent(x))
      case q"+$x" => getComponent(x)
      case q"$a + $b" => Add(getComponent(a), getComponent(b))
      case q"$a - $b" => Add(getComponent(a), Negate(getComponent(b)))
      case q"$a * $b" => Multiply(getComponent(a), getComponent(b))
      case Apply(Select(fun, TermName("apply")), List(a)) => Op1(fun, getComponent(a))
      case Apply(Select(fun, TermName("apply")), List(a, b)) => Op2(fun, getComponent(a), getComponent(b))
      case Apply(fun, List(a)) => Op1(fun, getComponent(a))
      case Apply(fun, List(a, b)) => Op2(fun, getComponent(a), getComponent(b))
    }
  }

  sealed trait Component {
    def toTree: c.Tree

    def backward(dout: Component): Seq[(Component, Component)]
  }

  case class Negate(value: Component) extends Component {
    override def toTree: c.Tree = {
      import c.universe._
      q"-${value.toTree}"
    }

    override def backward(dout: Component) = Seq(value -> Negate(dout))
  }

  case class Multiply(first: Component, second: Component) extends Component {
    override def toTree: c.Tree = {
      import c.universe._
      q"${first.toTree} * ${second.toTree}"
    }

    override def backward(dout: Component) =
      Seq(
        first -> Multiply(dout, second),
        second -> Multiply(dout, first)
      )
  }

  case class Invert(value: Component) extends Component {

    override def toTree: c.Tree = {
      import c.universe._
      q"1.0 / ${value.toTree}"
    }

    override def backward(dout: Component) =
      Seq(
        value -> Multiply(Negate(dout), Invert(Multiply(value, value)))
      )
  }

  case class Add(first: Component, second: Component) extends Component {
    override def toTree: c.Tree = {
      import c.universe._
      q"${first.toTree} + ${second.toTree}"
    }

    override def backward(dout: Component) =
      Seq(
        first -> dout,
        second -> dout
      )
  }

  case class Op1(fn: c.Tree, value: Component) extends Component {
    override def toTree: c.Tree = {
      import c.universe._
      q"$fn(${value.toTree})"
    }

    override def backward(dout: Component) =
      Seq(
        value -> Op2(Select(fn, TermName("grad")), value, dout)
      )
  }

  case class Op2(fn: c.Tree, first: Component, second: Component) extends Component {
    override def toTree: c.Tree = {
      import c.universe._
      q"$fn(${first.toTree}, ${second.toTree})"
    }

    override def backward(dout: Component) =
      Seq(
        first -> Op3(Select(fn, TermName("grad1")), first, second, dout),
        second -> Op3(Select(fn, TermName("grad2")), first, second, dout),
      )
  }

  case class Op3(fn: c.Tree, first: Component, second: Component, third: Component) extends Component {
    override def toTree: c.Tree = {
      import c.universe._
      q"$fn(${first.toTree}, ${second.toTree}, ${third.toTree})"
    }

    override def backward(dout: Component): Seq[(Component, Component)] = ???
  }

  case class Variable(name: String) extends Component {

    override def toTree: c.Tree = {
      import c.universe._
      Ident(TermName(name))
    }

    override def backward(dout: Component) = Seq.empty
  }

  case class DoubleConstant(value: Double) extends Component {

    override def toTree: c.Tree = {
      import c.universe._
      Literal(Constant(value))
    }

    override def backward(dout: Component) = Seq.empty
  }

}
