package scappla

import scala.collection.mutable
import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import scala.reflect.api.Trees

object Macro {

  def backward[A, B](fn: A => B): (A, B) => A = macro backward_impl[A, B]

  def backward_impl[A, B](c: Context)(fn: c.Expr[A => B])
      (implicit aTag: c.WeakTypeTag[A], bTag: c.WeakTypeTag[B])
  : c.Expr[(A, B) => A] = {

    import c.universe._

    val Function(List(ValDef(_, name, _, _)), funcBody) = fn.tree
    val TermName(strName) = name

    // topologically sort components - leaf components first, top last
    val allComponents = mutable.ListBuffer.empty[Component]
    def visitComponent(component: Component): Unit = {
      if (!allComponents.contains(component)) {
        component match {
          case Negate(arg) =>
            visitComponent(arg)
          case Invert(arg) =>
            visitComponent(arg)
          case Log(arg) =>
            visitComponent(arg)

          case Add(first, second) =>
            visitComponent(first)
            visitComponent(second)
          case Multiply(first, second) =>
            visitComponent(first)
            visitComponent(second)
          case Power(base, power) =>
            visitComponent(base)
            visitComponent(power)

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
    println()
    val topComponents = extractComponents(funcBody)(c)
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

    val gradientTree = gradients(Variable(strName)).reduce(Add).toTree(c)
    println(s"Result: ${gradientTree}")

    println(showRaw(fn))
    val result = c.Expr[(A, B) => A](Function(
      List(
        ValDef(Modifiers(Flag.PARAM), TermName("dy"), TypeTree(weakTypeOf[B]), EmptyTree),
        ValDef(Modifiers(Flag.PARAM), name, TypeTree(weakTypeOf[A]), EmptyTree)
      ),
      Block(
        List(),
        q"dy * $gradientTree"
      )
    ))
    println("Result: " + showRaw(result))
    result
  }

  def extractComponents(tree: Trees#Tree)(implicit c: Context): List[Component] = {
    import c.universe._
    tree match {
      case q"$nextTree + $arg" =>
        getComponent(arg) :: extractComponents(nextTree)
      case q"$nextTree - $arg" =>
        Negate(getComponent(arg)) :: extractComponents(nextTree)
      case somethingElse => getComponent(somethingElse) :: Nil
    }
  }

  def getComponent(tree: Trees#Tree)(implicit c: Context): Component = {
    import c.universe._
    tree match {
      case Ident(TermName(x)) => Variable(x)
      case Literal(Constant(a)) => DoubleConstant(a.toString.toDouble)
      case q"-$x" => Negate(getComponent(x))
      case q"+$x" => getComponent(x)
      case q"$a + $b" => Add(getComponent(a), getComponent(b))
      case q"$a - $b" => Add(getComponent(a), Negate(getComponent(b)))
      case q"$a * $b" => Multiply(getComponent(a), getComponent(b))
      case q"scala.math.`package`.pow($a, $b)" => Power(getComponent(a), getComponent(b))
    }
  }

}

sealed trait Component {
  def toTree(implicit c: Context): c.Tree

  def backward(dout: Component): Seq[(Component, Component)]
}

trait UnaryComponent {
  def value: Component
}

trait BinaryComponent {
  def first: Component

  def second: Component
}

case class Negate(value: Component) extends Component with UnaryComponent {
  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    q"-${value.toTree}"
  }

  override def backward(dout: Component) = Seq(value -> Negate(dout))
}

case class Multiply(first: Component, second: Component) extends Component with BinaryComponent {
  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    q"${first.toTree} * ${second.toTree}"
  }

  override def backward(dout: Component) =
    Seq(
      first -> Multiply(dout, second),
      second -> Multiply(dout, first)
    )
}

case class Invert(value: Component) extends Component with UnaryComponent {

  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    q"1.0 / ${value.toTree}"
  }

  override def backward(dout: Component) =
    Seq(
      value -> Multiply(Negate(dout), Invert(Multiply(value, value)))
    )
}

case class Add(first: Component, second: Component) extends Component with BinaryComponent {
  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    q"${first.toTree} + ${second.toTree}"
  }

  override def backward(dout: Component) =
    Seq(
      first -> dout,
      second -> dout
    )
}

case class Power(first: Component, second: Component) extends Component with BinaryComponent {
  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    q"Math.pow(${first.toTree}, ${second.toTree})"
  }

  override def backward(dout: Component) =
    Seq(
      first -> Multiply(dout, Multiply(second, Power(first, Add(second, Negate(DoubleConstant(1.0)))))),
      second -> Multiply(dout, Multiply(Log(first), Power(first, second)))
    )
}

case class Log(value: Component) extends Component with UnaryComponent {
  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    q"Math.log(${value.toTree})"
  }

  override def backward(dout: Component) =
    Seq(
      value -> Multiply(dout, Invert(value))
    )
}

case class Variable(name: String) extends Component {

  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    Ident(TermName(name))
  }

  override def backward(dout: Component) = Seq.empty
}

case class DoubleConstant(value: Double) extends Component {

  override def toTree(implicit c: Context): c.Tree = {
    import c.universe._
    Literal(Constant(value))
  }

  override def backward(dout: Component) = Seq.empty
}

