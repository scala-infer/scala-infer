package scappla

import scala.annotation.StaticAnnotation
import scala.reflect.macros.whitebox
import scala.language.experimental.macros

class sampled extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro Cps.impl
}

class Cps(val c: whitebox.Context) {

  import c.universe._

  def impl(annottees: c.Expr[Any]*): c.Expr[Any] = {
    val inputs = annottees.map(_.tree).toList
    inputs match {
      case (method: DefDef) :: _ =>
        val tf = new Transformer {
//          override val treeCopy = newStrictTreeCopier

          var sampleCounter = 0

          private def nextSampleName(): TermName = {
            sampleCounter += 1
            TermName(s"sample$$$sampleCounter")
          }
//
          override def transform(tree: c.universe.Tree): c.universe.Tree = {
            println(s"TRANSFORMING: ${showRaw(tree)}")
            tree match {
              case Apply(Ident(TermName("sample")), List(arg)) =>
                println(s"  MATCHED sample")
                val tfArg = transform(arg)
                val sampleName = nextSampleName()
                val out = Block(
                  List(
                    ValDef(Modifiers(), sampleName, TypeTree(arg.tpe), tfArg)
                  ),
                  treeCopy.Apply(
                    tree,
                    Select(Ident(TermName("scappla")), TermName("sample")),
                    List(Ident(sampleName))
                  )
                )
                println(s"  OUT: ${showCode(out)}")
                out
              case _ =>
                super.transform(tree)
            }
          }
        }
        c.Expr[Any](tf.transform(c.untypecheck(method)))
      case _ =>
        (EmptyTree, inputs)
        c.Expr[Any](Literal(Constant(())))
    }
  }

}
