package scappla

import scala.language.experimental.macros

trait AbstractReal extends Value[Double, Unit] {

  override def field = BaseField.doubleBaseField

  override def shape = ()
}