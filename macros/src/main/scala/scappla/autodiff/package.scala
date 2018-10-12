package scappla

import scala.language.experimental.macros

package object autodiff {

  def toReal(fn: Double => Double): DFunction1 = macro AutoDiff.diffMacro

}
