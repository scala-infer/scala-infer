package scappla

object TestMacro extends App {

  Macro.backward[Double, Double](z => z * z)
}
