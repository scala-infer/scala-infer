package scappla

object TestMacro extends App {

  Macro.backward[Double, Double] { z =>
    //        val x = z * z
    val x = z
    x
  }

}
