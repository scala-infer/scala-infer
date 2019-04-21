package scappla

trait Sampleable[A] {

  def sample(interpreter: Interpreter): A
}
