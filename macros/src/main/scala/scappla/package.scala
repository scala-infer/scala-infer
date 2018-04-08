package object scappla {

  trait Distribution[A] {

    def sample: A
  }

  case class ConstantDistribution[A](value: A) extends Distribution[A] {
    override def sample: A = value
  }

  def sample[A](dist: Distribution[A]): A = {
    dist.sample
  }
}
