package scappla.test

import scala.reflect.runtime.universe._

import scappla._
import scappla.distributions._
import scala.collection.compat.BuildFrom

class CompiledInference {

  trait Data[T] {

    /**
      * The value of the data.  Known when used to wrap observational data,
      * in which case `observe` statements condition their distribution on it
      * to provide the likelihood.
      *
      * When compiling the model, the value is sampled (for primitive types)
      * or built from components (for compound types).  The value is only
      * known when the forward pass has completed and sufficient information
      * has been provided.
      */
    def value: T
  }

  object Data {

    def apply[T, DT <: Data[T]](
        value: T
    )(implicit df: DataFactory.Aux[T, DT]): DT =
      df(Some(value))
  }

  /**
    * Primitive data types can be sampled directly from (or observed for) elemental
    * probability distributions that are able to calculate (normalized) likelihoods.
    */
  class Primitive[T](observation: Option[T]) extends Data[T] {
    private var sample: Option[T] = None

    lazy val value: T = observation.getOrElse(sample.get)

    def set(v: T): Unit = {
      assert(observation.isEmpty && sample.isEmpty)
      sample = Some(v)
    }
  }

  type RealData = Primitive[Real]

  type BoolData = Primitive[Boolean]

  trait DataFactory[T] {

    type DataType <: Data[T]

    def apply(t: Option[T]): DataType
  }

  trait LowPrioDataFactory {

    implicit def aux[T](
        implicit o: DataFactory[T]
    ): DataFactory.Aux[T, o.DataType] = o
  }

  object DataFactory extends LowPrioDataFactory {

    type Aux[T, ET0] = DataFactory[T] {
      type DataType = ET0
    }

    implicit val realFactory: DataFactory[Real] = new DataFactory[Real] {

      type DataType = RealData

      def apply(t: Option[Real]): RealData = new RealData(t)
    }

    implicit def listFactory[T, ET <: Data[T]](
        implicit edf: DataFactory.Aux[T, ET]
    ): DataFactory[List[T]] = new DataFactory[List[T]] {

      type DataType = ListData[T, ET]

      def apply(t: Option[List[T]]): DataType = new ListData(t)(edf)
    }
  }

  class ListData[T, ET <: Data[T]](
      observation: Option[List[T]]
  )(
      implicit
      elFactory: DataFactory.Aux[T, ET]
  ) extends Data[List[T]] {

    private val builder: Option[ObsBuilder] =
      if (observation.isEmpty) Some(new ObsBuilder()) else None

    lazy val value: List[T] = observation.getOrElse(builder.get.build)

    def isEmpty: BoolData = {
      if (observation.isEmpty)
        builder.get.empty
      else
        new BoolData(Some(observation.get.isEmpty))
    }

    def head: ET = {
      if (observation.isDefined) {
        elFactory(observation.map { _.head })
      } else {
        builder.get.head
      }
    }

    def tail: ListData[T, ET] = {
      if (observation.isDefined) {
        new ListData(Some(observation.get.tail))
      } else {
        builder.get.tail
      }
    }

    private case class ObsBuilder() {
      lazy val empty: BoolData = new BoolData(None)
      lazy val head: ET = elFactory(None)
      lazy val tail: ListData[T, ET] = new ListData(None)

      lazy val build: List[T] = {
        if (empty.value) {
          Nil
        } else {
          head.value :: tail.value
        }
      }
    }
  }

  def observe[T, DT <: Primitive[T]](dist: Distribution[T], obs: DT): T = ???

  def generate(obs: ListData[Real, RealData]): List[Real] = {
    val last = observe(Bernoulli(0.5), obs.isEmpty)
    if (last) {
      Nil
    } else {
      val value = observe(Normal(0.5, 1.0), obs.head)
      value :: generate(obs.tail)
    }
  }

  val obs = Data(List(1.0: Real, 2.0: Real))
  val realData = Data(1.0: Real)

  val values = generate(obs)
}
