package scappla.test

import scappla._
import scappla.distributions._

class CompiledInference {

    trait Obs[T] {
        def get: T
        def set(t: T): Unit
    }

    private class ListBuilder[T](obs: Obs[List[T]]) {

        private var _isEmpty: Option[Boolean] = None
        private var _head: Option[T] = None
        private var _tail: Option[List[T]] = None

        def isEmpty(): Obs[Boolean] = {
            new Obs[Boolean] {
                def get = obs.get.isEmpty
                def set(t: Boolean) = {
                    _isEmpty = Some(t)
                }
            }
        }

        def head(): Obs[T] = {
            new Obs[T] {
                def get = obs.get.head
                def set(t: T) = {
                    _head = Some(t)
                }
            }
        }

        def tail(): Obs[List[T]] = {
            new Obs[List[T]] {
                def get = obs.get.tail
                def set(t: List[T]) = {
                    _tail = Some(t)
                }
            }
        }

        def build: Unit = {
            val result = (_isEmpty.map { e =>
                if (e) {
                    Nil
                } else {
                    _head.get :: _tail.get
                }
            }).getOrElse {
                _head.get :: _tail.get
            }
            obs.set(result)
        }
    }

    object ListBuilder {

        def apply[T, U](list: Obs[List[T]])(fn: ListBuilder[T] => U): U = {
            val builder = new ListBuilder(list)
            val result = fn(builder)
            builder.build
            result
        }
    }

    def observe[T](dist: Distribution[T], obs: Obs[T]): T = ???

    def generate(list: Obs[List[Real]]): List[Real] = {
        // implicit val builder: ListBuilder[Real] = new ListBuilder(list)
        ListBuilder(list) { builder =>
            val last = observe(Bernoulli(0.5), builder.isEmpty())
            if (last) {
                Nil
            } else {
                val value = observe(Normal(0.5, 1.0), builder.head())
                value :: generate(builder.tail())
            }
        }
    }

}
