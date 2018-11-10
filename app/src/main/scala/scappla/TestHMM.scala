package scappla

import scappla.Functions.sigmoid
import scappla.distributions.{Bernoulli, Distribution}
import scappla.guides.{BBVIGuide, Guide}
import scappla.optimization.SGD

object TestHMM extends App {
/*
  import Real._

  var trueObservations = Seq(true, false, false, false)

  val sgd = new SGD()
  val dataWithGuide = trueObservations.map { observation =>
    (observation, BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0, 1.0)))))
  }

  val model = infer {

    val transition = {
      (s: Boolean, dist: Guide[Boolean]) =>
        if (s) {
          sample(Bernoulli(0.7), dist)
        } else {
          sample(Bernoulli(0.3), dist)
        }
    }

    dataWithGuide.foldRight(Seq(true)) {
      case (datum, states) =>
        val lastState = states.head
        val newState = transition(lastState, datum._2)
        val p = if (newState) 0.9 else 0.1
        observe(Bernoulli(p), datum._1)
        newState +: states
    }
  }

  // warm up
  Range(0, 100).foreach { i =>
    sample(model)
  }

  // print some samples
  Range(0, 10).foreach { i =>
    val states = sample(model)
    println(s"  $states")
  }
  */

  /*
  var transition = function(s) {
    return s ? flip(0.7) : flip(0.3);
  };

  var observe = function(s) {
    return s ? flip(0.9) : flip(0.1);
  };

  var hmm = function(n) {
    var prev = (n == 1) ? {states: [true], observations: []} : hmm(n - 1);
    var newState = transition(prev.states[prev.states.length - 1]);
    var newObs = observe(newState);
    return {
      states: prev.states.concat([newState]),
      observations: prev.observations.concat([newObs])
    };
  };

  var trueObservations = [true, false, false, false];

  var dist = Infer({method: 'enumerate'}, function() {
    var r = hmm(4);
    factor(_.isEqual(r.observations, trueObservations) ? 0 : -Infinity);
    return r.states;
  });

  viz.table(dist);
  */
}
