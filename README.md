Scala Infer
===========
With the addition of a few keywords, `scala-infer` turns scala into a probabilistic programming language.
To achieve scalability, inference is based on gradients of the (variational approximation to) the
posterior distribution.  Each draw from a distribution is accompanied by a *guide* distribution.

A probabilistic model in `scala-infer` is written as regular scala code.  Values are drawn from
distributions and are used to generate data.  Such a model is known as *generative*, as it provides
an explicit process.

Three new keywords are introduced:
* infer
* sample
* observe

When a value is sampled, two distributions are needed; the *prior* and the (variational
approximation to) the *posterior*.  Actual sample values are drawn from the posterior, but
the prior is the starting point of the buildup of the posterior.  Without observations, the
posterior would be equal to the prior.

Parameters of the variational posterior are optimized by gradient descent.  For each sample from
the model, a backward pass calculates the gradients of the loss function.  For variational inference,
the loss function is the ELBO, a lower bound on the evidence.

Different tactics are used for discrete and continuous variables.  For continuous variables,
the reparametrization trick can be used to obtain a low variance estimator.  Discrete variables
use black-box variational inference, which only requires gradients of the score function to the
parameters.

It is possible to nest models.  This makes it possible to modularize the model.

```scala
// optimization algorithm: (stochastic) gradient descent
val sgd = new SGD()

// posterior distribution for the sprinkler, conditional on rain.
// The parameters run over the full real axis, they are mapped to the
// domain [0,1] by the sigmoid transformation.
val inRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))
val noRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

// conditional sampling of the sprinkler.  The probability that
// the sprinkler turned on, is dependent on whether it rained or not.
val sprinkle = infer {
  rain: Boolean =>
    if (rain) {
      sample(Bernoulli(0.01), inRain)
    } else {
      sample(Bernoulli(0.4), noRain)
    }
}

// posterior distribution for the rain
val rainPost = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

// full model of the rain-sprinkler-grass system.  
val model = infer {

  val rain = sample(Bernoulli(0.2), rainPost)
  val sprinkled = sample(sprinkle(rain))

  val p_wet = (rain, sprinkled) match {
    case (true,  true)  => 0.99
    case (false, true)  => 0.9
    case (true,  false) => 0.8
    case (false, false) => 0.001
  }

  // bind model to data / add observation
  observe(Bernoulli(p_wet), true)

  // return quantity we're interested in
  rain
}
```
