resolvers += Resolver.sonatypeRepo("releases")
resolvers += Resolver.DefaultMavenRepository
addSbtPlugin("org.foundweekends" % "sbt-bintray" % "0.5.4")
addSbtPlugin("com.github.gseitz" % "sbt-release" % "1.0.11")
