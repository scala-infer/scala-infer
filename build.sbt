name := "scala-infer-parent"

ThisBuild / organization := "fvlankvelt"
ThisBuild / version := "0.1"
ThisBuild / scalaVersion := "2.12.8"

val nd4jVersion = "1.0.0-beta3"

lazy val macros = (project in file("macros")).settings(
  moduleName := "scala-infer",
  libraryDependencies ++= Seq(
    "org.scala-lang"              % "scala-reflect"      % scalaVersion.value,
    "org.scala-lang"              % "scala-compiler"     % scalaVersion.value,
    "com.chuusai"                %% "shapeless"          % "2.3.2",
    "com.typesafe.scala-logging" %% "scala-logging"      % "3.9.0",
    "ch.qos.logback"              % "logback-classic"    % "1.2.3",

    "org.nd4j"                    % "nd4j-native-platform" % nd4jVersion,

    "com.github.tototoshi"       %% "scala-csv"          % "1.3.5",
    "org.scalatest"              %% "scalatest"          % "3.0.1" % "test"
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0"))
)

lazy val app = (project in file("app")).settings(
  moduleName := "infer-app",
  mainClass in Compile := Some("scappla.app.TestSprinkler"),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  skip in publish := true
//  scalacOptions ++= Seq("-Ymacro-debug-verbose")
) dependsOn macros

