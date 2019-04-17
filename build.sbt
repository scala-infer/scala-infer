name := "scala-infer-parent"

ThisBuild / organization := "scala-infer"
ThisBuild / scalaVersion := "2.12.8"
ThisBuild / skip in publish := true

ThisBuild / bintrayOrganization := Some("scala-infer")

lazy val core = (project in file("core")).settings(
  moduleName := "scala-infer",
  libraryDependencies ++= Seq(
    "org.scala-lang"              % "scala-reflect"      % scalaVersion.value,
    "org.scala-lang"              % "scala-compiler"     % scalaVersion.value,
    "com.chuusai"                %% "shapeless"          % "2.3.2",
    "com.typesafe.scala-logging" %% "scala-logging"      % "3.9.0",
    "ch.qos.logback"              % "logback-classic"    % "1.2.3",

    "org.nd4j"                    % "nd4j-native-platform" % "1.0.0-beta3",

    "org.scalatest"              %% "scalatest"          % "3.0.1" % "test"
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  skip in publish := false,
  licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0"))
)

lazy val app = (project in file("app")).settings(
  moduleName := "infer-app",
//  mainClass in Compile := Some("scappla.app.TestChickweight"),
  // mainClass in Compile := Some("scappla.app.TestMixture"),
  mainClass in Compile := Some("scappla.app.TestTicTacToe"),
  libraryDependencies ++= Seq(
    "com.github.tototoshi"       %% "scala-csv"          % "1.3.5"
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  scalacOptions ++= Seq("-Ymacro-debug-lite", "-Xlog-implicits")
) dependsOn core

