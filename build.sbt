name := "macro-ppl"

version := "0.1"

scalaVersion := "2.12.4"

lazy val macros = (project in file("macros")).settings(
  libraryDependencies ++= Seq(
    "org.scala-lang"              % "scala-reflect"      % scalaVersion.value,
    "org.scala-lang"              % "scala-compiler"     % scalaVersion.value,
    "com.chuusai"                %% "shapeless"          % "2.3.2",
    "com.typesafe.scala-logging" %% "scala-logging"      % "3.9.0",
    "ch.qos.logback"              % "logback-classic"    % "1.2.3",

    "org.scalatest"              %% "scalatest"          % "3.0.1" % "test"
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
)

lazy val app = (project in file("app")).settings(
  mainClass in Compile := Some("scappla.TestAutoDiff"),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  scalacOptions ++= Seq("-Ymacro-debug-verbose")
) dependsOn macros

