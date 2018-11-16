name := "macro-ppl"

version := "0.1"

scalaVersion := "2.12.4"

val nd4jVersion = "1.0.0-beta3"

lazy val macros = (project in file("macros")).settings(
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
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
)

lazy val app = (project in file("app")).settings(
  mainClass in Compile := Some("scappla.TestChickweight"),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  scalacOptions ++= Seq("-Ymacro-debug-verbose")
) dependsOn macros

