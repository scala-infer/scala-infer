
name := "scala-infer-parent"

ThisBuild / organization := "scala-infer"
ThisBuild / scalaVersion := "2.13.1"
ThisBuild / skip in publish := true
ThisBuild / useCoursier := false

ThisBuild / bintrayOrganization := Some("scala-infer")

lazy val core = (project in file("core")).settings(
  moduleName := "scala-infer",
  libraryDependencies ++= Seq(
    "org.scala-lang"              % "scala-reflect"      % scalaVersion.value,
    "org.scala-lang"              % "scala-compiler"     % scalaVersion.value,
    "com.chuusai"                %% "shapeless"          % "2.3.3",
    "com.typesafe.scala-logging" %% "scala-logging"      % "3.9.2",
    "ch.qos.logback"              % "logback-classic"    % "1.2.3",
    "org.scalanlp" %% "breeze" % "1.0",
    "org.scalanlp" %% "breeze-natives" % "1.0",


    "org.scalatest"              %% "scalatest"          % "3.1.1" % "test"
  ),
  skip in publish := false,
  licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0")),
  scalacOptions ++= Seq("-Ymacro-annotations"),
//  scalacOptions ++= Seq("-Xlog-implicits")
//  scalacOptions ++= Seq("-Ymacro-debug-lite")
)

lazy val nd4j = (project in file("nd4j")).settings(
  moduleName := "scala-infer-nd4j",
  libraryDependencies ++= Seq(
    "org.nd4j"                    % "nd4j-native-platform" % "1.0.0-beta5",
    "org.scalatest"              %% "scalatest"          % "3.0.1" % "test"
  ),
  skip in publish := false,
  licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0")) //,
) dependsOn core

lazy val app = (project in file("app")).settings(
  moduleName := "infer-app",
  // mainClass in Compile := Some("scappla.app.TestChickweight"),
  // mainClass in Compile := Some("scappla.app.TestMixture"),
  mainClass in Compile := Some("scappla.app.TestFullTicTacToe"),
  // mainClass in Compile := Some("scappla.app.TestTicTacToe"),
  libraryDependencies ++= Seq(
    "com.github.tototoshi"       %% "scala-csv"          % "1.3.6"
  ),
  scalacOptions ++= Seq("-Ymacro-annotations"),
  // scalacOptions ++= Seq("-Ymacro-debug-lite", "-Xlog-implicits"),
) dependsOn core

