name := "macro-ppl"

version := "0.1"

scalaVersion := "2.11.8"

lazy val macros = (project in file("macros")).settings(
  libraryDependencies ++= Seq(
    "org.scala-lang" % "scala-reflect" % scalaVersion.value,
    "org.scalatest" %% "scalatest" % "3.0.1" % "test"
  )
)

lazy val app = (project in file("app")).settings(
  mainClass in Compile := Some("scappla.TestMacro")
) dependsOn macros
