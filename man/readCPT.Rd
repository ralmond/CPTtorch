\name{readCPT}
\alias{readCPT}
\alias{writeCPT}
\title{Reads/Writes a CPT to a JSON File.}
\description{

  Reads and/or writes a \code{\linkS4class{CPT_Model}} object to a JSON representation.
}
\usage{
readCPT(serial, device = CPTtorch_device())
writeCPT(cpt)
}
\arguments{
  \item{cpt}{A \code{\linkS4class{CPT_Model}} to be saved.}
  \item{serial}{A JSON representation of a \code{\linkS4class{CPT_Model}}.}
  \item{device}{Should this be built on the CPU or GPU?}
}
\value{

  The function \code{readCPT} returns an object of class \code{\linkS4class{CPT_Model}}.
  The function \code{writeCPT} returns a string containing a JSON representation.
}
\author{Krishna Saxena}
\examples{
  mod0 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod0$penalties <- list("bVec"=1,"sVec"=1)

  mod0Ser <- writeCPT(mod0)
  modRW <- readCPT(mod0Ser, device = torch_device("CPU"))


}
\keyword{ IO }
\concept{ CPT }
\concept{ JSON }

