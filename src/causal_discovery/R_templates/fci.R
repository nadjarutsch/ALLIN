# R script to prepare and run the PC algorithms

# MIT License
#
# Copyright (c) 2018 Diviyan Kalainathan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

library('pcalg')
library('kpcalg')
library('methods')
library('RCIT')

runFCI <- function(X, suffStat, parentsOf, alpha, variableSelMat, setOptions,
                  directed, verbose,fixedEdges,fixedGaps,
                  result, ...){

  dots <- list(...)
  if(length(dots) > 0){
    warning("options provided via '...' not taken")
  }

  # additional options for PC
  optionsList <- list("indepTest"={CITEST}, "fixedEdges"=fixedEdges,
                      "NAdelete"=TRUE, "m.max"=Inf,
                      "skel.method"= "stable", "type=" = "normal", "conservative"=FALSE,
                      "maj.rule"=TRUE, "jci"="123", "contextVars"={CONTEXTVARS}, numCores={NJOBS})



  if(is.null(suffStat)){
    suffStat <- list({METHOD_INDEP})  # data=X, ic.method=method_indep)
  }

  print(optionsList$jci)
  fci.fit <- pcalg::fci(suffStat,
                      indepTest = optionsList$indepTest,
                      p=ncol(X),
                      alpha = alpha,
                      fixedGaps= fixedGaps,
                      fixedEdges = fixedEdges ,
                      NAdelete= optionsList$NAdelete,
                      m.max= optionsList$m.max,
                      skel.method= optionsList$skel.method,
                      conservative= optionsList$conservative,
                      maj.rule= optionsList$maj.rule,
                      jci = optionsList$jci,
                      contextVars = optionsList$contextVars,
                      verbose= verbose,
                      selectionBias=FALSE)

  fcimat <- fci.fit@amat

  result <- vector("list", length = length(parentsOf))

  for (k in 1:length(parentsOf)){
    result[[k]] <- which(as.logical(fcimat[, parentsOf[k]]))
    attr(result[[k]],"parentsOf") <- parentsOf[k]
  }

  if(length(parentsOf) < ncol(X)){
    fcimat <- fcimat[,parentsOf]
  }

  list(resList = result, resMat = fcimat)
}

dataset <- read.csv(file='{FOLDER}{FILE}', header=FALSE, sep=",");

variableSelMat = {SELMAT}  # NULL
directed = {DIRECTED}  # TRUE
verbose = {VERBOSE} # FALSE
setOptions = {SETOPTIONS} # NULL
parentsOf = 1:ncol(dataset)
alpha <- {ALPHA} # 0.01
fixedGaps = NULL
fixedEdges = NULL

if ({E_GAPS}){
  fixedGaps <- read.csv(file='{FOLDER}{GAPS}', sep=",", header=FALSE) # NULL
  fixedGaps = (data.matrix(fixedGaps))
  rownames(fixedGaps) <- colnames(fixedGaps)
}
if ({E_EDGES}){
  fixedEdges <- read.csv(file='{FOLDER}{EDGES}', sep=",", header=FALSE) # NULL
  fixedEdges = (data.matrix(fixedEdges))
  rownames(fixedEdges) <- colnames(fixedEdges)
}
result <- runFCI(dataset, suffStat = NULL, parentsOf, alpha,
               variableSelMat, setOptions,
               directed, verbose, fixedEdges, fixedGaps, CI_test)

write.csv(result$resMat,row.names = FALSE, file = '{FOLDER}{OUTPUT}');