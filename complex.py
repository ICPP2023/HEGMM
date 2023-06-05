import numpy as np
import random
import time
from resource import getrusage as resource_usage, RUSAGE_SELF
import argparse
import tracemalloc
from algo2 import algo2
from algo1 import algo1
from HuangMid import huang
from baselineap0 import baselineap0
from baselinelog import baselinelog
import logging
import sys

DEBUG = 0

def main():
  parser = argparse.ArgumentParser(description='Parameter Processing')
  parser.add_argument('--traceMemo', type=bool, default=False, help='trace memory or not, if trace, time is longer than usual')

  args = parser.parse_args()
  LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
  logging.basicConfig(filename='test.log', level=logging.DEBUG, format=LOG_FORMAT)

  E2DM_S = 1
  E2DM_R = 1
  HEGMM = 1
  HEGMM_EN = 1
  HUANG = 1

  a = np.random.randint(low=1,high=64)
  b = np.random.randint(low=1,high=64)
  c = np.random.randint(low=1,high=64)

  # print("a: ", a," b: ", b," c: ", c)
  cipherSize = 2**14

  matrixA = np.zeros((a,b))
  matrixB = np.zeros((b,c))

  for i in range(matrixA.shape[0]):
    for j in range(matrixA.shape[1]):
      matrixA[i][j] = np.random.randint(low=1,high=9)

  for i in range(matrixB.shape[0]):
    for j in range(matrixB.shape[1]):
      matrixB[i][j] = np.random.randint(low=1,high=9)

  #Multiplies the input matrices using permutation matrices.
  trueC = np.matmul(matrixA, matrixB)

  algoOne = algo1(cipherSize)
  algoTwo = algo2(cipherSize)
  algoThr = algo3(cipherSize)
  algoHuang = huang(cipherSize)
  baseap = baselineap0(cipherSize)
  baselg = baselinelog(cipherSize)

  BaseMatrixA = matrixA.copy()
  BaseMatrixB = matrixB.copy()
  if E2DM_R:
    baseS = resource_usage(RUSAGE_SELF)
    if args.traceMemo:
      tracemalloc.start()
    Arow = BaseMatrixA.shape[0]
    Acol = BaseMatrixA.shape[1]
    Brow = BaseMatrixB.shape[0]
    Bcol = BaseMatrixB.shape[1]
    # maxDim = (int(max(Brow,Bcol)/Arow) + 1*(max(Brow,Bcol)%Arow)) * Arow
    maxDim = (int(max(Brow,Bcol)/Arow) + 1*np.sign(max(Brow,Bcol)%Arow)) * Arow
    if Arow <= Acol and maxDim <= 64:
      BaseLGMatrixA = np.pad(BaseMatrixA, ((0,0),(0,maxDim-Acol)))
      BaseLGMatrixB = np.pad(BaseMatrixB, ((0,maxDim-Brow),(0,maxDim-Bcol)))
      baseLGC = baselg.matrixMul(BaseLGMatrixA, BaseLGMatrixB)
      baseLGC = baseLGC[:Arow, :Bcol]
      if args.traceMemo:
        _,E2DM_RPeak = tracemalloc.get_traced_memory() 
        tracemalloc.stop()
      basetimeLGE = resource_usage(RUSAGE_SELF)
    else:
      if args.traceMemo:
        _,E2DM_RPeak = tracemalloc.get_traced_memory() 
        tracemalloc.stop()
      baseLGC = trueC
      basetimeLGE = baseS

    if (baseLGC == trueC).all():
      logging.info("E2DM_R Right")
      logging.info("%d %d %d %.2f", a,b,c
        , ((basetimeLGE.ru_utime - baseS.ru_utime)+(basetimeLGE.ru_stime - baseS.ru_stime))*1000
        )
      if args.traceMemo:
        logging.info("%d %d %d %.2fKB", a,b,c,E2DM_RPeak / 1024)
    else:
      logging.warning("E2DM_R Wrong")
      logging.warning("%d %d %d %.2f", a,b,c, ((basetimeLGE.ru_utime - baseS.ru_utime)+(basetimeLGE.ru_stime - baseS.ru_stime))*1000)

  if E2DM_S:
    baseAPS = resource_usage(RUSAGE_SELF)
    if args.traceMemo:
      tracemalloc.start()
    baseAP = baseap.matrixMul(BaseMatrixA, BaseMatrixB)
    if args.traceMemo:
      _,E2DM_SPeak = tracemalloc.get_traced_memory() 
      tracemalloc.stop()
    baseE = resource_usage(RUSAGE_SELF)

    if (baseAP == trueC).all():
      logging.info("E2DM_S Right")
      logging.info("%d %d %d %.2f", a,b,c
        , ((baseE.ru_utime - baseAPS.ru_utime)+(baseE.ru_stime - baseAPS.ru_stime))*1000
        )
      if args.traceMemo:
        logging.info("%d %d %d %.2fKB", a,b,c,E2DM_SPeak / 1024)
    else:
      logging.warning("E2DM_S Wrong")
      logging.warning("%d %d %d %.2f", a,b,c, ((baseE.ru_utime - baseAPS.ru_utime)+(baseE.ru_stime - baseAPS.ru_stime))*1000)

  if HEGMM:
    algoOneS = resource_usage(RUSAGE_SELF)
    AlgoMatrixA = matrixA.copy()
    AlgoMatrixB = matrixB.copy()
    if args.traceMemo:
      tracemalloc.start()
    algoOneC = algoOne.matrixMul(AlgoMatrixA, AlgoMatrixB)
    if args.traceMemo:
      _,HEGMMPeak = tracemalloc.get_traced_memory() 
      tracemalloc.stop()
    algoOneE = resource_usage(RUSAGE_SELF)

    if (algoOneC == trueC).all():
      logging.info("HEGMM Right")
      logging.info("%d %d %d %.2f", a,b,c
        , ((algoOneE.ru_utime - algoOneS.ru_utime)+(algoOneE.ru_stime - algoOneS.ru_stime))*1000
        )
      if args.traceMemo:
        logging.info("%d %d %d %.2fKB", a,b,c,HEGMMPeak / 1024)
    else:
      logging.warning("HEGMM Wrong")
      logging.warning("%d %d %d %.2f", a,b,c, ((algoOneE.ru_utime - algoOneS.ru_utime)+(algoOneE.ru_stime - algoOneS.ru_stime))*1000)

  if HEGMM_EN:
    algoTwoS = resource_usage(RUSAGE_SELF)
    AlgoMatrixA = matrixA.copy()
    AlgoMatrixB = matrixB.copy()
    if args.traceMemo:
      tracemalloc.start()
    algoTwoC = algoTwo.matrixMul(AlgoMatrixA, AlgoMatrixB)
    if args.traceMemo:
      _,HEGMM_ENPeak = tracemalloc.get_traced_memory() 
      tracemalloc.stop()
    algoTwoE = resource_usage(RUSAGE_SELF)

    if (algoTwoC == trueC).all():
      logging.info("HEGMM_EN Right")
      logging.info("%d %d %d %.2f", a,b,c
        , ((algoTwoE.ru_utime - algoTwoS.ru_utime)+(algoTwoE.ru_stime - algoTwoS.ru_stime))*1000
        )
      if args.traceMemo:
        logging.info("%d %d %d %.2fKB", a,b,c,HEGMM_ENPeak / 1024)
    else:
      print("HEGMM_EN Wrong"," ",a," ",b," ",c)
      logging.warning("HEGMM_EN Wrong")
      logging.warning("%d %d %d %.2f", a,b,c, ((algoTwoE.ru_utime - algoTwoS.ru_utime)+(algoTwoE.ru_stime - algoTwoS.ru_stime))*1000)

  if HUANG:
    algoHuangS = resource_usage(RUSAGE_SELF)
    AlgoMatrixA = matrixA.copy()
    AlgoMatrixB = matrixB.copy()
    if args.traceMemo:
      tracemalloc.start()
    HuangC = algoHuang.matrixMul(AlgoMatrixA, AlgoMatrixB)
    if args.traceMemo:
      _,HUANGPeak = tracemalloc.get_traced_memory() 
      tracemalloc.stop()
    algoHuangE = resource_usage(RUSAGE_SELF)

    if (HuangC == trueC).all():
      logging.info("Huang Right")
      # logging.info("%d %d %d %.2f", a,b,c
      #   , ((algoHuangE.ru_utime - algoHuangS.ru_utime)+(algoHuangE.ru_stime - algoHuangS.ru_stime))*1000
      #   )
      if args.traceMemo:
        logging.info("%d %d %d %.2fKB", a,b,c,HUANGPeak / 1024)
    else:
      logging.warning("Huang Wrong")
      logging.warning("%d %d %d %.2f", a,b,c, ((algoHuangE.ru_utime - algoHuangS.ru_utime)+(algoHuangE.ru_stime - algoHuangS.ru_stime))*1000)

  print("done")

if __name__ == '__main__':
    main()