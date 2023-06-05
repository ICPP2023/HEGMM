import numpy as np
import time
import math
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from resource import getrusage as resource_usage, RUSAGE_SELF

DEBUG = 0

class algo1(object):

  def __init__(self, cipherSize):
    self.cipherSize = cipherSize
    self.HE = Pyfhel()
    self.HE.contextGen(scheme='bfv', n=cipherSize, t=65537, t_bits=20) #m is 2048 or 8192
    self.HE.keyGen()
    self.HE.relinKeyGen()
    self.HE.rotateKeyGen()

  def generateDiagMatrix(self, matrix, pos='left'):
    assert pos == 'left' or pos == 'right'
    if pos == 'left':
      for rowNum, row in enumerate(matrix):
        row = np.roll(row, -rowNum)
        matrix[rowNum] = row

    elif pos == 'right':
      for colNum, col in enumerate(matrix.T):
        col = np.roll(col, -colNum)
        matrix.T[colNum] = col

  def generateUploadVector(self, matrixA, matrixB):
    Arow = matrixA.shape[0]
    Acol = matrixA.shape[1]
    Brow = matrixB.shape[0]
    Bcol = matrixB.shape[1]

    #Senario 1
    if Arow <= Brow and Acol >= Bcol:
      resA = np.zeros((Brow, Acol))
      for colNum, col in enumerate(matrixA.T):
        col = np.pad(col,(0,Brow-Arow))
        resA.T[colNum] = col
      resB = matrixB

    #Senario 2
    if Arow <= Brow and Acol <= Bcol:
      repeatTimes = int(Bcol/Acol)
      tailNumbers = Bcol%Acol

      resA = np.zeros((Brow, Bcol))

      for rowNum, row in enumerate(matrixA):
        row = np.tile(row, repeatTimes)
        row = np.append(row, row[:tailNumbers])
        resA[rowNum] = row
      resB = matrixB

    #Senario 3
    if Arow > Brow and Acol <= Bcol:
      repeatTimes = int(Arow/Brow)
      tailNumbers = Arow%Brow

      resB = np.zeros((Arow, Bcol))

      for colNum, col in enumerate(matrixB.T):
        col = np.tile(col,repeatTimes)
        col = np.append(col, col[:tailNumbers])
        resB.T[colNum] = col

      repeatTimes = int(Bcol/Acol)
      tailNumbers = Bcol%Acol

      resA = np.zeros((Arow, Bcol))

      for rowNum, row in enumerate(matrixA):
        row = np.tile(row, repeatTimes)
        row = np.append(row, row[:tailNumbers])
        resA[rowNum] = row

    #Senario 4
    if Arow > Brow and Acol > Bcol:
      repeatTimes = int(Arow/Brow)
      tailNumbers = Arow%Brow

      resB = np.zeros((max(Arow, Brow), Bcol))

      for colNum, col in enumerate(matrixB.T):
        col = np.tile(col,repeatTimes)
        col = np.append(col, col[:tailNumbers])
        resB.T[colNum] = col
      resA = matrixA

    if 'resA' not in dir():
      print('cannot get resA and B: ', Arow, ' ', Acol, ' ', Brow, ' ', Bcol)
    return resA, resB

  def generateAzeroMask(self, shapeB):
    mask = [1] * shapeB[0] * shapeB[1]
    return mask

  def generateAiMask(self, shapeA, shapeB, c):
    Arow = shapeA[0]
    Acol = shapeA[1]
    Brow = shapeB[0]
    Bcol = shapeB[1]

    result = {}

    mask = [0]*c*Arow + [1]*Bcol*Arow

  def generatePermutationMask(self, shapeA, shapeB, r):
    Arow = shapeA[0]
    Acol = shapeA[1]
    Brow = shapeB[0]
    Bcol = shapeB[1]

    repeatTimes = int(Acol/Bcol)
    if Arow > Acol:
      tailNumbers = Arow%Acol
    else: 
      tailNumbers = 0

    result = {}

    result[r] = ([1]*(Brow-r) + [0]*r) * Bcol
    result[r-Brow+tailNumbers] = ([0]*(Brow-r) + [1]*r) * Bcol

    return result

  def generateRowPermutationMask(self, shapeA, shapeAB, r):
    Arow = shapeA[0]
    Acol = shapeA[1]
    ABrow = shapeAB[0]
    ABcol = shapeAB[1]

    tailNumbers = ABcol%Acol

    result = {}

    if Acol <= ABcol:
      result[r*ABrow] = [1]*ABrow*(ABcol-r)
      result[(r-ABcol+tailNumbers)*ABrow] = [0]*ABrow*(ABcol-r) + [1]*(ABrow*r)
    else:
      if r <= Acol-ABcol:
        result[r*ABrow] = [1]*ABrow*(Acol-r)
      else:
        result[r*ABrow] = [1]*ABrow*(Acol-r)
        result[(r-Acol)*ABrow] = [0]*ABrow*(Acol-r) + [1]*(ABrow*r)

    return result

  def generateAiBi(self, encA, encB, shapeA, shapeB):
    Arow = shapeA[0]
    Acol = shapeA[1]
    Brow = shapeB[0]
    Bcol = shapeB[1]

    ArepeatTimes = int(Bcol/Acol)
    AtailNumbers = Bcol%Acol

    dicA = []
    dicB = []

    dicB.append(encB)

    Azeroshape = (max(Arow,Brow), Bcol)
    AzeroRow = Azeroshape[0]
    AzeroCol = Azeroshape[1]
    AzeroMask = self.generateAzeroMask(Azeroshape)
    pAzeroMask = self.HE.encode(AzeroMask)
    Azero = encA * pAzeroMask
    dicA.append(Azero)

    AzeroCopy = PyCtxt(copy_ctxt=Azero)

    for c in range(1, Acol):
      AzeroCopy = PyCtxt(copy_ctxt=encA)
      AcolPermutationMask = self.generateRowPermutationMask(shapeA, Azeroshape, c)
      accumulateStep = 0
      for idx, item in enumerate(AcolPermutationMask.items()):
        rotStep = item[0]
        mask = item[1]
        encodeMaskRot = self.HE.encode(mask)
        if idx==0:
            Atransp = self.HE.rotate(AzeroCopy, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        else:
            Atransp += self.HE.rotate(AzeroCopy, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        self.HE.relinearize(Atransp)
        accumulateStep += (rotStep-accumulateStep)%(self.cipherSize/2)
      dicA.append(PyCtxt(copy_ctxt=Atransp))

    for r in range(1, Acol):
      BzeroCopy = PyCtxt(copy_ctxt=encB)
      BcolPermutationMask = self.generatePermutationMask(shapeA, Azeroshape, r)
      accumulateStep = 0
      for idx, item in enumerate(BcolPermutationMask.items()):
        rotStep = item[0]
        mask = item[1]
        encodeMaskRot = self.HE.encode(mask)
        if idx==0:
            Btransp = self.HE.rotate(BzeroCopy, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        else:
            Btransp += self.HE.rotate(BzeroCopy, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        self.HE.relinearize(Btransp)
        accumulateStep += (rotStep-accumulateStep)%(self.cipherSize/2)
      dicB.append(PyCtxt(copy_ctxt=Btransp))

    return dicA, dicB

  def dupA(self, matrixA):
    Arow = matrixA.shape[0]
    Acol = matrixA.shape[1]

    resA = np.tile(matrixA,(int(Acol/Arow),1))
    resA = np.pad(resA, ((0,Acol-resA.shape[0]),(0,0)))

    return resA

  def matrixMul(self, matrixA, matrixB):
    Arow = matrixA.shape[0]
    Acol = matrixA.shape[1]
    Brow = matrixB.shape[0]
    Bcol = matrixB.shape[1]
    shapeA = matrixA.shape
    shapeB = matrixB.shape

    if Acol != Brow:
      print('cannot do matrix multiplication, dimention does not match')
      exit()

    algo1EnS = resource_usage(RUSAGE_SELF)
    # [Step 1]
    self.generateDiagMatrix(matrixA, pos='left')
    self.generateDiagMatrix(matrixB, pos='right')

    matrixA, matrixB = self.generateUploadVector(matrixA, matrixB)

    # [Step 2]
    flatA = matrixA.copy().flatten(order='F')
    flatB = matrixB.copy().flatten(order='F')

    encryptA = self.HE.encrypt(flatA)
    encryptB = self.HE.encrypt(flatB)

    algo1EnE = resource_usage(RUSAGE_SELF)
    # print("algo1 EncryptPre time: ", ((algo1EnE.ru_utime - algo1EnS.ru_utime)+(algo1EnE.ru_stime - algo1EnS.ru_stime))*1000)

    # [Step 3]
    algo1MulAddS = resource_usage(RUSAGE_SELF)
    Agroup, Bgroup = self.generateAiBi(encryptA, encryptB, shapeA, shapeB)

    algo1AiBi = resource_usage(RUSAGE_SELF)

    # print("algo1 generateAiBi time: ", ((algo1AiBi.ru_utime - algo1MulAddS.ru_utime)+(algo1AiBi.ru_stime - algo1MulAddS.ru_stime))*1000)

    algo1MulAddS = resource_usage(RUSAGE_SELF)
    for i, (ai,bi) in enumerate(zip(Agroup, Bgroup)):
      c = ai * bi
      if i==0:
          encryptC = c
      else:
          encryptC += c

    algo1MulAddE = resource_usage(RUSAGE_SELF)
    # print("algo1 HE-Mult & HE-Add time: ", ((algo1MulAddE.ru_utime - algo1MulAddS.ru_utime)+(algo1MulAddE.ru_stime - algo1MulAddS.ru_stime))*1000)

    # [Step 5]
    algo1DecryS = resource_usage(RUSAGE_SELF)
    decryptC = encryptC.decrypt()[:max(Arow,Brow)*Bcol]

    matrixC = np.zeros((max(Arow,Brow), Bcol), dtype=int)

    for i in range(Bcol):
      for j in range(max(Arow,Brow)):
        matrixC[j][i] = decryptC[i * max(Arow,Brow) + j]

    matrixC = matrixC[:Arow]
    algo1DecryE = resource_usage(RUSAGE_SELF)
    # print("algo1 Decry&Distribute time: ", ((algo1DecryE.ru_utime - algo1DecryS.ru_utime)+(algo1DecryE.ru_stime - algo1DecryS.ru_stime))*1000)

    return matrixC