import numpy as np
import time
import math
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from resource import getrusage as resource_usage, RUSAGE_SELF

DEBUG = 0

class algo2(object):

  def __init__(self, cipherSize):
    self.cipherSize = cipherSize
    self.HE = Pyfhel()
    self.HE.contextGen(scheme='bfv', n=cipherSize, t=65537, t_bits=20) #m is 2048 or 8192
    self.HE.keyGen()
    self.HE.relinKeyGen()
    self.HE.rotateKeyGen()

  def dupA(self, matrixA):
    resA = np.tile(matrixA, (math.ceil(self.l1/self.m),1))
    return resA

  def dupB(self, matrixB):
    resB = np.tile(matrixB,(1,math.ceil(self.l2/self.n)))
    return resB

  def padA(self, matrixA):
    if self.dupState == 'N':
      if self.m <= self.l2:
        resA = np.zeros((self.l2, self.l1))
        for colNum, col in enumerate(matrixA.T):
          col = np.pad(col,(0,self.l2-self.m))
          resA.T[colNum] = col
      else:
        resA = matrixA

    return resA

  def sigmaTrans(self, matrix):
    for rowNum, row in enumerate(matrix):
      row = np.roll(row, -rowNum)
      matrix[rowNum] = row

  def tauTrans(self, matrix):
    for colNum, col in enumerate(matrix.T):
      col = np.roll(col, -colNum)
      matrix.T[colNum] = col

  def epsilon0Trans(self, matrixA, shapeB):
    m = matrixA.shape[0]
    l1 = matrixA.shape[1]
    l2 = self.shpBduped[0]
    n = self.shpBduped[1]

    if l1 >= n:
      resA = matrixA
    else:
      repeatTimes = int(n/l1)
      tailNumbers = n%l1
      resA = np.zeros((m, n))

      for rowNum, row in enumerate(matrixA):
        row = np.tile(row, repeatTimes)
        row = np.append(row, row[:tailNumbers])
        resA[rowNum] = row

    return resA

  def omega0Trans(self, matrixB, shapeA):
    m = self.shpAduped[0]
    l1 = self.shpAduped[1]
    l2 = matrixB.shape[0]
    n = matrixB.shape[1]

    if m <= l2:
      resB = matrixB
    else:
      repeatTimes = int(m/l2)
      tailNumbers = m%l2

      resB = np.zeros((m, n))

      for colNum, col in enumerate(matrixB.T):
        col = np.tile(col,repeatTimes)
        col = np.append(col, col[:tailNumbers])
        resB.T[colNum] = col

    return resB

  def executePermutation(self, ctMatrix, rotDict):
    subMatrix = []

    accumulateStep = 0
    for idx, item in enumerate(rotDict.items()):
      rotStep = item[0]
      mask = item[1]
      encodeMaskRot = self.HE.encode(mask)
      # try:
      if idx==0:
          Atransp = self.HE.rotate(ctMatrix, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
      else:
          Atransp += self.HE.rotate(ctMatrix, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
      self.HE.relinearize(Atransp)
      accumulateStep += (rotStep-accumulateStep)%(self.cipherSize/2)
      # except:
      #   print("rotStep: ",rotStep," m: ",self.m," l: ",self.l1," n: ", self.n)
      #   continue
    subMatrix.append(PyCtxt(copy_ctxt=Atransp))

    return subMatrix

  def genMaskI(self, srcShp, dstShp, r):
    srcdim1 = srcShp[0]
    srcdim2 = srcShp[1]
    dstdim1 = dstShp[0]
    dstdim2 = dstShp[1]

    result = {}
    # result[r*dstdim1] = [1]*min(dstdim1*(srcdim2-r), dstdim1*dstdim2)
    # result[dstdim1*(r-srcdim2)] = [0]*dstdim1*(srcdim2-r) + [1]*dstdim1*(dstdim2+r-srcdim2)
    if r*dstdim1 + dstdim1*dstdim2 <= srcdim1*srcdim2:
      result[r*dstdim1] = [1]*dstdim1*dstdim2
    else:
      result[r*dstdim1] = [1]*dstdim1*(srcdim2-r)
      result[dstdim1*(r-srcdim2)] = [0]*dstdim1*(srcdim2-r) + [1]*dstdim1*(dstdim2+r-srcdim2)

    return result

  def genMaskII(self, srcShp, dstShp, r):
    srcdim1 = srcShp[0]
    srcdim2 = srcShp[1]
    dstdim1 = dstShp[0]
    dstdim2 = dstShp[1]

    result = {}
    result[r] = ([1]*(dstdim1-r) + [0]*r) * dstdim2
    result[r-srcdim1] = ([0]*(dstdim1-r) + [1]*r) * dstdim2

    return result

  def epsilonTrans(self, encA):
    dicA = []

    #get epsilon 0
    srcShp = self.shpAduped
    dstShp = (max(self.m,self.shpAduped[0]), max(self.n,self.shpBduped[1]))
    if self.dupState == 'B':
        srcShp = srcShp[::-1]
        dstShp = dstShp[::-1]
    OMask = [1]*dstShp[0]*dstShp[1]
    pOMask = self.HE.encode(OMask)
    eps0 = encA * pOMask
    dicA.append(eps0)

    #get epsilon i
    for c in range(1, self.loop):
      AzeroCopy = PyCtxt(copy_ctxt=encA)
      if self.dupState == 'B':
        mask = self.genMaskII(srcShp, dstShp, c)
      else:
        mask = self.genMaskI(srcShp, dstShp, c)
      subA = self.executePermutation(AzeroCopy, mask)
      dicA.extend(subA)

    return dicA

  def omegaTrans(self, encB):
    dicB = []

    #get omega 0
    srcShp = self.shpBduped
    dstShp = (max(self.m,self.shpAduped[0]), max(self.n,self.shpBduped[1]))
    if self.dupState == 'B':
        srcShp = srcShp[::-1]
        dstShp = dstShp[::-1]
    OMask = [1]*dstShp[0]*dstShp[1]
    pOMask = self.HE.encode(OMask)
    omg0 = encB * pOMask
    dicB.append(omg0)

    #get omega i
    for c in range(1, self.loop):
      BzeroCopy = PyCtxt(copy_ctxt=encB)
      if self.dupState == 'B':
        mask = self.genMaskI(srcShp, dstShp, c)
      else:
        mask = self.genMaskII(srcShp, dstShp, c)
      subB = self.executePermutation(BzeroCopy, mask)
      dicB.extend(subB)

    return dicB

  def shiftAdd(self, ctMa, ctMb):
    if self.dupState == 'A':
      odim1 = self.m
      odim2 = self.l1
      ddim2 = self.n
      shftAddShp = (self.shpAduped[0], self.n)
      loop = math.ceil(odim2/odim1)
    elif self.dupState == 'B':
      odim1 = self.n
      odim2 = self.l2
      ddim2 = self.m
      shftAddShp = (self.shpBduped[1], self.m)
      loop = math.ceil(odim2/odim1)
    else:
      odim1 = max(self.m,self.n,self.l1)+1
      odim2 = self.l1
      loop = 0

    algo2MultS = resource_usage(RUSAGE_SELF)
    for i, (ai,bi) in enumerate(zip(ctMa, ctMb)):
      c = ai * bi
      if i==0:
        encryptC = c
      elif i < odim2%odim1 or odim2%odim1==0:
        encryptC += c
      else:
        mask = ([1]*odim1*(int(odim2/odim1)) + [0]*odim1)*ddim2
        encodeMask = self.HE.encode(mask)
        ctransp = c * encodeMask
        encryptC += ctransp
    algo2MultE = resource_usage(RUSAGE_SELF)
    # print("algo2 mult time: ", ((algo2MultE.ru_utime - algo2MultS.ru_utime)+(algo2MultE.ru_stime - algo2MultS.ru_stime))*1000)
    

    self.HE.relinearize(encryptC)
    algo2AddS = resource_usage(RUSAGE_SELF)
    encryptResult = PyCtxt(copy_ctxt=encryptC)
    for j in range(1, loop):
      encryptCRot = PyCtxt(copy_ctxt=encryptC)
      mask = self.genMaskII(shftAddShp, shftAddShp, j*odim1)
      encryptCtransp = self.executePermutation(encryptCRot, mask)
      encryptResult += encryptCtransp[0]
    encryptC = encryptResult
    algo2AddE = resource_usage(RUSAGE_SELF)
    # print("algo2 rot-add time: ", ((algo2AddE.ru_utime - algo2AddS.ru_utime)+(algo2AddE.ru_stime - algo2AddS.ru_stime))*1000)
    

    return encryptC

  def check(self):
    if min(self.m, self.l1, self.n) == self.m:
      if max(self.m, self.l1, self.n) * self.m * math.ceil(self.l1/self.m) <= self.cipherSize/2:
        return True
    elif min(self.m, self.l1, self.n) == self.n:
      if max(self.m, self.l1, self.n) * self.n * math.ceil(self.l2/self.n) <= self.cipherSize/2:
        return True

    return False
  
  def matrixMul(self, matrixA, matrixB):
    self.m = matrixA.shape[0]
    self.l1 = matrixA.shape[1]
    self.l2 = matrixB.shape[0]
    self.n = matrixB.shape[1]

    if self.l1 != self.l2:
      print('cannot do matrix multiplication, dimention does not match')
      exit()

    algo2EnS = resource_usage(RUSAGE_SELF)
    if min(self.m, self.l1, self.n) == self.m and self.check():
      self.dupState = 'A' #dup A
      self.loop = self.m
      matrixA = self.dupA(matrixA)
    elif min(self.m, self.l1, self.n) == self.n and self.check():
      self.dupState = 'B' #dup B
      self.loop = self.n
      matrixB = self.dupB(matrixB)
    else:
      self.dupState = 'N' #dup none
      self.loop = self.l1
      matrixA = self.padA(matrixA)
      

    self.shpAduped = matrixA.shape
    self.shpBduped = matrixB.shape

    self.sigmaTrans(matrixA)
    self.tauTrans(matrixB)

    matrixA = self.epsilon0Trans(matrixA, self.shpBduped)
    matrixB = self.omega0Trans(matrixB, self.shpAduped)

    if self.dupState == 'B':
      flatA = matrixA.copy().flatten(order='C')
      flatB = matrixB.copy().flatten(order='C')
    else:
      flatA = matrixA.copy().flatten(order='F')
      flatB = matrixB.copy().flatten(order='F')

    encryptA = self.HE.encrypt(flatA)
    encryptB = self.HE.encrypt(flatB)

    algo2EnE = resource_usage(RUSAGE_SELF)
    # print("algo2 EncryptPre time: ", ((algo2EnE.ru_utime - algo2EnS.ru_utime)+(algo2EnE.ru_stime - algo2EnS.ru_stime))*1000)

    algo2MulAddS = resource_usage(RUSAGE_SELF)
    Ai = self.epsilonTrans(encryptA)
    Bi = self.omegaTrans(encryptB)
    algo2AiBi = resource_usage(RUSAGE_SELF)
    # print("algo2 generate AiBi time: ", ((algo2AiBi.ru_utime - algo2MulAddS.ru_utime)+(algo2AiBi.ru_stime - algo2MulAddS.ru_stime))*1000)

    algo2MulAddS = resource_usage(RUSAGE_SELF)
    encryptC = self.shiftAdd(Ai, Bi)
    
    algo2MulAddE = resource_usage(RUSAGE_SELF)
    # print("algo2 HE-Mult & HE-Add time: ", ((algo2MulAddE.ru_utime - algo2MulAddS.ru_utime)+(algo2MulAddE.ru_stime - algo2MulAddS.ru_stime))*1000)

    algo2DecryS = resource_usage(RUSAGE_SELF)
    decryptC = encryptC.decrypt()[:self.shpAduped[0]*self.shpBduped[1]]

    matrixC = np.zeros((self.m, self.n), dtype=int)

    if self.dupState == 'A':
      for i in range(self.n):
        for j in range(self.m):
          matrixC[j][i] = decryptC[i * self.shpAduped[0] + j]
    elif self.dupState == 'B':
      for i in range(self.m):
        for j in range(self.n):
          matrixC[i][j] = decryptC[i * self.shpBduped[1] + j]
    else:
      for i in range(self.n):
        for j in range(self.m):
          matrixC[j][i] = decryptC[i * self.shpAduped[0] + j]

    algo2DecryE = resource_usage(RUSAGE_SELF)
    # print("algo2 Decry&Distribute time: ", ((algo2DecryE.ru_utime - algo2DecryS.ru_utime)+(algo2DecryE.ru_stime - algo2DecryS.ru_stime))*1000)

    return matrixC