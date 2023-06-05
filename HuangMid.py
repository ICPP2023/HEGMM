import numpy as np
import time
import math
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from resource import getrusage as resource_usage, RUSAGE_SELF

DEBUG = 0

class huang(object):

  def __init__(self, cipherSize):
    self.cipherSize = cipherSize
    self.HE = Pyfhel()
    self.HE.contextGen(scheme='bfv', n=cipherSize, t=65537, t_bits=20) #m is 2048 or 8192
    self.HE.keyGen()
    self.HE.relinKeyGen()
    self.HE.rotateKeyGen()

  def findRmax(self, small, big):
    fac = 2
    res = small
    while res < big:
      res = small * fac
      fac += 1

    return res

  def genMaskII(self, srcShp, dstShp, r):
    srcdim1 = srcShp[0]
    srcdim2 = srcShp[1]
    dstdim1 = dstShp[0]
    dstdim2 = dstShp[1]

    result = {}
    result[r] = ([1]*(dstdim1-r) + [0]*r) * dstdim2
    result[r-srcdim1] = ([0]*(dstdim1-r) + [1]*r) * dstdim2

    return result

  def executePermutation(self, ctMatrix, rotDict):
    subMatrix = []

    accumulateStep = 0
    for idx, item in enumerate(rotDict.items()):
      rotStep = item[0]
      mask = item[1]
      encodeMaskRot = self.HE.encode(mask)
      try:
        if idx==0:
            Atransp = self.HE.rotate(ctMatrix, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        else:
            Atransp += self.HE.rotate(ctMatrix, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        self.HE.relinearize(Atransp)
        accumulateStep += (rotStep-accumulateStep)%(self.cipherSize/2)
      except:
        # print("rotStep: ",rotStep," m: ",self.m," l: ",self.l1," n: ", self.n)
        continue
    subMatrix.append(PyCtxt(copy_ctxt=Atransp))

    return subMatrix

  def omegaTrans(self, encB):
    dicB = []

    #get omega 0
    srcShp = (self.Rmax, self.Cmax)
    dstShp = (self.Rmax, self.Cmax)

    OMask = [1]*dstShp[0]*dstShp[1]
    pOMask = self.HE.encode(OMask)
    omg0 = encB * pOMask
    dicB.append(omg0)

    #get omega i
    for c in range(self.l1):
      BzeroCopy = PyCtxt(copy_ctxt=encB)
      mask = self.genMaskII(srcShp, dstShp, c)
      subB = self.executePermutation(BzeroCopy, mask)
      dicB.extend(subB)

    return dicB

  def get_bit_val(self,byte, index):
    if byte & (1 << index):
      return 1
    else:
      return 0

  def totalSum(self, ct):
    ctbar = PyCtxt(copy_ctxt=ct)
    res = PyCtxt(copy_ctxt=ct)
    e = 1
    t = self.Cmax

    for j in range(len(bin(t))-4,-1,-1):
      res += self.HE.rotate(ctbar,-e*self.Rmax)
      e *= 2
      if self.get_bit_val(t,j) == 1:
        ctbar = PyCtxt(copy_ctxt=res)
        res = ct + self.HE.rotate(ctbar,-1*self.Rmax)
        e += 1
      ctbar = PyCtxt(copy_ctxt=res)

    return res
  
  def sigmaTrans(self, matrix):
    for rowNum, row in enumerate(matrix):
      row = np.roll(row, -rowNum)
      matrix[rowNum] = row

  def matrixMul(self, matrixA, matrixB):
    self.m = matrixA.shape[0]
    self.l1 = matrixA.shape[1]
    self.l2 = matrixB.shape[0]
    self.n = matrixB.shape[1]

    if self.l1 != self.l2:
      print('cannot do matrix multiplication, dimention does not match')
      exit()

    self.sigmaTrans(matrixA)

    # 补零
    padzeroS = time.perf_counter()
    if(self.m < self.l1):
      self.Rmax = self.l2
      self.Cmax = max(self.l1, self.n)
      matrixA = np.pad(matrixA,((0,self.Rmax-self.m),(0,self.Cmax-self.l1)))
    else:
      self.Rmax = self.findRmax(self.l2, self.m)
      self.Cmax = max(self.l1, self.n)
      matrixA = np.pad(matrixA,((0,self.Rmax-self.m),(0,self.Cmax-self.l1)))
      matrixB = np.tile(matrixB,(int(self.Rmax/self.l2),1))
    padzeroE = time.perf_counter()
    # print("pad 0 time:", (padzeroE-padzeroS)*1000)

    # 生成mask
    maskS = time.perf_counter()
    mask = np.zeros((self.l1,self.Rmax,self.Cmax))
    for k in range(self.l1):
      for i in range(self.Rmax):
        for j in range(self.Cmax):
          if j==k:
            mask[k][i][j] = 1
          else:
            mask[k][i][j] = 0
    maskE = time.perf_counter()
    # print("mask time:", (maskE-maskS)*1000)


    # CMult提取对角线
    subA = list(range(self.l1))
    for i in range(self.l1):
      subA[i] = matrixA[:,i]
      subA[i] = subA[i].copy().flatten(order='F')
      subA[i] = self.HE.encrypt(subA[i])

    flatB = matrixB.copy().flatten(order='F')
    ctB = self.HE.encrypt(flatB)

    # totalSum复制元素
    totalSumS = time.perf_counter()
    ctAsub = list(range(self.l1))
    for k in range(self.l1):
      ctAsub[k] = self.totalSum(subA[k])
    totalSumE = time.perf_counter()
    print("totalSum time:", (totalSumE-totalSumS)*1000)
    
    omegaS = time.perf_counter()
    ctBsub = self.omegaTrans(ctB)
    omegaE = time.perf_counter()
    # print("omega time:", (omegaE-omegaS)*1000)

    huangMulAddS = resource_usage(RUSAGE_SELF)
    # 乘
    ctC = 0
    for k in range(self.l1):
      ctC += ctAsub[k] * ctBsub[k+1]

    # 加和
    flatC = self.HE.decrypt(ctC)
    matrixC = np.zeros((self.Rmax,self.Cmax))
    for i in range(self.Rmax):
      for j in range(self.Cmax):
        matrixC[i][j] = flatC[j * self.Rmax + i]

    huangMulAddE = resource_usage(RUSAGE_SELF)
    # print("huang HE-Mult & HE-Add time: ", ((huangMulAddE.ru_utime - huangMulAddS.ru_utime)+(huangMulAddE.ru_stime - huangMulAddS.ru_stime))*1000)

    return matrixC[:self.m,:self.n]