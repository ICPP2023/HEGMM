import numpy as np
import time
import math
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from resource import getrusage as resource_usage, RUSAGE_SELF

DEBUG = 0

class baselinelog(object):

  def __init__(self, cipherSize):
    self.cipherSize = cipherSize
    self.HE = Pyfhel()
    self.HE.contextGen(scheme='bfv', n=cipherSize, t=65537, t_bits=20) #m is 2048 or 8192
    self.HE.keyGen()
    self.HE.relinKeyGen()
    self.HE.rotateKeyGen()

  def decryptMatrix(self, batch_matrix):
    result = np.empty(batch_matrix.shape, dtype=int)
    for i in range(batch_matrix.shape[0]):
        for j in range(batch_matrix.shape[1]):
            result[i][j] = self.HE.decryptFrac(batch_matrix[i][j])

    return result

  def encryMatrix(self, matrixA, matrixB):
    vecA = matrixA.copy().flatten()
    vecB = matrixB.copy().flatten()

    encryA = self.HE.encrypt(vecA)
    encryB = self.HE.encrypt(vecB)

    return encryA, encryB

  def deleteRowCol(self, matrixA, matrixB, infoA, infoB, type):
    if type == "rc":# delete rows in matrixA and columns in matrixB
        deletedRow = []
        deletedCol = []
        for r, row in enumerate(matrixA):
            if np.all(row==0):
                deletedRow.append(r)

        for c, col in enumerate(np.transpose(matrixB)):
            if np.all(col==0):
                deletedCol.append(c)

        convertInfoA = np.delete(infoA, deletedRow)
        convertInfoB = np.delete(infoB, deletedCol)

        Aprime = np.delete(matrixA, deletedRow, axis=0)
        Bprime = np.delete(matrixB, deletedCol, axis=1)
    elif type == "cr":# delete columns in matrixA and columns in matrixB
        deletedRow = []
        for r, row in enumerate(matrixB):
            if np.all(row==0):
                deletedRow.append(r)

        convInfoA = np.delete(infoA, deletedRow)
        convInfoB = np.delete(infoB, deletedRow)

        Atiled = np.delete(matrixA, deletedRow, axis=1)
        Btiled = np.delete(matrixB, deletedRow, axis=0)

        deletedCol = []
        for c, col in enumerate(np.transpose(Atiled)):
            if np.all(col==0):
                deletedCol.append(c)

        convertInfoA = np.delete(convInfoA, deletedCol)
        convertInfoB = np.delete(convInfoB, deletedCol)

        Aprime = np.delete(Atiled, deletedCol, axis=1)
        Bprime = np.delete(Btiled, deletedCol, axis=0)

    return Aprime, Bprime, convertInfoA, convertInfoB

  def generateMask(self, dim):
    result = {}
    for i in range(dim):
        maskModel = [0] * dim * dim
        for j in range(i*dim, (i+1)*dim-i):
            maskModel[j] = 1
        result[i] = maskModel

    for i in range(-dim+1, 0):
        maskModel = [0] * dim * dim
        for j in range(i*dim-i, (i+1)*dim):
            maskModel[j] = 1
        result[i] = maskModel

    return result

  def generatePermuteMask(self, dim):
    result = []
    tempDict = {}
    for i in range(1,dim):
        maskOne = [1] * dim * dim
        maskZero = [0] * dim * dim
        for k in range(1,i+1):
            for j in range(dim-k, len(maskOne), dim):
                maskOne[j] = 0
                maskZero[j] = 1
        tempDict[1] = maskOne.copy()
        tempDict[-1] = maskZero.copy()
        result.append(tempDict.copy())

    return result

  def generateColPermuteMask(self, dim):
    tempDict = {}
    for i in range(0,dim):
        maskOne = [0] * dim * dim
        maskmOne = [0] * dim * dim
        for j in range(i,dim*dim,dim+1):
            maskOne[j] = 1
        for k in range(i*dim,dim*dim,dim+1):
            maskmOne[k] = 1
        tempDict[i*(dim-1)] = maskOne.copy()
        tempDict[-i*(dim-1)] = maskmOne.copy()

    return tempDict

  def generateColPermutationMask(self, dim, r):
    result = {}

    result[r] = ([1]*(dim-r) + [0]*r) * dim
    result[r-dim] = ([0]*(dim-r) + [1]*r) * dim

    return result

  def generateRowPermutationMask(self, dim, r):
    result = {}

    result[r*dim] = [1]*(dim*(dim-r))
    result[-dim*(dim-r)] = [0]*(dim*(dim-r)) + [1]*(dim*r)

    return result

  def permuteVector(self, encryVec, dims, matrix='A'):
    row = dims[0]
    col = dims[1]

    result = []

    ABzero = PyCtxt(copy_ctxt=encryVec)
    encryVecRot = PyCtxt(copy_ctxt=encryVec)
    result.append(ABzero)

    if matrix == 'A':
      Azeroshape = (col, col)
      for r in range(1, row):
        AzeroCopy = PyCtxt(copy_ctxt=encryVec)
        AcolPermutationMask = self.generateColPermutationMask(col, r)
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
        result.append(PyCtxt(copy_ctxt=Atransp))

    if matrix == 'B':
      Bzeroshape = (col, col)
      for r in range(1, row):
        BzeroCopy = PyCtxt(copy_ctxt=encryVec)
        BcolPermutationMask = self.generateRowPermutationMask(col, r)
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
        result.append(PyCtxt(copy_ctxt=Btransp))

    return result

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

  def preprocess(self, matrixA, matrixB):
    a = matrixA.shape[0]
    b1 = matrixA.shape[1]
    b2 = matrixB.shape[0]
    c = matrixB.shape[1]

    if b1 != b2:
      print('cannot do matrix multiplication, dimention does not match')
      exit()

    maxDim = max(max(a, b1),c)

    resA = np.pad(matrixA, ((0,maxDim-a),(0,maxDim-b1)))
    resB = np.pad(matrixB, ((0,maxDim-b2),(0,maxDim-c)))

    return resA, resB

  def matrixMul(self, matrixA, matrixB):
    Arow = matrixA.shape[0]
    Acol = matrixA.shape[1]
    Brow = matrixB.shape[0]
    Bcol = matrixB.shape[1]
    shapeA = matrixA.shape
    shapeB = matrixB.shape
    matrixA = np.tile(matrixA, (int(Acol/Arow),1))
    matrixA, matrixB = self.preprocess(matrixA, matrixB)

    E2DM_REnS = resource_usage(RUSAGE_SELF)
    diagS = time.perf_counter()
    self.generateDiagMatrix(matrixA, pos='left')
    self.generateDiagMatrix(matrixB, pos='right')
    diagE = time.perf_counter()
    # print("baseline diag time: ", (diagE - diagS)*1000)

    encryA, encryB = self.encryMatrix(matrixA, matrixB)

    E2DM_REnE = resource_usage(RUSAGE_SELF)
    # print("E2DM_R EncryptPre time: ", ((E2DM_REnE.ru_utime - E2DM_REnS.ru_utime)+(E2DM_REnE.ru_stime - E2DM_REnS.ru_stime))*1000)

    
    Adims = (Arow, Acol)
    Bdims = (Brow, Bcol)

    generateAiBiS = resource_usage(RUSAGE_SELF)
    groupA = self.permuteVector(encryA, Adims)
    groupB = self.permuteVector(encryB, Adims, matrix='B')
    generateAiBiE = resource_usage(RUSAGE_SELF)
    # print("E2DM_R generateAiBi time: ", ((generateAiBiE.ru_utime - generateAiBiS.ru_utime)+(generateAiBiE.ru_stime - generateAiBiS.ru_stime))*1000)

    E2DM_RMulAddS = resource_usage(RUSAGE_SELF)
    for idx, (vecA, vecB) in enumerate(zip(groupA, groupB)):
        if idx == 0:
            resC = vecA * vecB
        else:
            resC += vecA * vecB

    self.HE.relinearize(resC)
    encryptResult = PyCtxt(copy_ctxt=resC)
    Azeroshape = (max(Arow,Brow), Bcol)
    for j in range(1,int(Acol/Arow)):
      encryptCRot = PyCtxt(copy_ctxt=resC)
      encryptCMask = self.generateRowPermutationMask(Acol, j*Arow)
      accumulateStep = 0
      for idx, item in enumerate(encryptCMask.items()):
        rotStep = item[0]
        mask = item[1]
        encodeMaskRot = self.HE.encode(mask)
        if idx==0:
            encryptCtransp = self.HE.rotate(encryptCRot, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        else:
            encryptCtransp += self.HE.rotate(encryptCRot, (rotStep-accumulateStep)%(self.cipherSize/2)) * encodeMaskRot
        self.HE.relinearize(encryptCtransp)
        accumulateStep += (rotStep-accumulateStep)%(self.cipherSize/2)

      encryptResult += encryptCtransp

    resC = encryptResult
    E2DM_RMulAddE = resource_usage(RUSAGE_SELF)
    # print("E2DM_R HE-Mult & HE-Add time: ", ((E2DM_RMulAddE.ru_utime - E2DM_RMulAddS.ru_utime)+(E2DM_RMulAddE.ru_stime - E2DM_RMulAddS.ru_stime))*1000)

    E2DM_RDecryS = resource_usage(RUSAGE_SELF)
    decryptC = self.HE.decrypt(resC)

    matrixC = np.zeros((Arow, Bcol), dtype=int)

    for i in range(Arow):
        for j in range(Bcol):
            matrixC[i][j] = decryptC[i * Bcol + j]

    matrixC = matrixC[:Arow, :Bcol]
    E2DM_RDecryE = resource_usage(RUSAGE_SELF)
    # print("E2DM_R Decry&Distribute time: ", ((E2DM_RDecryE.ru_utime - E2DM_RDecryS.ru_utime)+(E2DM_RDecryE.ru_stime - E2DM_RDecryS.ru_stime))*1000)

    return matrixC