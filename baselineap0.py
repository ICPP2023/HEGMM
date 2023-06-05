import numpy as np
import time
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from resource import getrusage as resource_usage, RUSAGE_SELF

DEBUG = 0

class baselineap0(object):

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
    vecA = matrixA.copy().flatten(order='F')
    vecB = matrixB.copy().flatten(order='F')

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

        # if(len(deletedRow) > len(deletedCol)):
        #     deleteRowCol = deletedCol
        # else:
        #     deleteRowCol = deletedRow

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

  def permuteVector(self, encryVec, dim, matrix='A'):
    result = []
    permuteMask = self.generatePermuteMask(dim)

    ABzero = PyCtxt(copy_ctxt=encryVec)
    encryVecRot = PyCtxt(copy_ctxt=encryVec)
    result.append(ABzero)

    if matrix == 'A':
      Azeroshape = (dim, dim)
      for r in range(1, dim):
        AzeroCopy = PyCtxt(copy_ctxt=encryVec)
        AcolPermutationMask = self.generateRowPermutationMask(dim, r)
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
      Bzeroshape = (dim, dim)
      for r in range(1, dim):
        BzeroCopy = PyCtxt(copy_ctxt=encryVec)
        BcolPermutationMask = self.generateColPermutationMask(dim, r)
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
    matrixA, matrixB = self.preprocess(matrixA, matrixB)

    E2DM_SEnS = resource_usage(RUSAGE_SELF)
    self.generateDiagMatrix(matrixA, pos='left')
    self.generateDiagMatrix(matrixB, pos='right')

    encryA, encryB = self.encryMatrix(matrixA, matrixB)

    E2DM_SEnE = resource_usage(RUSAGE_SELF)
    # print("E2DM_S EncryptPre time: ", ((E2DM_SEnE.ru_utime - E2DM_SEnS.ru_utime)+(E2DM_SEnE.ru_stime - E2DM_SEnS.ru_stime))*1000)

    E2DM_SMulAddS = resource_usage(RUSAGE_SELF)
    dim = matrixA.shape[0]
    groupA = self.permuteVector(encryA, dim)
    groupB = self.permuteVector(encryB, dim, matrix='B')

    muladdS = time.perf_counter()
    for idx, (vecA, vecB) in enumerate(zip(groupA, groupB)):
        if idx == 0:
            resC = vecA * vecB
        else:
            resC += vecA * vecB
    E2DM_SMulAddE = resource_usage(RUSAGE_SELF)
    # print("E2DM_S HE-Mult & HE-Add time: ", ((E2DM_SMulAddE.ru_utime - E2DM_SMulAddS.ru_utime)+(E2DM_SMulAddE.ru_stime - E2DM_SMulAddS.ru_stime))*1000)


    E2DM_SDecryS = resource_usage(RUSAGE_SELF)
    decryptC = self.HE.decrypt(resC)

    matrixC = np.zeros((matrixA.shape[0], matrixB.shape[1]), dtype=int)

    for i in range(dim):
        for j in range(dim):
            matrixC[i][j] = decryptC[j * matrixB.shape[1] + i]

    matrixC = matrixC[:Arow, :Bcol]
    E2DM_SDecryE = resource_usage(RUSAGE_SELF)
    # print("E2DM_S Decry&Distribute time: ", ((E2DM_SDecryE.ru_utime - E2DM_SDecryS.ru_utime)+(E2DM_SDecryE.ru_stime - E2DM_SDecryS.ru_stime))*1000)

    return matrixC