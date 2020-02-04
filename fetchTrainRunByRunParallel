#! /usr/bin/env python3

from __future__ import print_function
import argparse
import hashlib
import logging
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import timeit

isjalien = shutil.which("alien.py") is not None
requesttimes = []

class AlienTool: 

  def __init__(self):
    self.__lock = threading.Lock()
  
  def md5(self, fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

  def gridmd5sum(self, gridfile):
    # Function must be robust enough to fetch all possible
    # xrd error states which it usually gets from the stdout
    # of the query process
    gbout = ''
    errorstate = True
    while errorstate:
      errorstate = False
      md5cmd = "alien.py md5sum" if isjalien else "gbbox md5sum"
      gbout = subprocess.getstatusoutput("%s %s" %(md5cmd, gridfile))[1]
      if gbout.startswith("Error") or gbout.startswith("Warning") or "CheckErrorStatus" in gbout:
        errorstate = True
    return gbout.split('\t')[0]

  def copy(self, inputfile, outputfile):
    logging.info("Copying %s to %s", inputfile, outputfile)
    self.__lock.acquire(True)
    if not os.path.exists(os.path.dirname(outputfile)):
      os.makedirs(os.path.dirname(outputfile), 0o755)
    self.__lock.release()
    if isjalien:
      subprocess.call(['alien_cp', inputfile, 'file://%s'%outputfile])
    else:
      subprocess.call(['alien_cp', 'alien://%s'%inputfile, outputfile])
    # make check
    if os.path.exists(outputfile):
      localmd5 = self.md5(outputfile)
      gridmd5 = self.gridmd5sum(inputfile)
      logging.debug("MD5local: %s, MD5grid %s", localmd5, gridmd5)
      if localmd5 != gridmd5:
        logging.error("Mismatch in MD5 sum for file %s", outputfile)
        # incorrect MD5sum, outputfile probably corrupted
        os.remove(outputfile)
        return False
      else:
        logging.info("Output file %s copied correctly", outputfile)
        return True
    else:
      logging.error("output file %s not found" %outputfile)
      return False

  def listdir(self, inputdir):
    # Function must be robust agianst error states which
    # it can only get from the stdout. As long as the 
    # request ends in error state it should retry
    errorstate = True
    lscmd = "alien.py ls" if isjalien else "alien_ls"
    while errorstate:
      start = timeit.timeit()
      dirs = subprocess.getstatusoutput("%s %s" %(lscmd, inputdir))[1]
      end = timeit.timeit()
      requesttimes.append(end-start)

      errorstate = False
      result = []
      for d in dirs.split("\n"):
        if d.startswith("Error") or d.startswith("Warning"):
          errorstate = True
          break
        mydir = d.rstrip().lstrip()
        if mydir.endswith("/"):
          mydir = mydir.rstrip("/")
        if len(mydir):
          result.append(mydir)
      if errorstate:
        continue
      return result


class Filepair:
  
  def __init__(self, source, target, ntrials = 0):
    self.__source = source
    self.__target = target
    self.__ntrials = ntrials

  def setntrials(self, ntrials):
    self.__ntrials = ntrials

  def getntrials(self):
    return self.__ntrials

  def source(self):
    return self.__source

  def target(self):
    return self.__target

class DataPool :
  
  def __init__(self):
    self.__data = []
    self.__lock = threading.Lock()

  def insert_pool(self, filepair):
    self.__lock.acquire(True)
    logging.info("Adding to pool: %s - %s", filepair.source(), filepair.target())
    self.__data.append(filepair)
    self.__lock.release()

  def getpoolsize(self):
    return len(self.__data)

  def pop(self):
    result = None
    self.__lock.acquire(True)
    if(self.getpoolsize()):
      result = self.__data.pop(0)
    self.__lock.release()
    return result

class CopyHandler(threading.Thread):
  
  def __init__(self):
    threading.Thread.__init__(self)
    self.__datapool = None
    self.__poolfiller = None
    self.__alienhelper = None
    self.__maxtrials = 5
  
  def setalienhelper(self, alienhelper):
    self.__alienhelper = alienhelper
  
  def setdatapool(self, datapool):
    self.__datapool = datapool

  def setpoolfiller(self, poolfiller):
    self.__poolfiller = poolfiller

  def setmaxtrials(self, maxtrials):
    self.__maxtrials = maxtrials

  def waitforwork(self):
    if self.__datapool.getpoolsize():
      return
    if not self.__poolfiller.isactive():
      return
    while not self.__datapool.getpoolsize():
      if not self.__poolfiller.isactive():
        break
      time.sleep(5)

  def run(self):
    hasWork = True
    while hasWork:
      self.waitforwork()
      nextfile = self.__datapool.pop()
      if nextfile:
        copystatus = self.__alienhelper.copy(nextfile.source(), nextfile.target())
        if not copystatus:
          # put file back on the pool in case of copy failure
          # only allow for amaximum amount of copy trials
          trials = nextfile.getntrials()
          trials += 1
          if trials >= self.__maxtrials:
            logging.error("File %s failed copying in %d trials - giving up", nextfile.source(), self.__maxtrials)
          else:
            logging.error("File %s failed copying (%d/%d) - re-inserting into the pool ...", nextfile.source(), trials, self.__maxtrials)
            nextfile.setntrials(trials)
            self.__datapool.insert_pool(nextfile)
      if not self.__poolfiller.isactive():
        # if pool is empty exit, else keep thread alive for remaining files
        if not self.__datapool.getpoolsize():
          hasWork = False

class PoolFiller(threading.Thread):
  
  def __init__(self, outputdir, trainrun, legotrain, dataset, recpass, aodprod, filename, poolsize):
    threading.Thread.__init__(self)
    self.__datapool = None
    self.__alientool = None
    self.__outputdir =  outputdir
    self.__trainrun = trainrun
    self.__legotrain = legotrain
    self.__dataset = dataset
    self.__recpass = recpass
    self.__aodprod = aodprod
    self.__filename = filename
    self.__maxpoolsize = poolsize
    self.__active = False

  def setdatapool(self, datapool):
    self.__datapool = datapool

  def setalientool(self, alientool):
    self.__alientool = alientool

  def isactive(self):
    return self.__active

  def run(self):
    self.__active = True
    self.__find_trainfiles()
    logging.info("Finding train files done")
    self.__active = False

  def __extractYear(self, dataset):
    if not dataset.startswith("LHC"):
      return 0
    return int(dataset[3:5]) + 2000 
  
  def __isdata(self, dataset):
    if len(dataset) == 6:
      return True
    return False

  def __extractTrainID(self, idstring):
    trainid = idstring.split("_")[0]
    if trainid.isdigit():
      return int(trainid)
    return 0

  def __find_trainfiles(self):
    datatag = "sim"
    isdata = self.__isdata(self.__dataset)
    if isdata:
      datatag = "data"
    gridbase = "/alice/%s/%d/%s" %(datatag, self.__extractYear(self.__dataset), self.__dataset)
    logging.info("Searching output files in train directory %s", gridbase)
    for r in self.__alientool.listdir(gridbase):
      logging.info("Checking run %s" %r)
      if not r.isdigit():
        continue
      rundir = os.path.join(gridbase, r)
      if isdata:
        rundir = os.path.join(rundir, self.__recpass)
      if self.__aodprod:
        rundir = os.path.join(rundir, self.__aodprod)
      runoutputdir = os.path.join(self.__outputdir, r)
      data = self.__alientool.listdir(rundir)
      if not self.__legotrain.split("/")[0] in data:
        logging.error("PWG dir not found four run %s" %r)
        continue
      legotrainsdir = os.path.join(rundir, self.__legotrain.split("/")[0])
      legotrains = self.__alientool.listdir(legotrainsdir)
      if not self.__legotrain.split("/")[1] in legotrains:
        logging.error("Train %s not found in pwg dir for run %s" %(self.__legotrain.split("/")[1], r))
        continue
      trainbase = os.path.join(rundir, self.__legotrain)
      trainruns = self.__alientool.listdir(trainbase)
      traindir = [x for x in trainruns if self.__extractTrainID(x) == self.__trainrun]
      if not len(traindir):
        logging.error("Train run %d not found for run %s" %(self.__trainrun, r))
        continue
      fulltraindir = os.path.join(trainbase, traindir[0])
      trainfiles = self.__alientool.listdir(fulltraindir)
      if not self.__filename in trainfiles:
        logging.info("Train directory %s doesn't contain %s", fulltraindir, self.__filename)
      else:
        inputfile = os.path.join(fulltraindir, self.__filename)
        outputfile = os.path.join(runoutputdir, self.__filename)
        if os.path.exists(outputfile):
          logging.info("Output file %s already found - not copying again", outputfile)
        else:
          logging.info("Copying %s to %s", inputfile, outputfile)
          self.__wait()
          self.__datapool.insert_pool(Filepair(inputfile, outputfile))

  def __wait(self):
    # wait until pool is half empty
    if self.__datapool.getpoolsize() < self.__maxpoolsize:
      return
    # pool full, wait until half empty
    emptylimit = self.__maxpoolsize/2
    while self.__datapool.getpoolsize() > emptylimit:
      time.sleep(5)

def fetchtrainparallel(outputpath, trainrun, legotrain, dataset, recpass, aodprod, filename):
  if isjalien:
    logging.info("Using JAliEn ...")
  else:
    logging.info("Using legacy alien ...")
  datapool = DataPool()
  alienhelper = AlienTool()
  logging.info("Checking dataset %s for train with ID %d (%s)", dataset, trainrun, legotrain)

  poolfiller = PoolFiller(outputpath, trainrun, legotrain, dataset, recpass, aodprod if len(aodprod) > 0 else None, filename, 1000)
  poolfiller.setdatapool(datapool)
  poolfiller.setalientool(alienhelper)
  poolfiller.start()

  workers = []
  # use 4 threads in order to keep number of network request at an acceptable level
  for i in range(0, 4):
    worker = CopyHandler()
    worker.setdatapool(datapool)
    worker.setpoolfiller(poolfiller)
    worker.setalienhelper(alienhelper)
    worker.start()
    workers.append(worker)

  poolfiller.join()
  for worker in workers:
    worker.join()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
			prog="fetchTrainRunByRunParallel", 
			description="Tool to get runwise train output"
			)
  parser.add_argument("outputpath", metavar="OUTPUTPATH", help="Path where to store the output files run-by-run")
  parser.add_argument("trainrun", metavar="TRAINRUN", type=int, help="ID of the train run (number is sufficient, time stamp not necessary)")
  parser.add_argument("legotrain", metavar="LEGOTRAIN", help="Name of the lego train (i.e. PWGJE/Jets_EMC_pPb)")
  parser.add_argument("dataset", metavar="DATASET", help="Name of the dataset")
  parser.add_argument("-p", "--recpass", type=str, default="pass1", help="Reconstruction pass (only meaningfull in case of data) [default: pass1]")
  parser.add_argument("-a", "--aod",  type=str, default="", help="Dedicated AOD production (if requested) [default: not set]")
  parser.add_argument("-f", "--filename", type=str, default="AnalysisResults.root", help="File to copy from grid [default: AnalysisResults.root]")
  args = parser.parse_args()
  logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
  fetchtrainparallel(args.outputpath, args.trainrun, args.legotrain, args.dataset, args.recpass, args.aod, args.filename)

  #time statistics
  maxtime = max(requesttimes)
  meantime = statistics.mean(requesttimes)
  rmstime = statistics.stdev(requesttimes)
  logging.info("Stats ls requests: Max {} s, Mean {} s, RMS {}, estimated from {} tests".format(maxtime, meantime, rmstime, len(requesttimes)))
