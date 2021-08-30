# Class to hold information about raw data from Siemens MRI scanners
# Based on mapVBVD Matlab code
#
# Author: B. Marty - 09/2018
#
#

import numpy as np
import struct


class TwixObject:

    # Class constructor
    def __init__(self, arg, dataType, fname, version):

        try:
            dataType
        except NameError:
            dataType = "image"

        self.dataType = dataType
        self.filename = fname
        self.softwareVersion = version

        ##self.IsReflected      = logical([])
        ##self.IsRawDataCorrect = logical([]) ##SRY
        self.NAcq = 0
        self.allocSize = 4096
        self.currentAlloc = 0
        self.arg = dict()
        self.freadInfo = dict()
        self.setDefaultFlags()

        self.FlagRemoveOS = False
        self.FlagDoAverage = False
        self.FlagAverageReps = False
        self.FlagAverageSets = False
        self.FlagIgnoreSeg = False

        self.FlagSkipToFirstLine = False
        self.FlagDoRawDataCorrect = False
        self.RawDataCorrectionFactors = False

        self.NCol = []
        self.NCha = []
        self.Lin = []
        self.Par = []
        self.Sli = []
        self.Ave = []
        self.Phs = []
        self.Eco = []
        self.Rep = []
        self.Set = []
        self.Seg = []
        self.Ida = []
        self.Idb = []
        self.Idc = []
        self.Idd = []
        self.Ide = []
        self.centerCol = []
        self.centerLin = []
        self.centerPar = []
        self.timestamp = []
        self.pmutime = []
        self.IsReflected = []
        self.IsRawDataCorrect = []
        self.memPos = []

        if not isinstance(arg, dict):
            raise ValueError("arg is not a dict")
        
        self.arg = arg

        if self.softwareVersion == "vb":
            self.freadInfo["szScanHeader"] = 0  # [bytes]
            self.freadInfo["szChannelHeader"] = 128  # [bytes]
            self.freadInfo["iceParamSz"] = 4

        elif self.softwareVersion == "vd":
            if self.arg.get("doRawDataCorrect", None):
                print("raw data correction for VD not supported/tested yet")
            self.freadInfo["szScanHeader"] = 192  # [bytes]
            self.freadInfo["szChannelHeader"] = 32  # [bytes]
            self.freadInfo["iceParamSz"] = 24  # vd version supports up to 24 ice params

        else:
            print("software version not supported")

    def readMDH(self, mdh, filePos):

        cLin = mdh["sLC"][0] + 1
        cPar = mdh["sLC"][3] + 1
        cSli = mdh["sLC"][2] + 1
        cAve = mdh["sLC"][1] + 1
        cPhs = mdh["sLC"][5] + 1
        cEco = mdh["sLC"][4] + 1
        cRep = mdh["sLC"][6] + 1
        cSet = mdh["sLC"][7] + 1
        cSeg = mdh["sLC"][8] + 1
        cIda = mdh["sLC"][9] + 1
        cIdb = mdh["sLC"][10] + 1
        cIdc = mdh["sLC"][11] + 1
        cIdd = mdh["sLC"][12] + 1
        cIde = mdh["sLC"][13] + 1

        cAcq = self.NAcq + 1
        self.NAcq = cAcq

        if cAcq > self.currentAlloc:

            self.currentAlloc = self.currentAlloc + self.allocSize
            # alloc           = np.zeros((1,self.allocSize), dtype = 'f')
            alloc = np.zeros((1, self.allocSize), dtype="i")
            self.NCol = np.concatenate((self.NCol, alloc[0]), axis=0).astype(int)
            self.NCha = np.concatenate((self.NCha, alloc[0]), axis=0).astype(int)
            self.Lin = np.concatenate((self.Lin, alloc[0]), axis=0).astype(int)
            self.Par = np.concatenate((self.Par, alloc[0]), axis=0).astype(int)
            self.Sli = np.concatenate((self.Sli, alloc[0]), axis=0).astype(int)
            self.Ave = np.concatenate((self.Ave, alloc[0]), axis=0).astype(int)
            self.Phs = np.concatenate((self.Phs, alloc[0]), axis=0).astype(int)
            self.Eco = np.concatenate((self.Eco, alloc[0]), axis=0).astype(int)
            self.Rep = np.concatenate((self.Rep, alloc[0]), axis=0).astype(int)
            self.Set = np.concatenate((self.Set, alloc[0]), axis=0).astype(int)
            self.Seg = np.concatenate((self.Seg, alloc[0]), axis=0).astype(int)
            self.Ida = np.concatenate((self.Ida, alloc[0]), axis=0).astype(int)
            self.Idb = np.concatenate((self.Idb, alloc[0]), axis=0).astype(int)
            self.Idc = np.concatenate((self.Idc, alloc[0]), axis=0).astype(int)
            self.Idd = np.concatenate((self.Idd, alloc[0]), axis=0).astype(int)
            self.Ide = np.concatenate((self.Ide, alloc[0]), axis=0).astype(int)
            self.centerCol = np.concatenate((self.centerCol, alloc[0]), axis=0).astype(
                int
            )
            self.centerLin = np.concatenate((self.centerLin, alloc[0]), axis=0).astype(
                int
            )
            self.centerPar = np.concatenate((self.centerPar, alloc[0]), axis=0).astype(
                int
            )
            self.timestamp = np.concatenate((self.timestamp, alloc[0]), axis=0).astype(
                int
            )
            self.pmutime = np.concatenate((self.pmutime, alloc[0]), axis=0).astype(int)
            self.IsReflected = np.concatenate(
                (self.IsReflected, np.zeros((1, self.allocSize), dtype=bool)[0]), axis=0
            ).astype(int)
            self.IsRawDataCorrect = np.concatenate(
                (self.IsRawDataCorrect, np.zeros((1, self.allocSize), dtype=bool)[0]),
                axis=0,
            ).astype(int)

            if hasattr(self, "slicePos"):
                self.slicePos = np.concatenate(
                    (self.slicePos, np.zeros((7, self.allocSize), dtype=np.float32)[:]),
                    axis=1,
                )
            else:
                self.slicePos = np.zeros((7, self.allocSize), dtype=np.float32)

            if hasattr(self, "iceParam"):
                self.iceParam = np.concatenate(
                    (
                        self.iceParam,
                        np.zeros(
                            (self.freadInfo["iceParamSz"], self.allocSize),
                            dtype=np.float32,
                        )[:],
                    ),
                    axis=1,
                )
            else:
                self.iceParam = np.zeros(
                    (self.freadInfo["iceParamSz"], self.allocSize), dtype=np.float32
                )

            if hasattr(self, "freeParam"):
                self.freeParam = np.concatenate(
                    (
                        self.freeParam,
                        np.zeros((4, self.allocSize), dtype=np.float32)[:],
                    ),
                    axis=1,
                )
            else:
                self.freeParam = np.zeros((4, self.allocSize), dtype=np.float32)

            self.memPos = np.concatenate(
                (self.memPos, np.zeros(self.allocSize))).astype(int)

        # save mdh information about current line

        self.NCol[cAcq - 1] = mdh["ushSamplesInScan"] + 0
        self.NCha[cAcq - 1] = mdh["ushUsedChannels"] + 0

        self.Lin[cAcq - 1] = cLin
        self.Par[cAcq - 1] = cPar
        self.Sli[cAcq - 1] = cSli
        self.Ave[cAcq - 1] = cAve
        self.Phs[cAcq - 1] = cPhs
        self.Eco[cAcq - 1] = cEco
        self.Rep[cAcq - 1] = cRep
        self.Set[cAcq - 1] = cSet
        self.Seg[cAcq - 1] = cSeg
        self.Ida[cAcq - 1] = cIda
        self.Idb[cAcq - 1] = cIdb
        self.Idc[cAcq - 1] = cIdc
        self.Idd[cAcq - 1] = cIdd
        self.Ide[cAcq - 1] = cIde
        self.centerCol[cAcq - 1] = mdh["ushKSpaceCentreColumn"] + 1
        self.centerLin[cAcq - 1] = mdh["ushKSpaceCentreLineNo"] + 1
        self.centerPar[cAcq - 1] = mdh["ushKSpaceCentrePartitionNo"] + 1
        self.timestamp[cAcq - 1] = mdh["ulTimeStamp"] + 1
        self.pmutime[cAcq - 1] = mdh["ulPMUTimeStamp"] + 1
        self.slicePos[:, cAcq - 1] = list(mdh["SlicePos"])
        self.iceParam[:, cAcq - 1] = list(mdh["aushIceProgramPara"])
        self.freeParam[:, cAcq - 1] = list(mdh["aushFreePara"])
        self.IsReflected[cAcq - 1] = int(
            bool(min((mdh["aulEvalInfoMask"][0] & 2 ** 24), 1))
        )
        self.IsRawDataCorrect[cAcq - 1] = int(
            bool(min((mdh["aulEvalInfoMask"][0] & 2 ** 10), 1))
        )
        # Save memory position
        self.memPos[cAcq - 1] = filePos

    def clean(self):

        if self.NAcq == 0:
            return -1

        # cut mdh data to actual size (remove over-allocated part)
        self.NCol = self.NCol[0 : self.NAcq]
        self.NCha = self.NCha[0 : self.NAcq]
        self.Lin = self.Lin[0 : self.NAcq]
        self.Par = self.Par[0 : self.NAcq]
        self.Sli = self.Sli[0 : self.NAcq]
        self.Ave = self.Ave[0 : self.NAcq]
        self.Phs = self.Phs[0 : self.NAcq]
        self.Eco = self.Eco[0 : self.NAcq]
        self.Rep = self.Rep[0 : self.NAcq]
        self.Set = self.Set[0 : self.NAcq]
        self.Seg = self.Seg[0 : self.NAcq]
        self.Ida = self.Ida[0 : self.NAcq]
        self.Idb = self.Idb[0 : self.NAcq]
        self.Idc = self.Idc[0 : self.NAcq]
        self.Idd = self.Idd[0 : self.NAcq]
        self.Ide = self.Ide[0 : self.NAcq]
        self.centerCol = self.centerCol[0 : self.NAcq]
        self.centerLin = self.centerLin[0 : self.NAcq]
        self.centerPar = self.centerPar[0 : self.NAcq]
        self.IsReflected = self.IsReflected[0 : self.NAcq]
        self.timestamp = self.timestamp[0 : self.NAcq]
        self.pmutime = self.pmutime[0 : self.NAcq]
        self.IsRawDataCorrect = self.IsRawDataCorrect[0 : self.NAcq]
        self.slicePos = self.slicePos[:, 0 : self.NAcq]
        self.iceParam = self.iceParam[:, 0 : self.NAcq]
        self.freeParam = self.freeParam[:, 0 : self.NAcq]
        self.memPos = self.memPos[0 : self.NAcq]

        self.NLin = max(self.Lin)
        self.NPar = max(self.Par)
        self.NSli = max(self.Sli)
        self.NAve = max(self.Ave)
        self.NPhs = max(self.Phs)
        self.NEco = max(self.Eco)
        self.NRep = max(self.Rep)
        self.NSet = max(self.Set)
        self.NSeg = max(self.Seg)
        self.NIda = max(self.Ida)
        self.NIdb = max(self.Idb)
        self.NIdc = max(self.Idc)
        self.NIdd = max(self.Idd)
        self.NIde = max(self.Ide)

        # ok, let us assume for now that all NCol and NCha entries are
        # the same for all mdhs:
        self.NCol = self.NCol[0]
        self.NCha = self.NCha[0]

        self.dataDims = [
            "Col",
            "Cha",
            "Lin",
            "Par",
            "Sli",
            "Ave",
            "Phs",
            "Eco",
            "Rep",
            "Set",
            "Seg",
            "Ida",
            "Idb",
            "Idc",
            "Idd",
            "Ide",
        ]

        # to reduce the matrix sizes of non-image scans, the size
        # of the refscan_obj()-matrix is reduced to the area of the
        # actually scanned acs lines (the outer part of k-space
        # that is not scanned is not filled with zeros)
        # self behaviour is controlled by flagSkipToFirstLine which is
        # set to true by default for everything but image scans
        if not self.FlagSkipToFirstLine:
            # the output matrix should include all leading zeros
            self.skipLin = 0
            self.skipPar = 0
        else:
            # otherwise, cut the matrix size to the start of the
            # first actually scanned line/partition (e.g. the acs/
            # phasecor data is only acquired in the k-space center)
            self.skipLin = min(self.Lin) - 1
            self.skipPar = min(self.Par) - 1

        NLinAlloc = max(1, self.NLin - self.skipLin)
        NParAlloc = max(1, self.NPar - self.skipPar)

        self.fullSize = [
            self.NCol,
            self.NCha,
            NLinAlloc,
            NParAlloc,
            self.NSli,
            self.NAve,
            self.NPhs,
            self.NEco,
            self.NRep,
            self.NSet,
            self.NSeg,
            self.NIda,
            self.NIdb,
            self.NIdc,
            self.NIdd,
            self.NIde,
        ]

        self.dataSize = self.fullSize

        if self.arg.get("removeOS", False):
            self.dataSize[0] = self.NCol / 2

        if self.arg.get("doAverage", False):
            self.dataSize[5] = 1

        if self.arg.get("averageReps", False):
            self.dataSize[8] = 1

        if self.arg.get("averageSets", False):
            self.dataSize[9] = 1

        if self.arg.get("ignoreSeg", False):
            self.dataSize[10] = 1

        # calculate sqzSize
        self.calcSqzSize()

        # calculate indices to target & source(raw)
        self.calcIndices()

        nByte = self.NCha * (self.freadInfo["szChannelHeader"] + 8 * self.NCol)

        # size for fread
        self.freadInfo["sz"] = [2, nByte / 8]
        # reshape size
        self.freadInfo["shape"] = [
            self.NCol + self.freadInfo["szChannelHeader"] / 8,
            self.NCha,
        ]
        # we need to cut MDHs from fread data
        self.freadInfo["cut"] = np.arange(
            self.freadInfo["szChannelHeader"] / 8,
            self.NCol + self.freadInfo["szChannelHeader"] / 8,
            1,
        )

    def readImage(self):

        # Transform Twix object in numpy array of raw data

        [selRange, selRangeSz, outSize] = self.calcRange()

        #        tmp = reshape(1:prod(double(self.fullSize(3:end))), self.fullSize(3:end))

        A = np.prod(np.asarray(self.fullSize[2:]))
        MM = np.arange(1, A + 1)
        tmp = np.reshape(MM, np.asarray(self.fullSize[2:]))

        # tmp = tmp(selRange{3:end})
        # cIxToRaw = self.ixToRaw[tmp]
        # del tmp
        # cIxToRaw = cIxToRaw[:]

        cIxToRaw = self.ixToRaw

        # delete all entries that point to zero (the "NULL"-pointer)
        cIxToRaw = np.delete(cIxToRaw, np.where(cIxToRaw == -1))

        # calculate cIxToTarg for possibly smaller, shifted + segmented
        # target matrix:

        # cIx = np.zeros((14,len(cIxToRaw)), dtype = 'f4')
        cIx = np.zeros((14, len(cIxToRaw)), dtype="i4")
        cIx[0, :] = self.Lin[cIxToRaw] - self.skipLin
        cIx[1, :] = self.Par[cIxToRaw] - self.skipPar
        cIx[2, :] = self.Sli[cIxToRaw]

        if self.arg.get("doAverage", False):
            cIx[3, :] = 1
        else:
            cIx[3, :] = self.Ave[cIxToRaw]

        cIx[4, :] = self.Phs[cIxToRaw]
        cIx[5, :] = self.Eco[cIxToRaw]
        if self.arg.get("averageReps", False):
            cIx[6, :] = 1
        else:
            cIx[6, :] = self.Rep[cIxToRaw]

        if self.arg.get("averageSets", False):
            cIx[7, :] = 1
        else:
            cIx[7, :] = self.Set[cIxToRaw]

        if self.arg.get("ignoreSeg", False):
            cIx[8, :] = 1
        else:
            cIx[8, :] = self.Seg[cIxToRaw]

        cIx[9, :] = self.Ida[cIxToRaw]
        cIx[10, :] = self.Idb[cIxToRaw]
        cIx[11, :] = self.Idc[cIxToRaw]
        cIx[12, :] = self.Idd[cIxToRaw]
        cIx[13, :] = self.Ide[cIxToRaw]

        # make sure that indices fit inside selection range
        for k in range(2, len(selRange)):
            tmp = cIx[k - 2, :]
            for l in range(0, len(selRange[k])):
                cIx[k - 2, tmp == selRange[k][l]] = l + 1

        cIxToTarg = self.sub2ind_double(
            selRangeSz[2:],
            (
                cIx[0, :],
                cIx[1, :],
                cIx[2, :],
                cIx[3, :],
                cIx[4, :],
                cIx[5, :],
                cIx[6, :],
                cIx[7, :],
                cIx[8, :],
                cIx[9, :],
                cIx[10, :],
                cIx[11, :],
                cIx[12, :],
                cIx[13, :],
            ),
        )

        mem = self.memPos[cIxToRaw]
        # sort mem for quicker access, sort cIxToTarg/Raw accordingly
        # [mem,ix]  = sort(mem)
        ix = np.argsort(mem)
        mem = np.sort(mem)
        cIxToTarg = cIxToTarg[ix]
        cIxToRaw = cIxToRaw[ix]

        out = np.zeros(np.asarray(outSize), dtype="complex64")
        out = np.reshape(out, (selRangeSz[0], selRangeSz[1], -1))

        # counter for proper scaling of averages/segments
        count_ave = np.zeros((1, 1, np.shape(out)[2]), dtype="i4")

        # subsref overloading makes self.that-calls slow, so we need to
        # avoid them whenever possible
        szScanHeader = self.freadInfo["szScanHeader"]
        readSize = self.freadInfo["sz"]
        readShape = self.freadInfo["shape"]
        readCut = self.freadInfo["cut"]

        # cutOS        = self.NCol/4+1:self.NCol*3/4

        bRemoveOS = self.arg.get("removeOS", False)
        bIsReflected = self.IsReflected
        # SRY store information about raw data correction
        bDoRawDataCorrect = self.arg.get("doRawDataCorrect", False)
        bIsRawDataCorrect = self.IsRawDataCorrect
        if bDoRawDataCorrect:
            rawDataCorrect = self.arg["rawDataCorrectionFactors"]

        fid = self.fileopen()
        for k in range(len(mem)):
            # skip scan header
            fid.seek(mem[k] + szScanHeader, 0)
            yy = np.prod(np.asarray(readSize)).astype(int)
            frt = "f" * yy
            raw = np.asarray(struct.unpack(frt, fid.read(yy * 4)))
            new_size = np.asarray(readSize, dtype=int)

            raw = np.reshape(raw, (new_size[1], new_size[0]))
            raw = (raw[:, 0] + 1j * raw[:, 1]).astype("complex64")
            raw = np.reshape(raw, np.asarray(readShape, dtype=int))
            raw = raw[np.asarray(readCut, dtype="i"), :]

            # SRY apply raw data correction if necessary
            # if ( bDoRawDataCorrect && bIsRawDataCorrect(cIxToRaw(k)) ):
            # there are two ways to do this: multiply where the flag is
            # set, or divide where it is not set.  There are significantly
            # more points without the flag, so multiplying is more
            # efficient
            # raw = bsxfun(@times, raw, rawDataCorrect)

            # select channels
            raw = raw[:, selRange[1].astype(int) - 1]

            if bRemoveOS:
                # remove oversampling in read
                raw = np.fft.ifft(raw)
                raw = np.delete(raw, cutOS, 0)
                raw = fft(raw)

            if bIsReflected[cIxToRaw[k]]:
                raw = np.flip(raw, 0)

            # select columns and sort data

            out[:, :, cIxToTarg[k].astype(int) - 1] = (
                out[:, :, cIxToTarg[k].astype(int) - 1]
                + raw[selRange[0].astype(int) - 1, :]
            )
            count_ave[0, 0, cIxToTarg[k].astype(int) - 1] = (
                count_ave[0, 0, cIxToTarg[k].astype(int) - 1] + 1
            )

        fid.close()

        # proper scaling (we don't want to sum our data but average it)
        count_ave = 1.0 / np.maximum(count_ave, 1)
        count_ave = np.tile(count_ave, (np.shape(out)[0], np.shape(out)[1], 1))
        out = np.multiply(out, count_ave, dtype="complex64")

        # reshape out
        outSize = np.array(outSize, dtype="i")
        outSize[2:] = np.flip(outSize[2:], 0)

        out = np.reshape(out, outSize)
        out = np.transpose(out, (0, 1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2))

        return out

    def calcRange(self):

        import numpy as np

        selRange = []
        for k in range(0, len(self.dataSize)):
            selRange.append(np.arange(1, self.dataSize[k] + 1))

        outSize = self.dataSize

        for k in range(0, len(selRange)):
            if max(selRange[k]) > self.dataSize[k]:
                print("selection out of range")
                break

        selRangeSz = np.ones((len(self.dataSize),), dtype="i")
        for k in range(0, len(selRange)):
            selRangeSz[k] = len(selRange[k])

        return (selRange, selRangeSz, outSize)

    def setDefaultFlags(self):
        # method to set flags to default values
        self.arg["removeOS"] = False
        self.arg["doAverage"] = False
        self.arg["averageReps"] = False
        self.arg["averageSets"] = False
        self.arg["ignoreSeg"] = False
        self.arg["doRawDataCorrect"] = False
        if self.dataType == "image":
            self.arg["skipToFirstLine"] = False
        else:
            self.arg["skipToFirstLine"] = True
        if "rawDataCorrectionFactors" not in self.arg.keys():
            self.arg["rawDataCorrectionFactors"] = []

    def resetFlags(self):
        # method to reset flags to default values
        self.flagRemoveOS = False
        self.flagDoAverage = False
        self.flagAverageReps = False
        self.flagAverageSets = False
        self.flagIgnoreSeg = False
        self.flagDoRawDataCorrect = False
        if self.dataType == "image":
            self.flagSkipToFirstLine = False
        else:
            self.flagSkipToFirstLine = True

    def set_flagRemoveOS(self, val):
        # set method for removeOS
        self.arg["removeOS"] = bool(val)
        # we also need to recalculate our data size:
        if self.arg["removeOS"]:
            self.dataSize[0] = self.NCol[0] / 2
            self.sqzSize[0] = self.NCol[0] / 2
        else:
            self.dataSize[0] = self.NCol[0]
            self.sqzSize[0] = self.NCol[0]

    def get_flagRemoveOS(self):
        out = self.arg["removeOS"]
        return out

    def set_flagDoAverage(self, val):
        # set method for doAverage
        self.arg["doAverage"] = bool(val)
        if self.arg.doAverage:
            self.dataSize[5] = 1
        else:
            self.dataSize[5] = self.NAve
        # update sqzSize
        self.calcSqzSize

    def get_flagDoAverage(self):
        out = self.arg["doAverage"]
        return out

    def set_flagAverageReps(self, val):
        # set method for doAverage
        self.arg["averageReps"] = bool(val)
        if self.arg["averageReps"]:
            self.dataSize[8] = 1
        else:
            self.dataSize[8] = self.NRep
        # update sqzSize
        self.calcSqzSize

    def get_flagAverageReps(self):
        out = self.arg["averageReps"]
        return out

    def set_flagAverageSets(self, val):
        # set method for doAverage
        self.arg["averageSets"] = bool(val)
        if self.arg["averageSets"]:
            self.dataSize[9] = 1
        else:
            self.dataSize[9] = self.NSet
        # update sqzSize
        self.calcSqzSize

    def get_flagAverageSets(self):
        out = self.arg["averageSets"]
        return out

    def set_flagSkipToFirstLine(self, val):
        val = bool(val)
        if val != self.arg["skipToFirstLine"]:
            self.arg["skipToFirstLine"] = val

            if self.arg["skipToFirstLine"]:
                self.skipLin = min(self.Lin) - 1
                self.skipPar = min(self.Par) - 1
            else:
                self.skipLin = 0
                self.skipPar = 0
            NLinAlloc = max(1, self.NLin - self.skipLin)
            NParAlloc = max(1, self.NPar - self.skipPar)
            self.fullSize[2:3] = [NLinAlloc, NParAlloc]
            self.dataSize[2:3] = self.fullSize[2:3]

            # update sqzSize
            self.calcSqzSize
            # update indices
            self.calcIndices

    def get_flagSkipToFirstLine(self):
        out = self.arg["skipToFirstLine"]
        return out

    def set_flagIgnoreSeg(self, val):
        # set method for ignoreSeg
        self.arg["ignoreSeg"] = bool(val)
        if self.arg["ignoreSeg"]:
            self.dataSize[10] = 1
        else:
            self.dataSize[10] = self.NSeg
        # update sqzSize
        self.calcSqzSize

    def get_flagIgnoreSeg(self):
        out = self.arg["ignoreSeg"]
        return out

    # SRY: accessor methods for raw data correction
    def get_flagDoRawDataCorrect(self):
        out = self.arg["doRawDataCorrect"]
        return out

    def set_flagDoRawDataCorrect(self, val):
        val = bool(val)
        if val == True or self.softwareVersion == "vd":
            print("raw data correction for VD not supported/tested yet")
            return -1
        self.arg["doRawDataCorrect"] = val

    def get_RawDataCorrectionFactors(self):
        out = self.arg["rawDataCorrectionFactors"]
        return out

    def set_RawDataCorrectionFactors(self, val):
        # this may not work if trying to set the factors before NCha has
        # a meaningful value (ie before calling clean)
        if not isrow(val) or len(val) != self.NCha:
            print("RawDataCorrectionFactors must be a 1xNCha row vector")
            return -1
        self.arg["rawDataCorrectionFactors"] = bool(val)

    def calcSqzSize(self):
        # calculate sqzSize and sqzDims
        self.sqzSize = []
        self.sqzDims = []

        self.sqzSize.append(self.dataSize[0])
        self.sqzDims.append("Col")

        for k in range(1, len(self.dataSize)):
            if self.dataSize[k] > 1:
                self.sqzSize.append(self.dataSize[k])
                self.sqzDims.append(self.dataDims[k])

    def calcIndices(self):
        # calculate indices to target & source(raw)
        LinIx = self.Lin - self.skipLin
        ParIx = self.Par - self.skipPar

        self.ixToTarget = self.sub2ind_double(
            tuple(np.asarray(self.fullSize[2:])),
            (
                LinIx,
                ParIx,
                self.Sli,
                self.Ave,
                self.Phs,
                self.Eco,
                self.Rep,
                self.Set,
                self.Seg,
                self.Ida,
                self.Idb,
                self.Idc,
                self.Idd,
                self.Ide,
            ),
        )
        # now calculate inverse index
        # inverse index of lines that are not measured is zero

        self.ixToRaw = -1 * np.ones((np.prod(self.fullSize[2:])), dtype="i")

        # subsref overloading makes self.that-calls slow, so we need to
        # avoid them whenever possible
        ixToTarg = self.ixToTarget
        for k in range(0, len(ixToTarg)):
            self.ixToRaw[ixToTarg[k] - 1] = k

    def fileopen(self):

        # Open .dat file associated to twix object
        fid = open(self.filename, "rb")

        return fid

    ############# helper function ############
    def sub2ind_double(self, sz, varargin):
        # Linear index from multiple subscripts.
        ###########################################

        ndx = varargin[-1] - 1

        for i in range(len(sz) - 1, -1, -1):
            ix = varargin[i]
            ndx = sz[i] * ndx + ix - 1

        ndx = ndx + 1

        return ndx
