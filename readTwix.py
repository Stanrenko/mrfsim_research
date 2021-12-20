##############################################################################
#
# readTwix.py
# Module of functions necessayr to read Siemens raw .dat file (for VD and VE only)
# Recquires TwixObject.py class
# Author: B. Marty - 10/2018
""" this modules does something """

import struct
import time
import numpy as np

import TwixObject


def map_VBVD(filename, *options):
    """ this function does something """

    ##############################################################################
    # Inputs:
    #    filename = path of the .dat file
    #    options (optional) = dictionnary of different options
    #            A boolean value is associated to each option
    #            List of compatible options: ReadHeader, RemoveOS, DoAverage,
    #                                        AverageReps, AverageSets,
    #                                        IgnoreSegments, RawDataCorrect
    ##############################################################################

    ###### Parse options #####
    arg = dict()

    arg["bReadImaScan"] = True
    arg["bReadNoiseScan"] = True
    arg["bReadPCScan"] = True
    arg["bReadRefScan"] = True
    arg["bReadRefPCScan"] = True
    arg["bReadRTfeedback"] = True
    arg["bReadPhaseStab"] = True
    arg["bReadHeader"] = True

    if "ReadHeader" in options:
        arg["bReadHeader"] = options["ReadHeader"]
    if "RemoveOS" in options:
        arg["bRemoveOS"] = options["RemoveOS"]
    if "DoAverage" in options:
        arg["bDoAverage"] = options["DoAverage"]
    if "AverageReps" in options:
        arg["bAverageReps"] = options["AverageReps"]
    if "ReadHeader" in options:
        arg["bAverageSets"] = options["AverageSets"]
    if "IgnoreSegments" in options:
        arg["bIgnoreSegments"] = options["IgnoreSegments"]
    if "RawDataCorrect" in options:
        arg["bRawDataCorrect"] = options["RawDataCorrect"]

    ##############################################################################


    start_time = time.time()

    fid = open(filename, "rb")
    # with open(filename, "r") as fid:
    fid.seek(0, 2)
    fileSize = int(fid.tell())

    ##### Start of actual measurement data (without header)
    fid.seek(0, 0)

    firstInt = struct.unpack("i", fid.read(4))
    firstInt = firstInt[0]
    secondInt = struct.unpack("i", fid.read(4))
    secondInt = secondInt[0]

    if firstInt < 10000 and secondInt <= 64:
        version = "vd"
        print("Software version: VD/VE")

        ## number of different scans in file stored in 2nd
        NScans = secondInt
        measID = struct.unpack("i", fid.read(4))
        fileID = struct.unpack("i", fid.read(4))
        ## measOffset: points to beginning of header, usually at 10240 bytes
        measOffset = struct.unpack("ii", fid.read(8))
        measOffset = measOffset[0]
        measLength = struct.unpack("ii", fid.read(8))
        measLength = measLength[0]
        fid.seek(measOffset, 0)
        hdrLength = struct.unpack("i", fid.read(4))
        hdrLength = hdrLength[0]

    else:
        version = "vb"
        print("Software version: VB")
        measOffset = 0
        hdrLength = firstInt
        NScans = 1  # VB does not support multiple scans in one file

    datStart = measOffset + hdrLength

    #   1) reading all MDHs to find maximum line no., partition no.,... for
    #      ima, ref,... scan
    # data will be read in two steps (two while loops):
    #   2) reading the data

    percentFinished = 0
    cPos = measOffset

    twix_obj = dict()

    for s in range(0, NScans):

        twix_obj[str(s)] = dict()

        twix_obj[str(s)]["image"] = TwixObject.TwixObject(
            arg, "image", filename, version
        )
        twix_obj[str(s)]["noise"] = TwixObject.TwixObject(
            arg, "noise", filename, version
        )
        twix_obj[str(s)]["phasecor"] = TwixObject.TwixObject(
            arg, "phasecor", filename, version
        )
        twix_obj[str(s)]["refscan"] = TwixObject.TwixObject(
            arg, "refscan", filename, version
        )
        twix_obj[str(s)]["refscanPC"] = TwixObject.TwixObject(
            arg, "refscanPC", filename, version
        )
        twix_obj[str(s)]["RTfeedback"] = TwixObject.TwixObject(
            arg, "RTfeedback", filename, version
        )
        twix_obj[str(s)]["vop"] = TwixObject.TwixObject(arg, "vop", filename, version)
        twix_obj[str(s)]["phasestab"] = TwixObject.TwixObject(
            arg, "phasestab", filename, version
        )
        if arg["bReadHeader"]:
            twix_obj[str(s)]["hdr"] = read_twix_hdr(fid)

        fid.seek(cPos, 0)
        hdr_len = struct.unpack("i", fid.read(4))[0]

        # jump to first mdh
        cPos = cPos + hdr_len

        while 1:

            if version == "vd":
                (mdh, mask) = evalMDHvd(fid, cPos)
            else:
                print("error: only vd/ve software versions supported")
                break

            if mask["MDH_ACQEND"] or mdh["ulDMALength"] == 0:
                cPos = cPos + mdh["ulDMALength"]
                if cPos % 512 > 0:  # % = modulo
                    cPos = cPos + 512 - (cPos % 512)
                break
            elif cPos + 128 > fileSize:  # fail-safe; in case we miss MDH_ACQEND
                print("This file seems to be corrupted (likely missing data)")
                break

            if mask["MDH_SYNCDATA"]:
                cPos = cPos + mdh["ulDMALength"]
                continue

            if mask["MDH_IMASCAN"] and arg["bReadImaScan"]:
                twix_obj[str(s)]["image"].readMDH(mdh, cPos)

            if mask["MDH_NOISEADJSCAN"] and arg["bReadNoiseScan"]:
                twix_obj[str(s)]["noise"].readMDH(mdh, cPos)

            if (mask["MDH_PHASCOR"] and not mask["MDH_PATREFSCAN"]) and arg[
                "bReadPCScan"
            ]:
                twix_obj[str(s)]["phasecor"].readMDH(mdh, cPos)

            if (
                not mask["MDH_PHASCOR"]
                and (mask["MDH_PATREFSCAN"] or mask["MDH_PATREFANDIMASCAN"])
            ) and arg["bReadRefScan"]:
                twix_obj[str(s)]["refscan"].readMDH(mdh, cPos)

            if (mask["MDH_PATREFSCAN"] and mask["MDH_PHASCOR"]) and arg[
                "bReadRefPCScan"
            ]:
                twix_obj[str(s)]["refscanPC"].readMDH(mdh, cPos)

            if (
                (mask["MDH_RTFEEDBACK"] or mask["MDH_HPFEEDBACK"])
                and not mask["MDH_VOP"]
                and arg["bReadRTfeedback"]
            ):
                twix_obj[str(s)]["RTfeedback"].readMDH(mdh, cPos)

            if (mask["MDH_RTFEEDBACK"] and mask["MDH_VOP"]) and arg["bReadRTfeedback"]:
                twix_obj[str(s)]["vop"].readMDH(mdh, cPos)

            if (mask["MDH_PHASESTABSCAN"] or mask["MDH_REFPHASESTABSCAN"]) and arg[
                "bReadPhaseStab"
            ]:
                twix_obj[str(s)]["phasestab"].readMDH(mdh, cPos)

            # jump to mdh of next scan
            cPos = cPos + mdh["ulDMALength"]

            if cPos / fileSize * 100 > percentFinished + 1:
                percentFinished = int(cPos // fileSize * 100)
                elapsed_time = time.time()
                elapsed_time = elapsed_time - start_time
                time_left = (fileSize // cPos - 1) * elapsed_time

                if not "progress_str" in locals():
                    prevLength = 0
                else:
                    prevLength = len(progress_str)

                progress_str = "%i %% parsed in %d s; estimated time left: %d s \n" % (
                    percentFinished,
                    elapsed_time,
                    time_left,
                )

                #print(progress_str)
            # fprintf([repmat('\b',1,prevLength) '%s'],progress_str)
        # buf = "C = %d\n" % c

        if twix_obj[str(s)]["image"].NAcq == 0:
            del twix_obj[str(s)]["image"]
        else:
            twix_obj[str(s)]["image"].clean()

        if twix_obj[str(s)]["noise"].NAcq == 0:
            del twix_obj[str(s)]["noise"]
        else:
            twix_obj[str(s)]["noise"].clean()

        if twix_obj[str(s)]["phasecor"].NAcq == 0:
            del twix_obj[str(s)]["phasecor"]
        else:
            twix_obj[str(s)]["phasecor"].clean()

        if twix_obj[str(s)]["refscan"].NAcq == 0:
            del twix_obj[str(s)]["refscan"]
        else:
            twix_obj[str(s)]["refscan"].clean()

        if twix_obj[str(s)]["refscanPC"].NAcq == 0:
            del twix_obj[str(s)]["refscanPC"]
        else:
            twix_obj[str(s)]["refscanPC"].clean()

        if twix_obj[str(s)]["RTfeedback"].NAcq == 0:
            del twix_obj[str(s)]["RTfeedback"]
        else:
            twix_obj[str(s)]["RTfeedback"].clean()

        if twix_obj[str(s)]["vop"].NAcq == 0:
            del twix_obj[str(s)]["vop"]
        else:
            twix_obj[str(s)]["vop"].clean()

        if twix_obj[str(s)]["phasestab"].NAcq == 0:
            del twix_obj[str(s)]["phasestab"]
        else:
            twix_obj[str(s)]["phasestab"].clean()

    fid.close()
    elapsed_time = time.time()
    elapsed_time = elapsed_time - start_time
    progress_str = "Raw data parsed in %f s \n" % round(elapsed_time, 2)

    print(progress_str)

    # import pdb; pdb.set_trace()
    # 1/0
    return twix_obj


def evalMDHvd(fid, cPos):
    # see pkg/MrServers/MrMeasSrv/SeqIF/MDH/mdh.h
    # and pkg/MrServers/MrMeasSrv/SeqIF/MDH/MdhProxy.h

    # we need to differentiate between 'scan header' and 'channel header'
    # since these are used in VD versions:
    import struct as st

    szScanHeader = 192  # [bytes]
    szChannelHeader = 32  # [bytes]

    mdh = dict()

    # inlining of readScanHeader
    fid.seek(cPos, 0)

    FirstByte = st.unpack("I", fid.read(4))[0]

    mdh["ulDMALength"] = FirstByte & (2 ** 25 - 1)
    mdh["ulPackBit"] = FirstByte >> 25 & (2 ** 1 - 1)
    mdh["ulPCI_rx"] = FirstByte >> 26

    mdh["lMeasUID"] = st.unpack("I", fid.read(4))[0]
    mdh["ulScanCounter"] = st.unpack("I", fid.read(4))[0]
    mdh["ulTimeStamp"] = st.unpack("I", fid.read(4))[0]
    mdh["ulPMUTimeStamp"] = st.unpack("I", fid.read(4))[0]

    fid.seek(cPos + 40)

    mdh["aulEvalInfoMask"] = st.unpack("II", fid.read(8))
    # mdh.aulEvalInfoMask              = fread(fid,  [1 2], 'uint32')
    dummy = st.unpack("HH", fid.read(4))
    mdh["ushSamplesInScan"] = dummy[0]
    mdh["ushUsedChannels"] = dummy[1]
    mdh["sLC"] = st.unpack("HHHHHHHHHHHHHH", fid.read(28))
    dummy = st.unpack("HHHHHHHHHH", fid.read(20))
    #     mdh['sCutOff']                    = dummy[0:1];
    mdh["ushKSpaceCentreColumn"] = dummy[2]
    # mdh['ushCoilSelect']              = dummy[3];
    mdh["ushKSpaceCentreLineNo"] = dummy[8]
    mdh["ushKSpaceCentrePartitionNo"] = dummy[9]
    mdh["SlicePos"] = st.unpack("fffffff", fid.read(28))
    dummy = st.unpack("HHHHHHHHHHHHHHHHHHHHHHHHHHHH", fid.read(56))
    mdh["aushIceProgramPara"] = dummy[0:24]
    mdh["aushFreePara"] = dummy[24:28]  # actually aushReservedPara;
    # there's no freePara in VD

    mask = dict()

    # inlining of evalInfoMask
    mask["MDH_ACQEND"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 0), 1)
    mask["MDH_RTFEEDBACK"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 1), 1)
    mask["MDH_HPFEEDBACK"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 2), 1)
    mask["MDH_SYNCDATA"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 5), 1)
    mask["MDH_RAWDATACORRECTION"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 10), 1)
    mask["MDH_REFPHASESTABSCAN"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 14), 1)
    mask["MDH_PHASESTABSCAN"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 15), 1)
    mask["MDH_SIGNREV"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 17), 1)
    mask["MDH_PHASCOR"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 21), 1)
    mask["MDH_PATREFSCAN"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 22), 1)
    mask["MDH_PATREFANDIMASCAN"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 23), 1)
    mask["MDH_REFLECT"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 24), 1)
    mask["MDH_NOISEADJSCAN"] = min((mdh["aulEvalInfoMask"][0] & 2 ** 25), 1)
    #     mask.MDH_VOP             = min((mdh['aulEvalInfoMask'][1] & 2**(53-32)),1)
    mask["MDH_VOP"] = 0

    mask["MDH_IMASCAN"] = 1

    if (
        mask["MDH_ACQEND"]
        or mask["MDH_RTFEEDBACK"]
        or mask["MDH_HPFEEDBACK"]
        or mask["MDH_PHASCOR"]
        or mask["MDH_NOISEADJSCAN"]
        or mask["MDH_SYNCDATA"]
    ):
        mask["MDH_IMASCAN"] = 0

    # otherwise the PATREFSCAN may be overwritten
    if mask["MDH_PHASESTABSCAN"] or mask["MDH_REFPHASESTABSCAN"]:
        mask["MDH_PATREFSCAN"] = 0
        mask["MDH_PATREFANDIMASCAN"] = 0
        mask["MDH_IMASCAN"] = 0

    if mask["MDH_PATREFSCAN"] and not mask["MDH_PATREFANDIMASCAN"]:
        mask["MDH_IMASCAN"] = 0

    #  pehses: the pack bit indicates that multiple ADC are packed into one
    #  DMA, often in EPI scans (controlled by fRTSetReadoutPackaging in IDEA)
    #  since this code assumes one adc (x NCha) per DMA, we have to correct
    #  the "DMA length"
    #      if mdh.ulPackBit
    #  it seems that the packbit is not always set correctly

    if not mask["MDH_SYNCDATA"] and not mask["MDH_ACQEND"] and mdh["ulDMALength"] != 0:
        mdh["ulDMALength"] = (
            szScanHeader
            + (2 * 4 * mdh["ushSamplesInScan"] + szChannelHeader)
            * mdh["ushUsedChannels"]
        )

    return (mdh, mask)


def read_twix_hdr(fid):

    import struct
    import re

    nbuffers = struct.unpack("i", fid.read(4))
    nbuffers = nbuffers[0]

    prot = dict()

    for b in range(0, nbuffers, 1):
        namesz = 0
        byte = 1
        while byte != 0:  # look for NULL-character
            byte = struct.unpack("b", fid.read(1))
            byte = byte[0]
            namesz = namesz + 1

        fid.seek(-namesz, 1)
        bufname = struct.unpack("s" * namesz, fid.read(namesz))
        bufname = "".join([c.decode("utf8") for c in bufname])
        buflen = struct.unpack("i", fid.read(4))
        buflen = buflen[0]
        buffer = struct.unpack("s" * buflen, fid.read(buflen))
        buffer = "".join([c.decode("utf8") for c in buffer])
        buffer = re.sub("\n\s+\n", "", buffer)  # delete empty lines

        prot[bufname] = parse_buffer(buffer, bufname)

    return prot


def parse_buffer(buffer, bufname):

    import re

    xprot_temp = re.split(
        "### ASCCONV BEGIN.+### ASCCONV END ###", buffer, flags=re.DOTALL
    )[0]

    ascconv = re.findall(
        "### ASCCONV BEGIN.+### ASCCONV END ###", buffer, flags=re.DOTALL
    )

    # if ascconv:
    # prot = parse_ascconv(ascconv);
    # else:
    prot = dict()

    if xprot_temp != None:
        prot = parse_xprot(xprot_temp)

        # if isstruct(xprot)
        # name   = cat(1,fieldnames(prot),fieldnames(xprot));
        # val    = cat(1,struct2cell(prot),struct2cell(xprot));
        # [~,ix] = unique(name);
        # prot   = cell2struct(val(ix),name(ix));

    return prot


def parse_xprot(buffer):

    import re
    import numpy as np

    xprot = dict()
    tokens = (
        re.findall('<ParamBool\."(\w+)">\s*{([^}]*)', buffer)
        + re.findall('<ParamLong\."(\w+)">\s*{([^}]*)', buffer)
        + re.findall('<ParamString\."(\w+)">\s*{([^}]*)', buffer)
        + re.findall(
            '<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)', buffer
        )
    )

    for m in range(0, len(tokens)):

        name = tokens[m][0]

        # field name has to start with letter
        if not name[0].isalpha():
            name = "x" + name

        value = re.sub('("*)|( *<\w*> *[^\n]*)', "", tokens[m][-1])
        value = value.strip()
        value = re.sub("\s+", " ", value)

        # find arrays of numerical
        value_array = re.findall("\d+\s", value)
        if value_array:
            new_value = np.zeros((1, len(value_array)))
            for i in range(0, len(value_array)):
                new_value[0, i] = float(value_array[i][0])
            value = new_value

        if len(value) == 0:
            value = []

        try:
            value = float(value)
        except:
            value = value

        xprot[name] = value

    return xprot


def parse_ascconv(buffer):

    import re
    import numpy as np

    mrprot = dict()

    # vararray = re.findall('(?<name>\S*)\s*=\s(?<value>\S*)', buffer)
    vararray = re.findall("(\S*)\s*=\s(\S*)", buffer)

    for b in range(0, len(vararray)):

        try:
            value = float(vararray[b][1])
        except:
            value = vararray[b][1]

        # now split array name and index (if present)
        v = re.findall(vararray[b][0], "(\w*)\[([0-9]*)\]|(\w*)")

        cnt = -1
        tmp = cell(2, numel(v))

        breaked = False

        for k in range(0, len(v)):
            cnt = cnt + 1
            tmp[0, cnt] = "."
            if not isalpha(v[k][0]):
                breaked = True
                break

            tmp[1, cnt] = v[k][1]
            if v[k][1]:
                cnt = cnt + 1
                tmp[1, cnt] = "{}"
                tmp[2, cnt] = 1 + float(v[k][1])

        # if ~breaked
        # S = substruct(tmp{:});
        # mrprot = subsasgn(mrprot,S,value);

    return mprot


def detect_TwixImg(parsedFile):

    # This function will detect which twix object is indeed the image (or spectrum) raw data
    # from the parsed twix file provided by mapVBVD
    
    intkeys = sorted([int(k) for k in parsedFile.keys()])

    idx_ok = float("nan")
    for k in intkeys[::-1]:
        if "image" in parsedFile[str(k)].keys():
            idx_ok = k
            break
    return idx_ok


# if __name__ == "__main__":
#    filename = r"R:\Rmn\BMarty\1_Siemens_3T\\1_Projets_Recherche\&3_2017_Fingerprinting\3_Data\180924\meas_MID00021_FID19307_JAMBES_raFin_phantom.dat"
#    mapVBVD(filename)
