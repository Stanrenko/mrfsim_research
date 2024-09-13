
#import matplotlib
#matplotlib.use("TkAgg")
from image_series import *
from dictoptimizers import SimpleDictSearch,BruteDictSearch
from mutools.optim.dictsearch import dictmodel

from PIL import Image
from utils_simu import *
#from utils_reco import calculate_sensitivity_map_3D,kdata_aggregate_center_part,calculate_displacement,build_volume_singular_2Dplus1_cc_allbins_registered,simulate_nav_images_multi,calculate_displacement_ml,estimate_weights_bins,calculate_displacements_singlechannel,calculate_displacements_singlechannel,calculate_displacements_allchannels,coil_compression_2Dplus1,build_volume_2Dplus1_cc_allbins
from utils_reco import *
from utils_mrf import *
import math
import nibabel as nib
try :
    import cupy as cp
except:
    pass
# machines
path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"

import sys
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context


os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
import cfl
from bart import bart

DEFAULT_OPT_CONFIG="opt_config.json"
DEFAULT_OPT_CONFIG_2STEPS="opt_config_twosteps.json"

@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("index", int, default=-1, description="Header index")
def getTR(filename,index):
    twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock"],
                                   optional_additional_arrays=["SliceThickness"])


    dTR=twix[index]["hdr"]["Meas"]["sWipMemBlock"]["adFree"][1]-twix[-1]["hdr"]["Meas"]["alTE"][0]/1000
    print(twix[index]["hdr"]["Meas"]["sWipMemBlock"]["adFree"][1])
    print(twix[index]["hdr"]["Meas"]["alTE"][0]/1000)

    print(twix[index]["hdr"]["Meas"]["sWipMemBlock"]["adFree"])
    
    total_TR=twix[index]["hdr"]["Meas"]["alTR"][0]/1e6
    print("Echo spacing is {} ms".format(dTR))
    print("Total TR is {} s".format(total_TR))
    return

@machine
@set_parameter("filename", str, default=None, description="Filename for getting metatada")
@set_parameter("filemha", str, default=None, description="Map (.mha)")
@set_parameter("suffix",str,default="")
@set_parameter("nifti",bool,default=False)
def getGeometry(filename,filemha,suffix,nifti):
    if filename is None:
        filename=str.split(filemha,"_MRF_map")[0]+".dat"
    if nifti:
        filemha_adjusted=str.replace(filemha,".mha","{}.nii".format(suffix))
    else:
        filemha_adjusted=str.replace(filemha,".mha","{}.mha".format(suffix))
    geom,is3D,orientation=get_volume_geometry(filename)
    data=np.array(io.read(filemha))
    print(data.shape)
    if is3D:
        #data=np.flip(np.moveaxis(data,0,-1),axis=(1,2))
        if orientation=="coronal":
            data=np.flip(np.moveaxis(data,(0,1,2),(1,2,0)),axis=(1))
            #data=np.moveaxis(data,0,2)
        elif orientation=="transversal":
            data=np.moveaxis(data,0,2)[:,::-1]
    else:
        data=np.moveaxis(data,0,2)[:,::-1]
    
    
    vol = io.Volume(data, **geom)
    io.write(filemha_adjusted,vol)
    return

@machine
@set_parameter("filedico", str, default=None, description="Dictionary .pkl containing parameters")
@set_parameter("filevolume", str, default=None, description="Volume (.npy)")
@set_parameter("suffix",str,default="")
@set_parameter("nifti",bool,default=False)
@set_parameter("apply_offset",bool,default=False,description="Apply offset (for having the absolute position in space) - should be false for unwarping volumes")
@set_parameter("reorient",bool,default=True,description="Reorient input volumes")
def convertArrayToImage(filedico,filevolume,suffix,nifti,apply_offset,reorient):


    extension=str.split(filevolume,".")[-1]

    print(extension)
    if ("nii" in extension) or (extension=="mha"):
        func_load=io.read
    elif (extension=="npy"):
        func_load=np.load
    else:
        raise ValueError("Unknown extension {}".format(extension))


    if nifti:
        filemha_adjusted=str.replace(filevolume,".{}".format(extension),"{}.nii".format(suffix))
    else:
        filemha_adjusted=str.replace(filevolume,".{}".format(extension),"{}.mha".format(suffix))
    
    with open(filedico,"rb") as file:
        dico=pickle.load(file)
    
    spacing=dico["spacing"]
    origin=dico["origin"]
    orientation=dico["orientation"]
    is3D=dico["is3D"]

    if apply_offset:

        offset=dico["offset"]
        print("Applying offset {}".format(offset))
        
        origin=np.array(origin)
        origin[-1]=origin[-1]+offset
        origin=tuple(origin)
        

    geom={"origin":origin,"spacing":spacing}
    print(geom)

    data=np.abs(np.array(func_load(filevolume)).squeeze())
    print(data.shape)
    if reorient:
        print("Reorienting input volume")
        if is3D:
            #data=np.flip(np.moveaxis(data,0,-1),axis=(1,2))
            offset=data.ndim-3
            if orientation=="coronal":
                
                data=np.flip(np.moveaxis(data,(offset,offset+1,offset+2),(offset+1,offset+2,offset)),axis=(offset,offset+1))
                #data=np.moveaxis(data,0,2)
            elif orientation=="transversal":
                # data=np.moveaxis(data,offset,offset+2)
                data=np.flip(np.moveaxis(data,offset,offset+2),axis=(offset+1,offset+2))

            elif orientation=="sagittal":
                # data=np.moveaxis(data,offset,offset+2)
                data=np.flip(np.moveaxis(data,(offset,offset+1,offset+2),(offset,offset+2,offset+1)))
        else:
            data=np.moveaxis(data,0,2)[:,::-1]
    
    
    vol = io.Volume(data, **geom)
    io.write(filemha_adjusted,vol)
    return



@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("dens_adj", bool, default=True, description="Radial density adjustment")
@set_parameter("suffix",str,default="")
@set_parameter("select_first_rep", bool, default=False, description="Select the first central partition repetition")
@set_parameter("index", int, default=-1, description="Header index")
def build_kdata(filename,suffix,dens_adj,select_first_rep,index):

    if dens_adj:
        filename_kdata = str.split(filename, ".dat")[0] + suffix + "_kdata.npy"
    #filename_kdata_no_densadj = str.split(filename, ".dat")[0] + suffix + "_no_densadj_kdata.npy"
    else:
         filename_kdata =str.split(filename, ".dat")[0] + suffix + "_no_densadj_kdata.npy"
    filename_save = str.split(filename, ".dat")[0] + "{}.npy".format(suffix)
    filename_nav_save = str.split(filename, ".dat")[0] + "{}_nav.npy".format(suffix)
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    folder = "/".join(str.split(filename, "/")[:-1])


    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        # twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
        #                            optional_additional_arrays=["SliceThickness"])
        #
        # if np.max(np.argwhere(np.array(twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]) > 0)) >= 16:
        #     use_navigator_dll = True
        # else:
        #     use_navigator_dll = False




        # alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        # x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        # y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        # z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]
        # dTR=twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["adFree"][1]-twix[-1]["hdr"]["Meas"]["alTE"][0]/1000
        # total_TR=twix[-1]["hdr"]["Meas"]["alTR"][0]/1e6
        #
        # nb_part = twix[-1]["hdr"]["Meas"]["Partitions"]
        #
        # dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV,
        #                   "use_navigator_dll": use_navigator_dll, "nb_part": nb_part,"total_TR":total_TR,"dTR":dTR}


        #del alFree
        hdr = io_twixt.parse_twixt_header(filename)
        print(len(hdr))
        print(index)
        #index=5
        alFree = get_specials(hdr, type="alFree",index=index)
        adFree = get_specials(hdr, type="adFree",index=index)
        print(alFree)
        geometry, is3D, orientation, offset = get_volume_geometry(hdr,index=index)

        nb_segments = alFree[4]

        x_FOV = hdr[index]['sSliceArray.asSlice[0].dReadoutFOV']
        y_FOV = hdr[index]['sSliceArray.asSlice[0].dReadoutFOV']
        z_FOV = hdr[index]['sSliceArray.asSlice[0].dThickness']
        nb_part = hdr[index]['sKSpace.lPartitions']

        minTE = hdr[index]["alTE[0]"] / 1e3
        echoSpacing = adFree[1]
        dTR = echoSpacing - minTE
        total_TR = hdr[index]["alTR[0]"] / 1e6
        invTime = adFree[0]

        if np.max(np.argwhere(alFree> 0)) >= 16:
            use_navigator_dll = True
        else:
            use_navigator_dll = False
        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV,"z_FOV": z_FOV, "TI": invTime, "total_TR": total_TR,
                          "dTR": dTR, "is3D": is3D, "orientation": orientation, "nb_part": nb_part, "offset": offset,"use_navigator_dll":use_navigator_dll}
        dico_seqParams.update(geometry)

        

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

    print(dico_seqParams)

    use_navigator_dll = dico_seqParams["use_navigator_dll"]
    nb_segments = dico_seqParams["alFree"][4]

    if use_navigator_dll:
        meas_sampling_mode = dico_seqParams["alFree"][15]
        nb_gating_spokes = dico_seqParams["alFree"][6]
        if not(nb_gating_spokes==0) and (int(nb_segments/nb_gating_spokes)<(nb_segments/nb_gating_spokes)):
            print("Nb segments not divisible by nb_gating_spokes - adjusting nb_gating_spokes")
            nb_gating_spokes+=1
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]
        nb_gating_spokes = 0

    
    undersampling_factor=dico_seqParams["alFree"][9]

    
    nb_part = dico_seqParams["nb_part"]
    nb_slices=int(nb_part)

    dummy_echos = dico_seqParams["alFree"][5]
    nb_rep_center_part = dico_seqParams["alFree"][11]
    nb_part = math.ceil(nb_part / undersampling_factor)
    nb_part_center=int(nb_part/2)
    nb_part = nb_part + nb_rep_center_part - 1
    # print(nb_part)
    # print(nb_rep_center_part)
    # print(nb_gating_spokes)
    # print(dico_seqParams["alFree"])

    del dico_seqParams

    if meas_sampling_mode==1:
        incoherent=False
    elif meas_sampling_mode==2:
        incoherent = True
    elif meas_sampling_mode==3:
        incoherent = True

    if incoherent:
        print("Non Stack-Of-Stars acquisition - 2Dplus1 reconstruction should not be used")

    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
        if 'twix' not in locals():
            print("Re-loading raw data")
            twix = twixtools.read_twix(filename)

        mdb_list = twix[-1]['mdb']
        if nb_gating_spokes == 0:
            data = []

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    data.append(mdb)

        else:
            print("Reading Navigator Data....")
            data_for_nav = []
            data = []
            nav_size_initialized = False
            # k = 0
            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh[14][9]):
                        mdb_data_shape = mdb.data.shape
                        mdb_dtype = mdb.data.dtype
                        nav_size_initialized = True
                        break

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh[14][9]):
                        data.append(mdb)
                    else:
                        data_for_nav.append(mdb)
                        data.append(np.zeros(mdb_data_shape, dtype=mdb_dtype))

                    # print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                    # k += 1
            data_for_nav = np.array([mdb.data for mdb in data_for_nav])
            data_for_nav = data_for_nav.reshape(
                (int(nb_part + dummy_echos), int(nb_gating_spokes)) + data_for_nav.shape[1:])

            if data_for_nav.ndim == 3:
                data_for_nav = np.expand_dims(data_for_nav, axis=-2)
            data_for_nav = data_for_nav[dummy_echos:]
            data_for_nav = np.moveaxis(data_for_nav, -2, 0)
            
            
            if select_first_rep:
                data_for_nav_select_first=np.zeros((data_for_nav.shape[0],nb_part-nb_rep_center_part+1,int(nb_gating_spokes),data_for_nav.shape[-1]),dtype=data_for_nav.dtype)
                data_for_nav_select_first[:,:(nb_part_center+1)]=data_for_nav[:,:(nb_part_center+1)]
                data_for_nav_select_first[:,(nb_part_center+1):]=data_for_nav[:,(nb_part_center+nb_rep_center_part):]
                data_for_nav=data_for_nav_select_first



            np.save(filename_nav_save, data_for_nav)

        data = np.array([mdb.data for mdb in data])
        data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
        data = data[dummy_echos:]
        data = np.moveaxis(data, 2, 0)
        data = np.moveaxis(data, 2, 1)

        if (undersampling_factor > 1)and(not(incoherent)):
            print("Filling kdata for undersampling {}".format(undersampling_factor))
            data_zero_filled=np.zeros((data.shape[0],int(nb_segments),nb_slices,data.shape[-1]),dtype=data.dtype)
            data_zero_filled_shape=data_zero_filled.shape
            data_zero_filled=data_zero_filled.reshape(data.shape[0],-1,8,nb_slices,data.shape[-1])
            data = data.reshape(data.shape[0], -1, 8, nb_part, data.shape[-1])

            curr_start=0

            for sl in range(nb_slices):
                data_zero_filled[:,curr_start::undersampling_factor, :,sl,:] = data[:,curr_start::undersampling_factor, :,int(sl/undersampling_factor),:]
                curr_start = curr_start + 1
                curr_start = curr_start % undersampling_factor

            data=data_zero_filled.reshape(data_zero_filled_shape)

            filename_us_weights = str.split(filename, ".dat")[0] + "_us_weights.npy"
            us_weights=np.zeros((1,int(nb_segments),nb_slices,1))
            us_weights_shape = us_weights.shape
            us_weights = us_weights.reshape(1, -1, 8, nb_slices, 1)
            curr_start = 0
            for sl in range(nb_slices):
                us_weights[:,curr_start::undersampling_factor, :,sl,:] = 1
                curr_start = curr_start + 1
                curr_start = curr_start % undersampling_factor

            us_weights=us_weights.reshape(us_weights_shape)
            np.save(filename_us_weights,us_weights)



        if select_first_rep:
                data_select_first=np.zeros((data.shape[0],int(nb_segments),nb_part-nb_rep_center_part+1,data.shape[-1]),dtype=data.dtype)
                data_select_first[:,:,:(nb_part_center+1)]=data[:,:,:(nb_part_center+1)]
                data_select_first[:,:,(nb_part_center+1):]=data[:,:,(nb_part_center+nb_rep_center_part):]
                data=data_select_first



        


        del mdb_list

        ##################################################
        try:
            del twix
        except:
            pass

        np.save(filename_save, data)

    else:
        data = np.load(filename_save)
        if nb_gating_spokes > 0:
            data_for_nav = np.load(filename_nav_save)

    npoint = data.shape[-1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    print(data.shape)
    if nb_gating_spokes>0:
        print(data_for_nav.shape)

    if dens_adj:
        print("Performing Density Adjustment....")
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(data.ndim - 1)))
        data *= density
    np.save(filename_kdata, data)

    return

@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("ch_opt", int, default=None, description="Optimal Channel For Pilot tone frequency extraction")
@set_parameter("fmin", int, default=230, description="Min frequency For Pilot tone frequency extraction")
@set_parameter("fmax", int, default=290, description="Max frequency For Pilot tone frequency extraction")
@set_parameter("dens_adj", bool, default=True, description="Radial density adjustment")
@set_parameter("suffix",str,default="")
def build_kdata_pilot_tone(filename,ch_opt,fmin,fmax,dens_adj,suffix):

    #Not very clean - global variables to allow for using Multiprocessing Pool inside the function
    global npoint
    global data_chopt
    global data
    global max_slices
    global f_list
    global nb_channels
    global fs_hat
    global nb_allspokes
    global As_hat


    filename_kdata = str.split(filename,".dat") [0]+"_kdata_no_dens_adj{}.npy".format("")
    filename_kdata_pt_corr = str.split(filename,".dat") [0]+"_kdata{}.npy".format("")
    filename_displacement_pt = str.split(filename,".dat") [0]+"_displacement_pt{}.npy".format("")
    filename_Ashat = str.split(filename,".dat") [0]+"_Ashat{}.npy".format("")
    filename_fshat = str.split(filename,".dat") [0]+"_fshat{}.npy".format("")
    filename_save = str.split(filename, ".dat")[0] + ".npy"
    filename_nav_save = str.split(filename, ".dat")[0] + "_nav.npy"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    folder = "/".join(str.split(filename, "/")[:-1])


    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
                                   optional_additional_arrays=["SliceThickness"])

        if np.max(np.argwhere(np.array(twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]) > 0)) >= 16:
            use_navigator_dll = True
        else:
            use_navigator_dll = False

        alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]
        dTR=twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["adFree"][1]-twix[-1]["hdr"]["Meas"]["alTE"][0]/1000
        total_TR=twix[-1]["hdr"]["Meas"]["alTR"][0]/1e6

        nb_part = twix[-1]["hdr"]["Meas"]["Partitions"]

        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV,
                          "use_navigator_dll": use_navigator_dll, "nb_part": nb_part,"total_TR":total_TR,"dTR":dTR}

        del alFree

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    nb_segments = dico_seqParams["alFree"][4]

    if use_navigator_dll:
        meas_sampling_mode = dico_seqParams["alFree"][15]
        nb_gating_spokes = dico_seqParams["alFree"][6]
        if int(nb_segments/nb_gating_spokes)<(nb_segments/nb_gating_spokes):
            print("Nb segments not divisible by nb_gating_spokes - adjusting nb_gating_spokes")
            nb_gating_spokes+=1
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]
        nb_gating_spokes = 0

    

    


    nb_part = dico_seqParams["nb_part"]
    dummy_echos = dico_seqParams["alFree"][5]
    nb_rep_center_part = dico_seqParams["alFree"][11]
    nb_part=nb_part + nb_rep_center_part-1

    nb_part = int(nb_part)

    del dico_seqParams

    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
        if 'twix' not in locals():
            print("Re-loading raw data")
            twix = twixtools.read_twix(filename)

        mdb_list = twix[-1]['mdb']
        if nb_gating_spokes == 0:
            data = []

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    data.append(mdb)

        else:
            print("Reading Navigator Data....")
            data_for_nav = []
            data = []
            nav_size_initialized = False
            # k = 0
            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh[14][9]):
                        mdb_data_shape = mdb.data.shape
                        mdb_dtype = mdb.data.dtype
                        nav_size_initialized = True
                        break

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh[14][9]):
                        data.append(mdb)
                    else:
                        data_for_nav.append(mdb)
                        data.append(np.zeros(mdb_data_shape, dtype=mdb_dtype))

                    # print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                    # k += 1
            data_for_nav = np.array([mdb.data for mdb in data_for_nav])
            data_for_nav = data_for_nav.reshape(
                (int(nb_part + dummy_echos), int(nb_gating_spokes)) + data_for_nav.shape[1:])

            if data_for_nav.ndim == 3:
                data_for_nav = np.expand_dims(data_for_nav, axis=-2)
            data_for_nav = data_for_nav[dummy_echos:]
            data_for_nav = np.moveaxis(data_for_nav, -2, 0)
            np.save(filename_nav_save, data_for_nav)

        data = np.array([mdb.data for mdb in data])
        data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
        data = data[dummy_echos:]
        data = np.moveaxis(data, 2, 0)
        data = np.moveaxis(data, 2, 1)

        del mdb_list

        ##################################################
        try:
            del twix
        except:
            pass

        np.save(filename_save, data)

    else:
        data = np.load(filename_save)
        if nb_gating_spokes > 0:
            data_for_nav = np.load(filename_nav_save)

    npoint = data.shape[-1]
    nb_channels=data.shape[0]
    nb_allspokes=data.shape[1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    np.save(filename_kdata, data)

    fig,ax=plt.subplots(6,6)
    axli=ax.flatten()
    all_radial_proj_all_ch_no_corr=[]
    sl=int(nb_part/2)

    from scipy.stats import kurtosis

    print("Drawing radial projection figures")

    for ch in range(nb_channels):
        kdata_PT=data[ch]
        #traj_PT=radial_traj.get_traj().reshape(nb_allspokes,nb_slices,-1,3)

        #finufft.nufft3d1(traj_PT[0][0],kdata_PT[0][0])

        all_radial_proj=[]

        for j in range(nb_allspokes):
            radial_proj=np.abs((np.fft.fft((kdata_PT[j][sl]))))
            all_radial_proj.append(radial_proj)

        all_radial_proj=np.array(all_radial_proj)



        axli[ch].plot(all_radial_proj[::27].T)
        axli[ch].set_title(ch)
        all_radial_proj_all_ch_no_corr.append(all_radial_proj)

        #all_kurt.append(np.mean(kurtosis(all_radial_proj[1::28],axis=-1)))
    
    plt.savefig(str.split(filename,".dat") [0]+"_radial_proj_allchannels.jpg")

   
    
    print("Channels sorted by radial projection kurtosis")
    print(np.argsort(np.mean(kurtosis(np.array(all_radial_proj_all_ch_no_corr)[:,1::28],axis=-1),axis=1))[::-1])

    if ch_opt is None:
        ch_opt=np.argsort(np.mean(kurtosis(np.array(all_radial_proj_all_ch_no_corr)[:,1::28],axis=-1),axis=1))[-1]

    plt.figure()
    plt.imshow(all_radial_proj_all_ch_no_corr[ch_opt])
    plt.savefig(str.split(filename,".dat") [0]+"_radial_proj_allspokes_chopt.jpg")


    print("Extracting pilot tone frequency")

    max_slices=int(nb_part)
    fs_hat=np.zeros((nb_allspokes,max_slices))

    f_list = np.expand_dims(np.arange(fmin, fmax, 0.01), axis=(1, 2))


    

    from multiprocessing import Pool

    if str.split(filename_fshat, "/")[-1] not in os.listdir(folder):

        data_chopt=data[ch_opt]

        with Pool(24) as pool:
            # perform calculations
            results = pool.map(task, range(nb_allspokes))


        for ts in tqdm(range(nb_allspokes)):
            fs_hat[ts, :] =np.load("./fshat/fshat_{}.npy".format(ts))
        np.save(filename_fshat,fs_hat)
    else:
        fs_hat=np.load(filename_fshat)

    print("Calculating Pilot Tone Amplitude Ashat")


    kdata_with_pt_corrected=np.zeros(data.shape,dtype=data.dtype)
    As_hat=np.zeros((nb_channels,nb_allspokes,max_slices))
    
    with Pool(24) as pool:
        # perform calculations
        results = pool.map(task_Ashat, range(nb_allspokes))


    for ts in tqdm(range(nb_allspokes)):
        As_hat[:,ts,:] =np.load("./fshat/Ashat_{}.npy".format(ts))
    for ts in tqdm(range(nb_allspokes)):
        kdata_with_pt_corrected[:,ts,:] =np.load("./fshat/kdata_pt_corrected_{}.npy".format(ts))

    As_hat[:,::28,:]=As_hat[:,1::28,:]

    np.save(filename_Ashat,As_hat)


    if dens_adj:
        print("Performing Density Adjustment....")
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_with_pt_corrected.ndim - 1)))
        kdata_with_pt_corrected *= density

    np.save(filename_kdata_pt_corr,kdata_with_pt_corrected)


    print("Drawing radial projection for all spokes non corrected and corrected")
    plt.close("all")
    sl = np.random.randint(max_slices)
    all_radial_proj_all_ch=[]
    for ch in range(nb_channels):
        kdata_PT=kdata_with_pt_corrected[ch]

        all_radial_proj=[]

        for j in range(nb_allspokes):
            radial_proj=np.abs(np.fft.fft(kdata_PT[j,sl]))
            all_radial_proj.append(radial_proj)

        all_radial_proj=np.array(all_radial_proj)

        all_radial_proj_all_ch.append(all_radial_proj)

    kdata_all_channels_all_slices=np.load(filename_kdata)
    all_radial_proj_all_ch_no_corr=[]
    for ch in range(nb_channels):
        kdata_PT=kdata_all_channels_all_slices[ch]
        #finufft.nufft3d1(traj_PT[0][0],kdata_PT[0][0])

        all_radial_proj=[]

        for j in range(nb_allspokes):
            radial_proj=np.abs((np.fft.fft((kdata_PT[j][sl]))))
            all_radial_proj.append(radial_proj)

        all_radial_proj=np.array(all_radial_proj)

        all_radial_proj_all_ch_no_corr.append(all_radial_proj)



    ch=np.random.randint(nb_channels)
    #ch=ch_opt
    #ch=6
    #ch=33
    print(ch)
    plt.figure()
    plt.title("Radial Projection Corrected from PT ch {} sl {}".format(ch,sl))
    plt.imshow(all_radial_proj_all_ch[ch],vmin=0,vmax=0.001)
    plt.savefig(str.split(filename,".dat") [0]+"_radial_proj_allspokes_ch{}_corrected.jpg".format(ch))

    plt.figure()
    plt.title("Radial Projection ch {} sl {}".format(ch,sl))
    plt.imshow(all_radial_proj_all_ch_no_corr[ch],vmin=0,vmax=0.001)
    plt.savefig(str.split(filename,".dat") [0]+"_radial_proj_allspokes_ch{}.jpg".format(ch))

    plt.close("all")


    print("Extracting displacement from Pilot Tone amplitude")
    from scipy.signal import savgol_filter
    from statsmodels.nonparametric.smoothers_lowess import lowess

    

    with Pool(24) as pool:
        # perform calculations
        results = pool.map(task_Ashat_filtered, range(nb_channels))

    As_hat_normalized=np.zeros(As_hat.shape)
    As_hat_filtered=np.zeros(As_hat.shape)

    for ch in tqdm(range(nb_channels)):
        As_hat_filtered[ch] =np.load("./fshat/Ashat_filtered_ch{}.npy".format(ch))
        As_hat_normalized[ch] = np.load("./fshat/As_hat_normalized_ch{}.npy".format(ch))


    data_for_pca=np.moveaxis(As_hat_filtered,-1,-2)
    data_for_pca=data_for_pca.reshape(nb_channels,-1)

    from sklearn.decomposition import PCA
    pca=PCA(n_components=1)
    pca.fit(data_for_pca.T)
    pcs=pca.components_@data_for_pca
    movement_all_slices=pcs[0]

    np.save(filename_displacement_pt,movement_all_slices.reshape(nb_part,nb_allspokes))

    return


@machine
@set_parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("nspokes_per_z_encoding", int, default=8, description="Number of spokes per z encoding")
@set_parameter("nb_ref_lines", int, default=24, description="Number of reference lines for Grappa calibration")
@set_parameter("suffix",str,default="")
def build_data_for_grappa_simulation(filename,undersampling_factor,nspokes_per_z_encoding,nb_ref_lines,suffix):

    filename_kdata = str.split(filename, ".dat")[0] + suffix + "_kdata.npy"
    filename_save = str.split(filename, ".dat")[0] + ".npy"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"
    filename_save_us = str.split(filename, ".dat")[0] + "_us{}ref{}_us.npy".format(undersampling_factor, nb_ref_lines)
    filename_save_calib = str.split(filename, ".dat")[0] + "_us{}ref{}_calib.npy".format(undersampling_factor,
                                                                                         nb_ref_lines)


    folder = "/".join(str.split(filename, "/")[:-1])

    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
                                   optional_additional_arrays=["SliceThickness"])

        alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

        nb_slices = twix[-1]["hdr"]["Meas"]["lPartitions"]

        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV, "US": undersampling_factor,
                          "nb_slices": nb_slices, "nb_ref_lines": nb_ref_lines}

        del alFree

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)

    nb_segments = dico_seqParams["alFree"][4]
    nb_slices = dico_seqParams["nb_slices"]
    del dico_seqParams

    if str.split(filename_save_us, "/")[-1] not in os.listdir(folder):

        if str.split(filename_save, "/")[-1] not in os.listdir(folder):
            if 'twix' not in locals():
                print("Re-loading raw data")
                twix = twixtools.read_twix(filename)
            mdb_list = twix[-1]['mdb']
            data = []

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    data.append(mdb)

            data = np.array([mdb.data for mdb in data])
            data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
            data = np.moveaxis(data, 2, 0)
            data = np.moveaxis(data, 2, 1)

            try:
                del twix
            except:
                pass

            np.save(filename_save, data)

        else:
            data = np.load(filename_save)

        data = groupby(data, nspokes_per_z_encoding, axis=1)
        # data=np.array(data)

        all_lines = np.arange(nb_slices).astype(int)

        shift = 0
        data_calib = []

        for ts in tqdm(range(len(data))):
            lines_measured = all_lines[shift::undersampling_factor]
            lines_to_estimate = list(set(all_lines) - set(lines_measured))
            center_line = int(nb_slices / 2)
            lines_ref = all_lines[(center_line - int(nb_ref_lines / 2)):(center_line + int(nb_ref_lines / 2))]

            data_calib.append(data[ts][:, :, lines_ref, :])
            data[ts] = data[ts][:, :, lines_measured, :]
            shift += 1
            shift = shift % (undersampling_factor)

        data_calib = np.array(data_calib)
        data = np.array(data)

        data = np.moveaxis(data, 0, 1)
        data_calib = np.moveaxis(data_calib, 0, 1)

        data = data.reshape(data.shape[0], -1, data.shape[-2], data.shape[-1])
        data_calib = data_calib.reshape(data_calib.shape[0], -1, data_calib.shape[-2], data_calib.shape[-1])

        np.save(filename_save_us, data)
        np.save(filename_save_calib, data_calib)


    try:
        del twix
    except:
        pass


    return


@machine
@set_parameter("filename_save_calib", str, default=None, description="Calib data .npy file")
@set_parameter("nspokes_per_z_encoding", int, default=8, description="Number of spokes per z encoding")
@set_parameter("len_kery", int, default=2, description="Number of lines in each kernel")
@set_parameter("len_kerx", int, default=7, description="Number of readout points in each line of the kernel")
@set_parameter("calibration_mode", ["Standard","Tikhonov"], default="Standard", description="Calibration methodology for Grappa")
@set_parameter("lambd", float, default=None, description="Regularization parameter")
@set_parameter("suffix", str, default="")
def calib_and_estimate_kdata_grappa(filename_save_calib, nspokes_per_z_encoding,len_kery,len_kerx,calibration_mode,lambd,suffix):
    data_calib = np.load(filename_save_calib)
    nb_channels=data_calib.shape[0]

    filename_split = str.split(filename_save_calib,"_us")
    filename = filename_split[0]+".dat"
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    filename_split_params = filename_split[1].split("ref")
    undersampling_factor = int(filename_split_params[0])
    nb_ref_lines=int(filename_split_params[1].split("_")[0])

    filename_save_us = str.split(filename, ".dat")[0] + "_us{}ref{}_us.npy".format(undersampling_factor, nb_ref_lines)

    if calibration_mode == "Tikhonov":
        if lambd is None:
            lambd = 0.01
            print("Warning : lambd was not set for Tikhonov calibration. Using default value of 0.01")

        str_lambda=str(lambd)
        str_lambda=str.split(str_lambda,".")
        str_lambda="_".join(str_lambda)
        filename_kdata_grappa = str.split(filename, ".dat")[0] + "_{}_{}_us{}ref{}_kdata_grappa.npy".format(calibration_mode,str_lambda,undersampling_factor,
                                                                                                  nb_ref_lines)

        filename_currtraj_grappa = str.split(filename, ".dat")[0] + "_{}_{}_us{}ref{}_currtraj_grappa.npy".format(calibration_mode,str_lambda,
            undersampling_factor, nb_ref_lines)

    else:
        filename_kdata_grappa = str.split(filename, ".dat")[0] + "_{}__us{}ref{}_kdata_grappa.npy".format(
            undersampling_factor,
            nb_ref_lines)

        filename_currtraj_grappa = str.split(filename, ".dat")[0] + "_us{}ref{}_currtraj_grappa.npy".format(
            undersampling_factor, nb_ref_lines)

    print(filename_kdata_grappa)

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    data = np.load(filename_save_us)

    indices_kz = list(range(1, (undersampling_factor)))

    kernel_y = np.arange(len_kery)
    kernel_x = np.arange(-(int(len_kerx / 2) - 1), int(len_kerx / 2))

    pad_y = ((np.array(kernel_y) < 0).sum(), (np.array(kernel_y) > 0).sum())
    pad_x = ((np.array(kernel_x) < 0).sum(), (np.array(kernel_x) > 0).sum())

    data_calib_padded = np.pad(data_calib, ((0, 0), (0, 0), pad_y, pad_x), mode="constant")
    data_padded = np.pad(data, ((0, 0), (0, 0), pad_y, pad_x), mode="constant")

    kdata_all_channels_completed_all_ts = []
    curr_traj_completed_all_ts = []

    replace_calib_lines = True
    useGPU = True

    shift=0

    nb_segments = dico_seqParams["alFree"][4]
    nb_allspokes= nb_segments
    npoint=data.shape[-1]
    nb_slices = dico_seqParams["nb_slices"]

    meas_sampling_mode = dico_seqParams["alFree"][12]
    if meas_sampling_mode == 1:
        incoherent = False
        mode = None
    elif meas_sampling_mode == 2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode == 3:
        incoherent = True
        mode = "new"

    radial_traj_all = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint, nb_slices=nb_slices,
                               incoherent=incoherent, mode=mode, nspoke_per_z_encoding=nspokes_per_z_encoding)
    traj_all = radial_traj_all.get_traj().reshape(nb_allspokes, nb_slices, npoint, 3)

    all_lines = np.arange(nb_slices).astype(int)


    for ts in tqdm(range(nb_allspokes)):

        if ((ts) % nspokes_per_z_encoding == 0) and (not (ts == 0)):
            shift += 1
            shift = shift % (undersampling_factor)

        lines_measured = all_lines[shift::undersampling_factor]
        lines_to_estimate = list(set(all_lines) - set(lines_measured))
        center_line = int(nb_slices / 2)
        lines_ref = all_lines[(center_line - int(nb_ref_lines / 2)):(center_line + int(nb_ref_lines / 2))]

        calib_lines = []
        calib_lines_ref_indices = []
        for i in indices_kz:
            current_calib_lines = []
            current_calib_lines_ref_indices = []

            for j in range(undersampling_factor * (kernel_y[0]) + i,
                           nb_ref_lines - (kernel_y[-1] * undersampling_factor - i)):
                current_calib_lines.append(lines_ref[j])
                current_calib_lines_ref_indices.append(j)
            calib_lines.append(current_calib_lines)
            calib_lines_ref_indices.append(current_calib_lines_ref_indices)

        calib_lines = np.array(calib_lines).astype(int)
        calib_lines_ref_indices = np.array(calib_lines_ref_indices).astype(int)

        weights = np.zeros((len(calib_lines), nb_channels, nb_channels * len(kernel_x) * len(kernel_y)),
                           dtype=data_calib.dtype)

        for index_ky_target in range(len(calib_lines_ref_indices)):
            calib_lines_ref_index = calib_lines_ref_indices[index_ky_target]
            F_target_calib = data_calib[:, ts, calib_lines_ref_indices[index_ky_target], :]  # .reshape(nb_channels,-1)
            F_source_calib = np.zeros((nb_channels * len(kernel_x) * len(kernel_y),) + F_target_calib.shape[1:],
                                      dtype=F_target_calib.dtype)

            for i, l in enumerate(calib_lines_ref_indices[index_ky_target]):
                l0 = l - (index_ky_target + 1)
                for j in range(npoint):
                    local_indices = np.stack(np.meshgrid(pad_y[0] + l0 + undersampling_factor * np.array(kernel_y),
                                                         pad_x[0] + j + np.array(kernel_x)), axis=0).T.reshape(-1, 2)
                    F_source_calib[:, i, j] = data_calib_padded[:, ts, local_indices[:, 0],
                                              local_indices[:, 1]].flatten()

            F_source_calib = F_source_calib.reshape(nb_channels * len(kernel_x) * len(kernel_y), -1)
            F_target_calib = F_target_calib.reshape(nb_channels, -1)

            if calibration_mode == "Tikhonov":
                F_source_calib_cupy = cp.asarray(F_source_calib)
                F_target_calib_cupy = cp.asarray(F_target_calib)
                weights[index_ky_target] = (F_target_calib_cupy @ F_source_calib_cupy.conj().T @ cp.linalg.inv(
                    F_source_calib_cupy @ F_source_calib_cupy.conj().T + lambd * cp.eye(
                        F_source_calib_cupy.shape[0]))).get()

            else:
                weights[index_ky_target] = (
                            cp.asarray(F_target_calib) @ cp.linalg.pinv(cp.asarray(F_source_calib))).get()


        F_target_estimate = np.zeros((nb_channels, len(lines_to_estimate), npoint), dtype=data.dtype)

        for i, l in (enumerate(lines_to_estimate)):
            F_source_estimate = np.zeros((nb_channels * len(kernel_x) * len(kernel_y), npoint),
                                         dtype=data_calib.dtype)
            index_ky_target = (l) % (undersampling_factor) - 1
            try:
                l0 = np.argwhere(np.array(lines_measured - shift) == l - (index_ky_target + 1))[0][0]
            except:
                print("Line {} for Ts {} not estimated".format(l, ts))
                continue
            for j in range(npoint):
                local_indices = np.stack(
                    np.meshgrid(pad_y[0] + l0 + np.array(kernel_y), pad_x[0] + j + np.array(kernel_x)),
                    axis=0).T.reshape(-1, 2)
                F_source_estimate[:, j] = data_padded[:, ts, local_indices[:, 0], local_indices[:, 1]].flatten()
            F_target_estimate[:, i, :] = weights[index_ky_target] @ F_source_estimate

        F_target_estimate[:,:,:pad_x[0]]=0
        F_target_estimate[:, :, -pad_x[1]:] = 0

        max_measured_line=np.max(lines_measured)
        min_measured_line=np.min(lines_measured)

        for i, l in (enumerate(lines_to_estimate)):
            if (l<min_measured_line)or(l>max_measured_line):
                F_target_estimate[:, i, :]=0

        for i, l in (enumerate(lines_to_estimate)):
            if l in lines_ref.flatten():
                ind_line = np.argwhere(lines_ref.flatten() == l).flatten()[0]
                F_target_estimate[:, i, :] = data_calib[:, ts, ind_line, :]

        kdata_all_channels_completed = np.concatenate([data[:, ts], F_target_estimate], axis=1)
        curr_traj_estimate = traj_all[ts, lines_to_estimate, :, :]
        curr_traj_completed = np.concatenate([traj_all[ts,lines_measured,:,:], curr_traj_estimate],
                                             axis=0)

        kdata_all_channels_completed = kdata_all_channels_completed.reshape((nb_channels, -1))
        curr_traj_completed = curr_traj_completed.reshape((-1, 3))

        kdata_all_channels_completed_all_ts.append(kdata_all_channels_completed)
        curr_traj_completed_all_ts.append(curr_traj_completed)

    kdata_all_channels_completed_all_ts = np.array(kdata_all_channels_completed_all_ts)
    kdata_all_channels_completed_all_ts = np.moveaxis(kdata_all_channels_completed_all_ts, 0, 1)
    curr_traj_completed_all_ts = np.array(curr_traj_completed_all_ts)

    for ts in tqdm(range(curr_traj_completed_all_ts.shape[0])):
        ind = np.lexsort((curr_traj_completed_all_ts[ts][:, 0], curr_traj_completed_all_ts[ts][:, 2]))
        curr_traj_completed_all_ts[ts] = curr_traj_completed_all_ts[ts, ind, :]
        kdata_all_channels_completed_all_ts[:, ts] = kdata_all_channels_completed_all_ts[:, ts, ind]

    kdata_all_channels_completed_all_ts = kdata_all_channels_completed_all_ts.reshape(nb_channels, nb_allspokes,
                                                                                      nb_slices, npoint)
    #kdata_all_channels_completed_all_ts[:, :, :, :pad_x[0]] = 0
    #kdata_all_channels_completed_all_ts[:, :, :, -pad_x[1]:] = 0
    #kdata_all_channels_completed_all_ts[:, :, :((undersampling_factor - 1) * pad_y[0]), :] = 0
    #kdata_all_channels_completed_all_ts[:, :, (-(undersampling_factor - 1) * pad_y[1]):, :] = 0

    np.save(filename_kdata_grappa, kdata_all_channels_completed_all_ts)
    np.save(filename_currtraj_grappa, curr_traj_completed_all_ts)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("dens_adj", bool, default=False, description="Memory usage")
@set_parameter("suffix",str,default="")
@set_parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
def build_coil_sensi(filename_kdata,filename_traj,sampling_mode,undersampling_factor,dens_adj,suffix,nb_rep_center_part):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = ("_b1"+suffix).join(str.split(filename_kdata, "_kdata"))

    filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    if use_navigator_dll:
        meas_sampling_mode = dico_seqParams["alFree"][15]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]

    undersampling_factor=dico_seqParams["alFree"][9]

    nb_slices = int(dico_seqParams["nb_part"])

    if meas_sampling_mode==1:
        incoherent=False
        mode = None
    elif meas_sampling_mode==2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode==3:
        incoherent = True
        mode = "new"


    if nb_rep_center_part>1:
        kdata_all_channels_all_slices=kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part)

    # if sampling_mode_list[0]=="stack":
    #     incoherent=False
    # else:
    #     incoherent=True

    # if len(sampling_mode_list)>1:
    #     mode=sampling_mode_list[1]
    # else:
    #     mode="old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts = np.load(filename_traj)
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    res = 16
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                 useGPU=False, light_memory_usage=True,density_adj=dens_adj,hanning_filter=True)

    image_file=str.split(filename_b1, ".npy")[0] + suffix + ".jpg"

    sl = int(b1_all_slices.shape[1]/2)

    list_images=list(np.abs(b1_all_slices[:,sl,:,:]))
    plot_image_grid(list_images,(6,6),title="Sensivitiy map for slice".format(sl),save_file=image_file)

    np.save(filename_b1, b1_all_slices)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("ntimesteps", int, default=175, description="Number of timesteps for the image serie")
@set_parameter("use_GPU", bool, default=True, description="Use GPU")
@set_parameter("light_mem", bool, default=True, description="Memory usage")
@set_parameter("dens_adj", bool, default=False, description="Density adjustment for radial data")
@set_parameter("suffix",str,default="")
def build_volumes(filename_kdata,filename_traj,sampling_mode,undersampling_factor,ntimesteps,use_GPU,light_mem,dens_adj,suffix):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))

    b1_all_slices=np.load(filename_b1)
    filename_volume = ("_volumes"+suffix).join(str.split(filename_kdata, "_kdata"))
    print(filename_volume)
    sampling_mode_list = str.split(sampling_mode, "_")

    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]*undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts=np.load(filename_traj)
        radial_traj=Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj, image_size,
                                                            b1=b1_all_slices, density_adj=dens_adj, ntimesteps=ntimesteps,
                                                            useGPU=use_GPU, normalize_kdata=False, memmap_file=None,
                                                            light_memory_usage=light_mem,normalize_volumes=True,normalize_iterative=True)
    np.save(filename_volume, volumes_all)


    gif=[]
    sl=int(volumes_all.shape[1]/2)
    volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    for i in range(volume_for_gif.shape[0]):
        img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
        img=img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_nav_save", str, default=None, description="Saved K-space navigator data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("light_mem", bool, default=True, description="Memory usage")
@set_parameter("dens_adj", bool, default=False, description="Density adjustment for radial data")
@set_parameter("suffix",str,default="")
def build_data_autofocus(filename_kdata,filename_traj,filename_nav_save,sampling_mode,undersampling_factor,light_mem,dens_adj,suffix):
    # b1_all_slices=b1_full

    print("Loading Files...")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    print("Loading coil sensi")
    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))
    b1_all_slices = np.load(filename_b1)

    print("Parsing parameters")
    filename_kdata_modif = ("_kdata_modif" + suffix).join(str.split(filename_kdata, "_kdata"))
    #print(filename_volume)
    sampling_mode_list = str.split(sampling_mode, "_")

    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_segments = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2] * undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    print(data_shape)

    print("Processing Nav Data...")
    data_for_nav = np.load(filename_nav_save)

    print(data_for_nav.shape)

    nb_allspokes = nb_segments
    nb_slices = data_for_nav.shape[1]
    nb_gating_spokes=data_for_nav.shape[2]
    nb_channels = data_for_nav.shape[0]
    npoint_for_nav = data_for_nav.shape[-1]


    all_timesteps = np.arange(nb_allspokes)
    nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_for_nav, nb_slices=nb_slices,
                           applied_timesteps=list(nav_timesteps))

    nav_image_size = (npoint_for_nav,)

    print("Rebuilding Nav Images...")
    #print("Calculating Sensitivity Maps for Nav Images...")
    #b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
    #b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

    #images_nav = np.abs(
    #    simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1=b1_nav_mean))

    best_ch=9
    images_nav = np.abs(simulate_nav_images_multi(np.expand_dims(data_for_nav[best_ch],axis=0), nav_traj, nav_image_size, b1=None))




    #print("Estimating Movement...")
    shifts = list(range(-30, 30))
    #bottom = -shifts[0]
    #top = nav_image_size[0]-shifts[-1]

    bottom = 40
    top=120
    displacements = calculate_displacement(images_nav, bottom, top, shifts, lambda_tv=0.00)

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts=np.load(filename_traj)
        radial_traj=Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    spoke_groups = np.argmin(np.abs(
        np.arange(0, nb_segments * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_slices,
                                                                          nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                  -1)),
        axis=-1)

    if not (nb_segments == nb_gating_spokes):
        spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
        spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_segments / nb_gating_spokes / 2) + 1:] - 1
        spoke_groups = spoke_groups.flatten()

    displacements_extrapolated = np.array([displacements[j] for j in spoke_groups])

    #x = np.arange(-1.0, 1.01, 0.1)
    #y = np.array([0])
    #z = np.arange(-0.5, 0.51, 0.1)

    #print("Calculating Entropy for all movement ranges...")
    #entropy_all = []
    #ent_min = np.inf
    #for dx in tqdm(x):
    #    entropy_xzy = []
    #    for dz in z:
    #        entropy_zy=[]
    #        for dy in y:
    #            alpha = np.array([dx, dy, dz])
    #            dr = np.expand_dims(alpha, axis=(0, 1)) * np.expand_dims(
    #                displacements_extrapolated.reshape(nb_slices, nb_segments).T, axis=(2))
    #            modif = np.exp(
    #                1j * np.sum((radial_traj.get_traj().reshape(nb_segments, -1, npoint, 3) * np.expand_dims(dr, axis=2)),
    #                            axis=-1))
    #            data_modif = kdata_all_channels_all_slices * modif
    #            volume_full_modif = \
    #                simulate_radial_undersampled_images_multi(data_modif, radial_traj, image_size, b1=b1_all_slices,
    #                                                          density_adj=dens_adj, ntimesteps=1, useGPU=False,
    #                                                          normalize_kdata=False, memmap_file=None,
    #                                                          light_memory_usage=light_mem,
    #                                                          normalize_volumes=True)[0]
    #            ent = calc_grad_entropy(volume_full_modif)
    #            entropy_zy.append(ent)
    #            if ent < ent_min:
    #                modif_final = modif
    #                alpha_min = alpha
    #                ent_min = ent

    #        entropy_xzy.append(entropy_zy)
    #    entropy_all.append(entropy_xzy)

    #print("Alpha min: {}".format(alpha_min))

    #filename_entropy = str.split(filename_kdata, ".npy")[0] + "_entropy.npy"
    #np.save(filename_entropy,np.array(entropy_all))

    ###########################
    #X, Y = np.meshgrid(x, y)


    #fig = plt.figure(figsize=(15, 15))
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, np.array(entropy_all), rstride=1, cstride=1, cmap='hot', linewidth=0,
    #                       antialiased=False)

    #filename_entropy_image = str.split(filename_kdata, ".npy")[0] + "_entropy.jpg"
    #plt.savefig(filename_entropy_image)
    ##################################""

    alpha = np.array([1, 0, 0])
    dr = np.expand_dims(alpha, axis=(0, 1)) * np.expand_dims(
                    displacements_extrapolated.reshape(nb_slices, nb_segments).T, axis=(2))
    modif_final = np.exp(
                    1j * np.sum((radial_traj.get_traj().reshape(nb_segments, -1, npoint, 3) * np.expand_dims(dr, axis=2)),
                                axis=-1))

    np.save(filename_kdata_modif, kdata_all_channels_all_slices*modif_final)


    # gif=[]
    # sl=int(volumes_all.shape[1]/2)
    # volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    # for i in range(volume_for_gif.shape[0]):
    #     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    #     img=img.convert("P")
    #     gif.append(img)
    #
    # filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    # gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return



@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_df_groups_global", str, default=None, description="Number of spokes per cycle per acquisition")
@set_parameter("filename_categories_global", str, default=None, description="Cycle for each spoke for each acquisition")
@set_parameter("nb_gating_spokes", int, default=50, description="Number of gating spokes per repetition")
@set_parameter("files_config", type=Config, default=None, description="Kdata filenames to use for aggregation")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")


def aggregate_kdata(filename_kdata,filename_df_groups_global,filename_categories_global,files_config,nb_gating_spokes,undersampling_factor):
    # b1_all_slices=b1_full

    print("Loading Files...")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    folder = "/".join(str.split(filename_kdata, "/")[:-1])
    base_folder="/".join(str.split(filename_kdata, "/")[:-2])

    bin_width=int(str.split(filename_categories_global, "bw")[-1][:-4])



    data_shape = kdata_all_channels_all_slices.shape
    nb_channels=data_shape[0]
    nb_segments = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2] * undersampling_factor
    nb_part = nb_slices

    print(data_shape)
    print(folder)

    files = files_config["files"]
    print(files)

    filename_kdata_final = base_folder + str.split(files[0], "_1.dat")[0] + "_bw{}_aggregated_kdata.npy".format(
        bin_width)

    categories_global = np.load(filename_categories_global)
    df_groups_global = pd.read_pickle(filename_df_groups_global)

    idx_cat = df_groups_global.displacement.idxmax()

    kdata_final = np.zeros(data_shape, dtype=kdata_all_channels_all_slices.dtype)
    del kdata_all_channels_all_slices
    count = np.zeros((nb_segments, nb_slices))

    for i, localfile in tqdm(enumerate(files)):

        filename = base_folder + localfile

        retained_nav_spokes = (categories_global[i] == idx_cat)

        retained_nav_spokes_index = np.argwhere(retained_nav_spokes).flatten()
        spoke_groups = np.argmin(np.abs(
            np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                              nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                      -1)),
                                 axis=-1)

        if not (nb_segments == nb_gating_spokes):
            spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
            spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
                nb_segments / nb_gating_spokes / 2) + 1:] - 1
            spoke_groups = spoke_groups.flatten()

        included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])

        included_spokes[::int(nb_segments / nb_gating_spokes)] = False
        included_spokes = included_spokes.reshape(nb_slices, nb_segments)
        included_spokes = included_spokes.T

        filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")

        print("Loading kdata for file {}".format(localfile))
        kdata_all_channels_all_slices = np.load(filename_kdata)
        #print(included_spokes.shape)
        kdata_all_channels_all_slices = kdata_all_channels_all_slices.reshape(nb_channels, nb_segments, nb_slices,
                                                                              npoint)
        print("Aggregating kdata for file {}".format(localfile))
        for ch in tqdm(range(nb_channels)):
            kdata_final[ch] += (np.expand_dims(1 * included_spokes, axis=-1)) * kdata_all_channels_all_slices[ch]
        count += (1 * included_spokes)

    count[count == 0] = 1

    print("Normalizing Final Kdata")
    for ch in tqdm(range(nb_channels)):
        kdata_final[ch] /= np.expand_dims(count, axis=(-1))

    np.save(filename_kdata_final, kdata_final)


    # gif=[]
    # sl=int(volumes_all.shape[1]/2)
    # volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    # for i in range(volume_for_gif.shape[0]):
    #     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    #     img=img.convert("P")
    #     gif.append(img)
    #
    # filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    # gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return

@machine
@set_parameter("file_deformation", str, default=None, description="Deformation map file")
@set_parameter("gr", int, default=4, description="Motion state")
@set_parameter("sl", int, default=None, description="Slice")
def plot_deformation(file_deformation,gr,sl):
    deformation_map=np.load(file_deformation)[:,gr,sl]
    file_deformation_plot=str.split(file_deformation,".npy")[0]+"_gr{}sl{}.jpg".format(gr,sl)
    plot_deformation_map(deformation_map,save_file=file_deformation_plot)


@machine
@set_parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@set_parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@set_parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@set_parameter("dens_adj", bool, default=False, description="Memory usage")
@set_parameter("threshold", float, default=None, description="Threshold for mask")
@set_parameter("suffix",str,default="")
@set_parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
def build_mask(filename_kdata,filename_traj,sampling_mode,undersampling_factor,dens_adj,threshold,suffix,nb_rep_center_part):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    if nb_rep_center_part>1:
        kdata_all_channels_all_slices=kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part)


    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))

    b1_all_slices=np.load(filename_b1)

    filename_mask =("_mask"+suffix).join(str.split(filename_kdata, "_kdata"))
    print(filename_mask)

    sampling_mode_list = str.split(sampling_mode, "_")

    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]*undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts = np.load(filename_traj)
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    selected_spokes=np.r_[10:400] 
    selected_spokes=None 
    mask = build_mask_single_image_multichannel(kdata_all_channels_all_slices, radial_traj, image_size,
                                                b1=b1_all_slices, density_adj=dens_adj, threshold_factor=threshold,
                                                normalize_kdata=False, light_memory_usage=True,selected_spokes=selected_spokes,normalize_volumes=True)


    np.save(filename_mask, mask)

    gif = []

    for i in range(mask.shape[0]):
        img = Image.fromarray(np.uint8(mask[i] / np.max(mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@machine
@set_parameter("filename_volume", str, default=None, description="Singular volumes")
@set_parameter("l", int, default=0, description="Singular volume number for mask calculation")
@set_parameter("threshold", float, default=None, description="Threshold for mask")
@set_parameter("it", int, default=3, description="Binary closing iterations")
def build_mask_from_singular_volume(filename_volume,l,threshold,it):
    filename_mask="".join(filename_volume.split(".npy"))+"_l{}_mask.npy".format(l)
    volumes=np.load(filename_volume)
    print(volumes.shape)
    if volumes.ndim==4:
        volume=volumes[l]
        mask=build_mask_from_volume(volume,threshold,it)
    elif volumes.ndim==5:
        print("Aggregating mask from all respiratory motions")
        volume_allbins=volumes[:,l]
        nb_bins=volume_allbins.shape[0]
        mask=False

        for gr in range(nb_bins):
            volume=volume_allbins[gr]
            current_mask=build_mask_from_volume(volume,threshold,it)
            mask=mask|current_mask



    elif volumes.ndim==3:
        print("Singular volume number l not used - input dim was 3")
        filename_mask="".join(filename_volume.split(".npy"))+"_mask.npy"
        mask=build_mask_from_volume(volume,threshold,it)
    else:
        raise ValueError("Volume number of dimensions should be 3 or 4 or 5")

    np.save(filename_mask,mask)

    gif = []

    for i in range(mask.shape[0]):
        img = Image.fromarray(np.uint8(mask[i] / np.max(mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@machine
@set_parameter("filename_mask", str, default=None, description="Mask")
def build_mask_full_from_mask(filename_mask):
    filename_mask_full=filename_mask.split("_mask.npy")[0]+"_mask_full.npy"
    mask=np.load(filename_mask)

    new_mask=np.zeros_like(mask)
    # for sl in range(mask.shape[0]):
    #     x_min=np.min(np.argwhere(mask[sl]>0)[:,0])
    #     y_min=np.min(np.argwhere(mask[sl]>0)[:,1])

    #     x_max=np.max(np.argwhere(mask[sl]>0)[:,0])
    #     y_max=np.max(np.argwhere(mask[sl]>0)[:,1])

    #     new_mask[sl,x_min:(x_max+1),y_min:(y_max+1)]=1

    x_min=np.min(np.argwhere(mask>0)[:,1])
    y_min=np.min(np.argwhere(mask>0)[:,2])

    x_max=np.max(np.argwhere(mask>0)[:,1])
    y_max=np.max(np.argwhere(mask>0)[:,2])

    new_mask[:,x_min:(x_max+1),y_min:(y_max+1)]=1
        
    np.save(filename_mask_full,new_mask)

    gif = []
    for i in range(new_mask.shape[0]):
        img = Image.fromarray(np.uint8(new_mask[i] / np.max(new_mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask_full, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return

@machine
@set_parameter("filename_volume", str, default=None, description="MRF time series")
@set_parameter("filename_mask", str, default=None, description="Mask")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("filename", str, default=None, description="filename for parameters")
@set_parameter("undersampling_factor", int, default=None, description="Kz undersampling factor")
@set_parameter("dictfile", str, default=None, description="Dictionary file")
@set_parameter("dictfile_light", str, default=None, description="Light Dictionary file for 2 steps matching")
@set_parameter("optimizer_config",type=Config,default=DEFAULT_OPT_CONFIG_2STEPS,description="Optimizer parameters")
@set_parameter("slices",str,default=None,description="Slices to consider for pattern matching")
@set_parameter("file_deformation", str, default=None, description="File with deformation map for motion correction")
@set_parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
def build_maps(filename_volume,filename_mask,filename_b1,filename_weights,filename,undersampling_factor,dictfile,dictfile_light,optimizer_config,slices,file_deformation,nb_rep_center_part):


    opt_type = optimizer_config["type"]
    print(opt_type)
    print(filename_volume)
    file_map = "".join(filename_volume.split(".npy")) + "_{}_MRF_map.pkl".format(opt_type)
    volumes_all = np.load(filename_volume)

    print(filename_mask)
    mask=np.load(filename_mask)

    print(volumes_all.shape)

    if file_deformation is not None:
        deformation_map=np.load(file_deformation)
    else:
        deformation_map=None

    ##filename_mask_full = filename_mask.split(".npy")[0] + "_full.npy"
    ##x_min=np.min(np.argwhere(mask>0)[:,1])
    ##y_min=np.min(np.argwhere(mask>0)[:,2])

    ##x_max=np.max(np.argwhere(mask>0)[:,1])
    ##y_max=np.max(np.argwhere(mask>0)[:,2])

    ##mask[:,x_min:(x_max+1),y_min:(y_max+1)]=1
    ##np.save(filename_mask_full,mask)

    if volumes_all.ndim>=5:# first bin is the respiratory bin
        offset=1
    else:
        offset=0
    ntimesteps=volumes_all.shape[offset]
    print("There are {} volumes to match for the fingerprinting".format(ntimesteps))

    if filename is None:
        filename = filename_volume.split("_volumes.npy")[0] + ".dat"


    folder = "/".join(str.split(filename, "/")[:-1])
    dico_seqParams = build_dico_seqParams(filename, folder)


    x_FOV = dico_seqParams["x_FOV"]
    y_FOV = dico_seqParams["y_FOV"]
    z_FOV = dico_seqParams["z_FOV"]
    #nb_part = dico_seqParams["nb_part"]


    npoint=2*volumes_all.shape[offset+2]
    nb_slices=volumes_all.shape[offset+1]
    dx = x_FOV / (npoint / 2)
    dy = y_FOV / (npoint / 2)
    dz = z_FOV / nb_slices



    if slices is not None:
        sl = np.array(slices.split(",")).astype(int)
        if not(len(sl)==0):
            mask_slice = np.zeros(mask.shape, dtype=mask.dtype)
            mask_slice[sl] = 1
            mask *= mask_slice
            sl=[str(s) for s in sl]
            file_map = "".join(filename_volume.split(".npy")) + "_sl{}_{}_MRF_map.pkl".format("_".join(sl),opt_type)

    print(file_map)

    # if (slices is not None) and ("spacing" in slices) and not(len(slices["spacing"])==0):
    #     spacing=slices["spacing"]
    # else:
    #     spacing = [5, 1, 1]



    threshold_pca=optimizer_config["pca"]
    split=optimizer_config["split"]
    useGPU = optimizer_config["useGPU"]

    if "niter" in optimizer_config.keys():
        niter=optimizer_config["niter"]
    else:
        niter=0

    print(opt_type)

    if opt_type=="CF":
        optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=None, split=split, pca=True,
                                     threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False, gen_mode="other")
        all_maps = optimizer.search_patterns_test(dictfile, volumes_all)

    elif opt_type=="Matrix":
        optimizer = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=True,
                                 threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,
                                 useGPU_simulation=False, gen_mode="other")
        all_maps = optimizer.search_patterns_matrix(dictfile, volumes_all)

    elif opt_type=="Brute":
        
        if "volumes_type" in optimizer_config:
            volumes_type=optimizer_config["volumes_type"]
        else:
            volumes_type="Standard"

        if volumes_type=="Singular":

            print("Projecting dictionary on singular basis")
        
            L0=ntimesteps
            print(dictfile)
            filename_phi=str.split(dictfile,".dict") [0]+"_phi_L0_{}.npy".format(L0)

            if filename_phi not in os.listdir():
                #mrfdict = dictsearch.Dictionary()

                print("Generating Phi : {}".format(filename_phi))
                keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

                import dask.array as da
                u,s,vh = da.linalg.svd(da.asarray(values))

                vh=np.array(vh)
                #s=np.array(s)

                phi=vh[:L0]
                np.save(filename_phi,phi.astype("complex64"))
                #del mrfdict
                del keys
                del values
                del u
                del s
                del vh
            else:
                phi=np.load(filename_phi)


            mrfdict = dictmodel.Dictionary()
            mrfdict.load(dictfile, force=True)
            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]
            array_water_projected=array_water@phi.T.conj()
            array_fat_projected=array_fat@phi.T.conj()
            optimizer=BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=split, pca=True,
                                    threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,n_clusters_dico=100,pruning=0.05
                                    )
            all_maps = optimizer.search_patterns((array_water_projected,array_fat_projected,keys), volumes_all)

        else:
            optimizer=BruteDictSearch(FF_list=np.arange(0,1.01,0.05),mask=mask,split=split, pca=True,
                                    threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,n_clusters_dico=100,pruning=0.05
                                    )
            all_maps = optimizer.search_patterns(dictfile, volumes_all)


    if opt_type=="CF_twosteps":
        
        

        if "log" in optimizer_config:
            log=optimizer_config["log"]
        else:
            log=False
        optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=None, split=split, pca=True,
                                     threshold_pca=threshold_pca, log=log, useGPU_dictsearch=useGPU,
                                     useGPU_simulation=False, gen_mode="other",threshold_ff=0.9,dictfile_light=dictfile_light,ntimesteps=ntimesteps)
        all_maps = optimizer.search_patterns_test_multi_2_steps_dico(dictfile, volumes_all)

    elif opt_type=="CF_iterative":
        niter=optimizer_config["niter"]
        mu=optimizer_config["mu"]
        if undersampling_factor is None:
            undersampling_factor=optimizer_config["US"]
        nspoke=optimizer_config["nspoke"]

        if "use_navigator_dll" in dico_seqParams:
            use_navigator_dll = dico_seqParams["use_navigator_dll"]

        else:
            use_navigator_dll=False
        if use_navigator_dll:
            meas_sampling_mode = dico_seqParams["alFree"][14]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        print("incoherent {}".format(incoherent))

        nb_segments = dico_seqParams["alFree"][4]


        radial_traj = Radial3D(total_nspokes=nb_segments, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        filename_b1 = filename_volume.split("_volumes.npy")[0] + "_b1.npy"
        b1_all_slices=np.load(filename_b1)

        if "mu_TV" not in optimizer_config:
            optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=split, pca=True,
                                         threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,
                                         useGPU_simulation=False, gen_mode="other",b1=b1_all_slices,mu=mu,ntimesteps=ntimesteps)
        else:
            if "weights_TV" not in optimizer_config:
                optimizer_config["weights_TV"]=[1.0,0.0,0.0]

            optimizer = SimpleDictSearch(mask=mask, niter=niter, seq=None, trajectory=radial_traj, split=split,
                                         pca=True,
                                         threshold_pca=threshold_pca, log=False, useGPU_dictsearch=useGPU,
                                         useGPU_simulation=False, gen_mode="other", b1=b1_all_slices, mu=mu,
                                         ntimesteps=int(nb_segments / nspoke),mu_TV=optimizer_config["mu_TV"],weights_TV=optimizer_config["weights_TV"])

        all_maps = optimizer.search_patterns_test_multi(dictfile, volumes_all)


    elif opt_type=="CF_iterative_2Dplus1":

        
        clustering=optimizer_config["clustering"]
        niter=optimizer_config["niter"]
        mu=optimizer_config["mu"]
        mu_TV=optimizer_config["mu_TV"]
        weights_TV=optimizer_config["weights_TV"]
        nspoke=optimizer_config["nspoke"]
        volumes_type=optimizer_config["volumes_type"]
        return_matched_signals=optimizer_config["return_matched_signals"]
        return_cost = optimizer_config["return_cost"]
        radial_traj=Radial(total_nspokes=nspoke,npoint=npoint)
        mu_bins=optimizer_config["mu_bins"]
        if "log" in optimizer_config:
            log=optimizer_config["log"]
        else:
            log=False

        L0=ntimesteps
        filename_phi=str.split(dictfile,".dict") [0]+"_phi_L0_{}.npy".format(L0)

        if filename_phi not in os.listdir():
            #mrfdict = dictsearch.Dictionary()

            print("Generating Phi : {}".format(filename_phi))
            keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

            import dask.array as da
            u,s,vh = da.linalg.svd(da.asarray(values))

            vh=np.array(vh)
            #s=np.array(s)

            phi=vh[:L0]
            np.save(filename_phi,phi.astype("complex64"))
            #del mrfdict
            del keys
            del values
            del u
            del s
            del vh
        else:
            phi=np.load(filename_phi)

        mrfdict = dictmodel.Dictionary()
        mrfdict.load(dictfile, force=True)
        keys = mrfdict.keys
        array_water = mrfdict.values[:, :, 0]
        array_fat = mrfdict.values[:, :, 1]
        array_water_projected=array_water@phi.T.conj()
        array_fat_projected=array_fat@phi.T.conj()

        mrfdict_light = dictmodel.Dictionary()
        mrfdict_light.load(dictfile_light, force=True)
        keys_light = mrfdict_light.keys
        array_water = mrfdict_light.values[:, :, 0]
        array_fat = mrfdict_light.values[:, :, 1]
        array_water_light_projected=array_water@phi.T.conj()
        array_fat_light_projected=array_fat@phi.T.conj()

        b1_all_slices=np.load(filename_b1)
        if filename_weights is not None:
            weights=np.load(filename_weights)
        else:
            weights=1

        optimizer = SimpleDictSearch(mask=mask,niter=niter,seq=None,trajectory=radial_traj,split=split,pca=True,threshold_pca=threshold_pca,log=log,useGPU_dictsearch=useGPU,useGPU_simulation=False,gen_mode="other",movement_correction=False,cond=None,ntimesteps=ntimesteps,b1=b1_all_slices,threshold_ff=0.9,dictfile_light=(array_water_light_projected,array_fat_light_projected,keys_light),mu=1,mu_TV=mu_TV,weights_TV=weights_TV,weights=weights,volumes_type=volumes_type,return_matched_signals=return_matched_signals,return_cost=return_cost,clustering=clustering,mu_bins=mu_bins,deformation_map=deformation_map,nb_rep_center_part=nb_rep_center_part)
        all_maps=optimizer.search_patterns_test_multi_2_steps_dico((array_water_projected,array_fat_projected,keys),volumes_all,retained_timesteps=None)




    curr_file=file_map
    file = open(curr_file, "wb")
    pickle.dump(all_maps,file)
    file.close()

    for iter in list(all_maps.keys()):

        map_rebuilt=all_maps[iter][0]
        mask=all_maps[iter][1]

        map_rebuilt["wT1"][map_rebuilt["ff"] > 0.7] = 0.0

        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
        #map_Python.buildParamMap()


        for key in ["ff","wT1","df","attB1"]:
            file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
            io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})

    return

@machine
@set_parameter("file_map",str,default=None,description="map file (.pkl)")
@set_parameter("config_image_maps",type=Config,default=None,description="Image Config")
@set_parameter("suffix", str, default="", description="suffix")
def generate_image_maps(file_map,config_image_maps,suffix):
    return_cost=config_image_maps["return_cost"]
    return_matched_signals=config_image_maps["return_matched_signals"]
    #keys=list(all_maps.keys())
    keys=config_image_maps["keys"]
    list_l=config_image_maps["singular_volumes_outputted"]

    print(keys)

    distances=config_image_maps["image_distances"]
    dx=distances[0]
    dy=distances[1]
    dz=distances[2]

    #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("")
    curr_file=file_map
    file = open(curr_file, "rb")
    all_maps = pickle.load(file)
    file.close()

    if not(keys):
        keys=list(all_maps.keys())

    for iter in keys:

        map_rebuilt=all_maps[iter][0]
        mask=all_maps[iter][1]

        map_rebuilt["wT1"][map_rebuilt["ff"] > 0.7] = 0.0

        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
        #map_Python.buildParamMap()


        for key in ["ff","wT1","df","attB1"]:
            file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,iter,key)
            if file_mha.startswith("/"):
                file_mha=file_mha[1:]
            io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})



        if return_matched_signals:
            for l in list_l:
                matched_volumes=makevol(all_maps[iter][-1][l],mask>0)
                #print(matched_volumes)
                file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                     "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_l{}_{}.mha".format(suffix,
                                                                                                                                  iter,l, "matchedvolumes")
                if file_mha.startswith("/"):
                    file_mha=file_mha[1:]
                io.write(file_mha, np.abs(matched_volumes), tags={"spacing": [dz, dx, dy]})


        if return_cost:
            try:
                file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                     "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
                                                                                                                              iter, "correlation")
                if file_mha.startswith("/"):
                    file_mha=file_mha[1:]
                io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

                file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                     "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
                                                                                                                              iter, "phase")
                if file_mha.startswith("/"):
                    file_mha=file_mha[1:]
                io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})
            except:
                continue
    return


@machine
@set_parameter("file_volume",str,default=None,description="volume file (.mha)")
@set_parameter("l", int, default=0, description="Singular volume")
def extract_singular_volume_allbins(file_volume,l):
    file_volume_target=str.replace(file_volume,"volumes_singular","volume_singular_l{}".format(l))
    volumes=np.load(file_volume)
    print(volumes.dtype)
    np.save(file_volume_target,volumes[:,l])
    return


@machine
@set_parameter("file_volume",str,default=None,description="volume file (.mha)")
@set_parameter("gr", int, default=0, description="Motion bin state")
def extract_allsingular_volumes_bin(file_volume,gr):
    file_volume_target=str.replace(file_volume,"volumes_singular_allbins","volumes_singular_gr{}".format(gr))
    volumes=np.load(file_volume)
    print(volumes.dtype)
    np.save(file_volume_target,volumes[gr,:])
    return

@machine
@set_parameter("file_volume",str,default=None,description="volume file (.mha)")
@set_parameter("nb_gr",int,default=4,description="number of respiratory bins")
@set_parameter("sl", int, default=None, description="Slice")
@set_parameter("x", int, default=None, description="x")
@set_parameter("y", int, default=None, description="y")
@set_parameter("slice_res_factor", int, default=5, description="Factor between slice thickness and in plane resolution")
@set_parameter("l", int, default=0, description="Singular volume")
@set_parameter("metric", ["abs","phase","real","imag"], default="abs", description="Metric to plot")
@set_parameter("single_volume", bool, default=False, description="One single volume - No bin or singular volume")
def generate_movement_gif(file_volume,nb_gr,sl,x,y,l,metric,slice_res_factor,single_volume):
    if sl is not None:
        filename_gif = str.split(file_volume.format(nb_gr), ".mha")[0] + "_sl{}_moving_singular.gif".format(sl)
    elif x is not None:
        filename_gif = str.split(file_volume.format(nb_gr), ".mha")[0] + "_x{}_moving_singular.gif".format(x)
    elif y is not None:
        filename_gif = str.split(file_volume.format(nb_gr), ".mha")[0] + "_y{}_moving_singular.gif".format(y)
    elif single_volume:
        filename_gif = str.split(file_volume, ".npy")[0] + "_moving_slices.gif"
    
    if file_volume.find(".mha")>0:
        def load(file):
            return io.read(file)
    else:
        def load(file):
            return np.load(file)
    
    test_volume=load(file_volume.format(0)).squeeze()
    
    if single_volume:
        print("Single volume - The GIF will be a movie navigating along the slices")
        test_volume=np.expand_dims(test_volume,axis=1)
    print(test_volume.shape)
    if test_volume.ndim==3:# each file contains only one motion phase
        all_matched_volumes=[]
        #print(file_volume)
        for gr in np.arange(nb_gr):
            file_mha= file_volume.format(gr)
            matched_volume=load(file_mha)
            matched_volume=np.array(matched_volume)
            all_matched_volumes.append(matched_volume)

        all_matched_volumes=np.array(all_matched_volumes)

    elif test_volume.ndim==4:#file contains all phases for one singular volume
        all_matched_volumes=test_volume


    elif test_volume.ndim==5:#file contains all phases and all singular volumes
        all_matched_volumes=test_volume[:,l]
        filename_gif=filename_gif.replace("moving_singular.gif","moving_singular_l{}.gif".format(l))

    print(all_matched_volumes.shape)
    if sl is not None:
        moving_image=np.concatenate([all_matched_volumes[:,sl],all_matched_volumes[1:-1,sl][::-1]],axis=0)
    elif x is not None:
        moving_image=np.concatenate([all_matched_volumes[:,:,x],all_matched_volumes[1:-1,:,x][::-1]],axis=0)
    elif y is not None:
        all_matched_volumes=np.repeat(all_matched_volumes,slice_res_factor,axis=1)
        moving_image=np.concatenate([all_matched_volumes[:,:,:,y],all_matched_volumes[1:-1,:,:,y][::-1]],axis=0)
    elif single_volume:
        moving_image=np.concatenate([all_matched_volumes[:,0],all_matched_volumes[1:-1,0][::-1]],axis=0)
    animate_images(moving_image,interval=10)

    from PIL import Image
    gif=[]

    if metric=="abs":
        volume_for_gif = np.abs(moving_image)
    elif metric=="phase":
        volume_for_gif = np.angle(moving_image)
        filename_gif = str.replace(filename_gif,"moving_singular","moving_singular_phase")

    elif metric=="real":
        volume_for_gif = np.real(moving_image)
        filename_gif = str.replace(filename_gif,"moving_singular","moving_singular_real")

    elif metric=="imag":
        volume_for_gif = np.imag(moving_image)
        filename_gif = str.replace(filename_gif,"moving_singular","moving_singular_imag")

    else:
        raise ValueError("metric unknown - choose from abs/phase/real/imag")

    for i in range(volume_for_gif.shape[0]):
        min_value=np.min(volume_for_gif[i])
        max_value=np.max(volume_for_gif[i])
        img = Image.fromarray(np.uint8((volume_for_gif[i]-min_value)/(max_value-min_value)*255), 'L')
        img=img.convert("P")
        gif.append(img)


    gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
    print(filename_gif)
    return

@machine
@set_parameter("file_map",str,default=None,description="map file (.pkl)")
@set_parameter("config_image_maps",type=Config,default=None,description="Image Config")
@set_parameter("suffix", str, default="", description="suffix")
def generate_matchedvolumes_allgroups(file_map,config_image_maps,suffix):
    curr_file=file_map
    file = open(curr_file, "rb")
    all_maps = pickle.load(file)
    file.close()

    distances=config_image_maps["image_distances"]
    dx=distances[0]
    dy=distances[1]
    dz=distances[2]


    keys=config_image_maps["keys"]

    matched_signals=all_maps[keys[0]][-1]
    nb_singular_images=matched_signals.shape[0]
    mask=all_maps[keys[0]][1]
    nb_signals=mask.sum()

    matched_signals=matched_signals.reshape(nb_singular_images,-1,nb_signals)
    nb_gr=matched_signals.shape[1]


    l_list=config_image_maps["singular_volumes_outputted"]


    for gr in range(nb_gr):
        for l in l_list:
            for iter in keys:
                matched_signals=all_maps[iter][-1]
                matched_signals=matched_signals.reshape(nb_singular_images,-1,nb_signals)
                matched_volumes=makevol(matched_signals[l][gr],mask>0)
                file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                            "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_gr{}_it{}_l{}_{}.mha".format(suffix,gr,
                            iter,l, "matchedvolumes")
                if file_mha.startswith("/"):
                    file_mha=file_mha[1:]
                io.write(file_mha, np.abs(matched_volumes), tags={"spacing": [dz, dx, dy]})



@machine
@set_parameter("files_config",type=Config,default=None,description="Files to consider for aggregate kdata reconstruction")
@set_parameter("disp_config",type=Config,default=None,description="Parameters for movement identification")
@set_parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
def build_data_nacq(files_config,disp_config,undersampling_factor):

    base_folder = "./data/InVivo/3D"
    files=files_config["files"]
    shifts=list(range(disp_config["shifts"][0],disp_config["shifts"][1]))
    ch=disp_config["channel"]
    bin_width=disp_config["bin_width"]

    folder = base_folder + "/".join(str.split(files[0], "/")[:-1])


    filename_categories_global = folder + "/categories_global_bw{}.npy".format(bin_width)
    filename_df_groups_global = folder + "/df_groups_global_bw{}.pkl".format(bin_width)

    categories_global = []
    df_groups_global = pd.DataFrame()

    for localfile in files:

        filename = base_folder + localfile
        filename_nav_save = str.split(filename, ".dat")[0] + "_nav.npy"
        folder = "/".join(str.split(filename, "/")[:-1])
        filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")
        filename_disp_image=str.split(filename, ".dat")[0] + "_nav_image.jpg".format("")


        dico_seqParams = build_dico_seqParams(filename, folder)

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if use_navigator_dll:
            meas_sampling_mode = dico_seqParams["alFree"][14]
            nb_gating_spokes = dico_seqParams["alFree"][6]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]
            nb_gating_spokes = 0


        nb_segments = dico_seqParams["alFree"][4]

        del dico_seqParams

        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        data, data_for_nav = build_data(filename, folder, nb_segments, nb_gating_spokes)

        data_shape = data.shape

        nb_allspokes = data_shape[-3]
        npoint = data_shape[-1]
        nb_slices = data_shape[-2]

        if str.split(filename_kdata, "/")[-1] in os.listdir(folder):
            del data

        if str.split(filename_kdata, "/")[-1] not in os.listdir(folder):
            # Density adjustment all slices

            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(data.ndim - 1)))

            print("Performing Density Adjustment....")
            data *= density
            np.save(filename_kdata, data)
            del data



        print("Calculating Coil Sensitivity....")

        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        nb_segments = radial_traj.get_traj().shape[0]

        if nb_gating_spokes > 0:
            print("Processing Nav Data...")
            data_for_nav = np.load(filename_nav_save)

            nb_allspokes = nb_segments
            nb_slices = data_for_nav.shape[1]
            nb_channels = data_for_nav.shape[0]
            npoint_nav = data_for_nav.shape[-1]

            all_timesteps = np.arange(nb_allspokes)
            nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

            nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                                   applied_timesteps=list(nav_timesteps))

            nav_image_size = (int(npoint_nav),)

            print("Building nav image for channel {}...".format(ch))
            # b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
            # b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

            image_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[ch], axis=0), nav_traj, nav_image_size)

            plt.figure()
            plt.plot(np.abs(image_nav_ch.reshape(-1, int(npoint / 2)))[5*nb_gating_spokes, :])
            plt.savefig(filename_disp_image)

            print("Estimating Movement...")

            bottom = -shifts[0]
            top = nav_image_size[0]-shifts[-1]


            displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, 0.001)

            displacement_for_binning = displacements

            max_bin = np.max(displacement_for_binning)
            min_bin = shifts[0]

            bins = np.arange(min_bin, max_bin + bin_width, bin_width)
            # print(bins)
            categories = np.digitize(displacement_for_binning, bins)
            df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T,
                                  columns=["displacement", "cat"])
            df_groups = df_cat.groupby("cat").count()
            curr_max = df_groups.displacement.max()

            if df_groups_global.empty:
                df_groups_global = df_groups
            else:
                df_groups_global += df_groups

            categories_global.append(categories)

    #################################################################################################################################"

    categories_global = np.array(categories_global)

    np.save(filename_categories_global, categories_global)
    df_groups_global.to_pickle(filename_df_groups_global)



    return

@machine
@set_parameter("filename_nav_save", str, default=None, description="Navigator data")
@set_parameter("seasonal_adj", bool, default=False, description="Seasonal adjustement")
def build_navigator_images(filename_nav_save,seasonal_adj):
    filename_image_nav= filename_nav_save.split("_nav.npy")[0] + "_image_nav.npy"
    filename_image_nav_plot = filename_nav_save.split("_nav.npy")[0] + "_image_nav.jpg"
    filename_image_nav_diff_plot = filename_nav_save.split("_nav.npy")[0] + "_image_nav_diff.jpg"

    data_for_nav = np.load(filename_nav_save)
    nb_channels = data_for_nav.shape[0]
    npoint_nav = data_for_nav.shape[-1]
    nb_gating_spokes = data_for_nav.shape[-2]
    nb_slices = data_for_nav.shape[1]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                           nb_gating_spokes=nb_gating_spokes)
    nav_image_size = (int(npoint_nav ),)

    image_nav_all_channels = []

    # for j in tqdm(range(nb_channels)):
    for j in tqdm(range(nb_channels)):
        images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j], axis=0), nav_traj,
                                                                 nav_image_size, b1=None)
        image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
        image_nav_all_channels.append(image_nav_ch)

    image_nav_all_channels = np.array(image_nav_all_channels)
    if seasonal_adj:
        from statsmodels.tsa.seasonal import seasonal_decompose

        image_reshaped = image_nav_all_channels.reshape(-1, npoint_nav)
        decomposition = seasonal_decompose(image_reshaped,
                                           model='multiplicative', period=nb_gating_spokes)
        image=image_reshaped/decomposition.seasonal
        image=image.reshape(-1,nb_gating_spokes,npoint_nav)
        image_nav_all_channels=image
        print(image.shape)


    np.save(filename_image_nav,image_nav_all_channels)

    plot_image_grid(
        np.moveaxis(image_nav_all_channels.reshape(nb_channels, -1, int(npoint_nav)), -1, -2)[:, :, :100],
        nb_row_col=(6, 6),save_file=filename_image_nav_plot)

    plot_image_grid(
        np.moveaxis(np.diff(image_nav_all_channels.reshape(nb_channels, -1, int(npoint_nav)), axis=-1), -1, -2)[:,
        :, :100], nb_row_col=(6, 6),save_file=filename_image_nav_diff_plot)

    print("Navigator images plot file: {}".format(filename_image_nav_diff_plot))

    return


@machine
@set_parameter("filename_nav_save", str, default=None, description="Navigator data")
@set_parameter("filename_displacement", str, default=None, description="Displacement data")
@set_parameter("nb_segments", int, default=1400, description="MRF Total Spoke number")
@set_parameter("bottom", int, default=-30, description="Lowest displacement for displacement estimation")
@set_parameter("top", int, default=30, description="Highest displacement for displacement estimation")
@set_parameter("ntimesteps", int, default=175, description="Number of MRF images")
@set_parameter("nspoke_per_z", int, default=8, description="number of spokes before partition jump when undersampling")
@set_parameter("us", int, default=1, description="undersampling_factor")
@set_parameter("incoherent", bool, default=True, description="3D sampling type")
@set_parameter("lambda_tv", float, default=0.001, description="Temporal regularization for displacement estimation")
@set_parameter("ch", int, default=None, description="channel if single channel estimation")
@set_parameter("filename_bins", str, default=None, description="bins file if data is binned according to another scan")
@set_parameter("filename_disp_respi", str, default=None, description="source displacement for distribution matching")
@set_parameter("retained_categories", str, default=None, description="retained bins")
@set_parameter("nbins", int, default=5, description="Number of motion states")
@set_parameter("gating_only", bool, default=False, description="Weights for gating only and not for density compensation")
@set_parameter("pad", int, default=10, description="Navigator images padding")
@set_parameter("randomize", bool, default=False, description="Randomization for baseline navigator image for displacement calc")
@set_parameter("equal_spoke_per_bin", bool, default=False, description="Distribute evenly the number of spokes per bin")
@set_parameter("use_ml", bool, default=False, description="Use segment anything for motion estimation")
@set_parameter("useGPU", bool, default=True, description="Use GPU")
@set_parameter("force_recalc_disp", bool, default=True, description="Force calculation of displacement")
@set_parameter("dct_frequency_filter", int, default=None, description="DCT filtering for displacement smoothing")
@set_parameter("seasonal_adj", bool, default=False, description="Seasonal adjustement")
@set_parameter("hard_interp", bool, default=False, description="Hard interpolation for inversion")
@set_parameter("nb_rep_center_part", int, default=1, description="Central partition repetitions")
@set_parameter("sim_us", int, default=1, description="Undersampling simulation")
@set_parameter("us_file", str, default=None, description="Undersampling simulation from file ")
@set_parameter("interp_bad_correl", bool, default=False, description="Interpolate displacements with neighbours when poorly correlated")
@set_parameter("nav_res_factor", int, default=None, description="bins rescaling if resolution of binning navigator different from current input")
@set_parameter("soft_weight", bool, default=False, description="use soft weight for full inspiration")
@set_parameter("stddisp", float, default=None, description="outlier exclusion for irregular breathing")

def calculate_displacement_weights(filename_nav_save,filename_displacement,nb_segments,bottom,top,ntimesteps,us,incoherent,lambda_tv,ch,filename_bins,retained_categories,nbins,gating_only,pad,randomize,equal_spoke_per_bin,use_ml,useGPU,force_recalc_disp,dct_frequency_filter,seasonal_adj,hard_interp,nb_rep_center_part,sim_us,us_file,interp_bad_correl,nspoke_per_z,nav_res_factor,soft_weight,stddisp,filename_disp_respi):

    '''
    Displacement calculation from raw navigator K-space data
    Remark:
    If use_ml is True, bottom, top, randomize, lambda_TV are not useful
    '''

    print(seasonal_adj)

    if filename_nav_save is None:
        nav_file=False
    else:
        nav_file=True

    if filename_displacement is None:
        filename_displacement=filename_nav_save.split("_nav.npy")[0] + "_displacement.npy"
        filename_weights=filename_nav_save.split("_nav.npy")[0] + "_weights.npy"
        filename_retained_ts=filename_nav_save.split("_nav.npy")[0] + "_retained_ts.pkl"
        filename_bins_output = filename_nav_save.split("_nav.npy")[0] + "_bins.npy"

        folder = "/".join(str.split(filename_nav_save, "/")[:-1])

    else:
        print("Displacement file given")
        filename_weights = filename_displacement.split("_displacement")[0] + "_weights.npy"
        filename_retained_ts = filename_displacement.split("_displacement")[0] + "_retained_ts.pkl"
        filename_bins_output = filename_displacement.split("_displacement")[0] + "_bins.npy"

        folder = "/".join(str.split(filename_displacement, "/")[:-1])
        print(folder)

    if retained_categories is not None:
        retained_categories = np.array(retained_categories.split(",")).astype(int)

    if nav_file:
        data_for_nav=np.load(filename_nav_save)


    if filename_disp_respi is not None:
        disp_respi=np.load(filename_disp_respi)
    else:
        disp_respi=None


    if ((str.split(filename_displacement, "/")[-1] not in os.listdir(folder)) or (force_recalc_disp)):
        if use_ml:
            if useGPU:
                device="cuda"
            else:
                device="cpu"
            displacements=calculate_displacement_ml(data_for_nav,nb_segments,ch=ch,device=device)

        else:
            if ch is None:
                displacements=calculate_displacements_allchannels(data_for_nav,nb_segments,shifts = list(range(bottom, top)),lambda_tv=lambda_tv,pad=pad,randomize=randomize)
            else:
                displacements = calculate_displacements_singlechannel(data_for_nav, nb_segments, shifts=list(range(bottom, top)),
                                                                    lambda_tv=lambda_tv,ch=ch,pad=pad,randomize=randomize,dct_frequency_filter=dct_frequency_filter,seasonal_adj=seasonal_adj,interp_bad_correl=interp_bad_correl)
        np.save(filename_displacement, displacements)

    else:
        displacements=np.load(filename_displacement)

    if nav_file:
        nb_slices=data_for_nav.shape[1]
        nb_gating_spokes=data_for_nav.shape[2]
    else:
        nb_slices=displacements.shape[0]
        nb_gating_spokes=displacements.shape[1]

    if hard_interp:
        disp_interp=copy(displacements).reshape(-1,nb_gating_spokes)
        disp_interp[:, :8] = ((disp_interp[:, 7] - disp_interp[:, 0]) / (7 - 0))[:, None] * np.arange(8)[None,
                                                                                            :] + disp_interp[:, 0][:,
                                                                                                 None]
        displacements = disp_interp.flatten()
        np.save(filename_displacement, displacements)

    filename_displacement_plot = filename_displacement.replace(".npy",".jpg")
    plt.plot(displacements.flatten())
    plt.savefig(filename_displacement_plot)

    radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=800,nb_slices=nb_slices*us,incoherent=incoherent,mode="old",nspoke_per_z_encoding=nspoke_per_z)

    if filename_bins is None:
        dico_traj_retained,dico_retained_ts,bins=estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=nbins,retained_categories=retained_categories,equal_spoke_per_bin=equal_spoke_per_bin,sim_us=sim_us,us_file=us_file,us=us,soft_weight_for_full_inspi=soft_weight,nb_rep_center_part=nb_rep_center_part,std_disp=stddisp,disp_respi=disp_respi)

        np.save(filename_bins_output,bins)
    else:
        bins=np.load(filename_bins)
        if nav_res_factor is not None:
            bins=nav_res_factor*bins
            print("Rescaled bins {}".format(bins))
        nb_bins=len(bins)+1
        print(nb_bins)
        dico_traj_retained,dico_retained_ts,_=estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=nb_bins,retained_categories=retained_categories,bins=bins,sim_us=sim_us,us_file=us_file,us=us,soft_weight_for_full_inspi=soft_weight,nb_rep_center_part=nb_rep_center_part,disp_respi=disp_respi)

    weights=[]
    for gr in dico_traj_retained.keys():
        weights.append(np.expand_dims(dico_traj_retained[gr],axis=-1))
    weights=np.array(weights)
    if gating_only:
        weights=(weights>0)*1
    print(weights.shape)
    np.save(filename_weights, weights)

    file = open(filename_retained_ts, "wb")
    pickle.dump(dico_retained_ts, file)
    file.close()
    return


@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@set_parameter("n_comp", int, default=None, description="Number of virtual coils")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("invert_dens_adj", bool, default=False, description="Remove Radial density adjustment")
@set_parameter("res",int,default=16,description="central points kept for coil sensitivity calc")
@set_parameter("res_kz",int,default=None,description="central partitions kept for coil sensitivity calc when no coil compression")
@set_parameter("cc_res",int,default=None,description="central points kept for coil compression")
@set_parameter("cc_res_kz",int,default=None,description="central partitions kept for coil compression")
def coil_compression(filename_kdata, dens_adj,n_comp,nb_rep_center_part,invert_dens_adj,res,cc_res,res_kz,cc_res_kz):
    kdata_all_channels_all_slices = np.load(filename_kdata)
    
    nb_channels=kdata_all_channels_all_slices.shape[0]
    print("Nb Channels : {}".format(nb_channels))

    filename_virtualcoils = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)
    filename_b12Dplus1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if nb_rep_center_part>1:
        kdata_all_channels_all_slices=kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part)

    if dens_adj:
        npoint=kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density


    if n_comp>=nb_channels:#no coil compression
        n_comp=nb_channels
        print("No Coil Compression")
        data_shape=kdata_all_channels_all_slices.shape
        print(data_shape)
        nb_allspokes = data_shape[-3]
        npoint = data_shape[-1]
        nb_slices = data_shape[-2]
        image_size=(nb_slices,int(npoint/2),int(npoint/2))
        radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=False,nspoke_per_z_encoding=nb_allspokes)
        print("Here")
        b1_all_slices_2Dplus1_pca=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=True,hanning_filter=True,res_kz=res_kz)
        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=nb_channels)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(nb_channels)
            pca_dict[sl]=deepcopy(pca)
        
        print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))


    else:
        pca_dict,b1_all_slices_2Dplus1_pca=coil_compression_2Dplus1(kdata_all_channels_all_slices, n_comp=n_comp,invert_dens_adj=invert_dens_adj,res=res,cc_res=cc_res,cc_res_kz=cc_res_kz)


    image_file=str.split(filename_b12Dplus1, ".npy")[0] + ".jpg"

    sl = int(b1_all_slices_2Dplus1_pca.shape[1]/2)

    list_images=list(np.abs(b1_all_slices_2Dplus1_pca[:,sl,:,:]))
    plot_image_grid(list_images,(6,6),title="Sensivitiy map for slice".format(sl),save_file=image_file)


    with open(filename_virtualcoils, "wb") as file:
        pickle.dump(pca_dict, file)

    np.save(filename_b12Dplus1, b1_all_slices_2Dplus1_pca)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@set_parameter("n_comp", int, default=None, description="Number of virtual coils")
@set_parameter("filename_cc", str, default=None, description="Filename for coil compression")
@set_parameter("calc_sensi", bool, default=True, description="Calculate coil sensitivities")
def coil_compression_bart(filename_kdata,dens_adj,n_comp,filename_cc,calc_sensi):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_channels,nb_segments,nb_slices,npoint=kdata_all_channels_all_slices.shape

    filename_virtualcoils = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_virtualcoils_{}.pkl".format(n_comp,n_comp)
    filename_b12Dplus1 = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_b12Dplus1_{}.npy".format(n_comp,n_comp)
    filename_kdata_compressed = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_kdata.npy".format(n_comp)
    
    if dens_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density
    
    kdata_bart=np.moveaxis(kdata_all_channels_all_slices,-1,1)
    kdata_bart=np.moveaxis(kdata_bart,0,-1)
    kdata_bart=kdata_bart[None,:]
    kdata_bart=kdata_bart.reshape(1,npoint,-1,nb_channels)
    
    if (filename_cc is None) or calc_sensi:
        

        incoherent=False
        radial_traj = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                nb_slices=nb_slices, incoherent=incoherent, mode=None,nspoke_per_z_encoding=nb_segments,)

        traj_python = radial_traj.get_traj()
        traj_python=traj_python.reshape(nb_segments,nb_slices,-1,3)
        traj_python=traj_python.T
        traj_python=np.moveaxis(traj_python,-1,-2)

        traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(
        npoint / 4)
        traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
            npoint / 4)
        traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
            nb_slices / 2)
        
        
        traj_python_bart=traj_python.reshape(3,npoint,-1)

        coil_img = bart(1,'nufft -a -t', traj_python_bart, kdata_bart)
        kdata_cart = bart(1,'fft -u 7', coil_img)

    if filename_cc is None:
        print("Calculating coil compression")
        filename_cc = str.split(filename_kdata, "_kdata.npy")[0] + "_bart_cc.cfl"
        cc=bart(1,"cc -M",kdata_cart)
        cfl.writecfl(filename_cc,cc)
    else:
        print("Loading Coil compression")
        cc=cfl.readcfl(filename_cc)

    print("Applying coil compression to k-space data")
    kdata_bart_cc = bart(1, 'ccapply -p {}'.format(n_comp), kdata_bart, cc)
    kdata_python_cc=kdata_bart_cc.squeeze().reshape(npoint,nb_segments,nb_slices,n_comp)
    kdata_python_cc=np.moveaxis(kdata_python_cc,-1,0)
    kdata_python_cc=np.moveaxis(kdata_python_cc,1,-1)
    np.save(filename_kdata_compressed,kdata_python_cc)

    if calc_sensi:
        print("Calculating coil sensi")
        kdata_cart_cc = bart(1, 'ccapply -p {}'.format(n_comp), kdata_cart, cc)
        b1_bart_cc=bart(1,"ecalib -m1",kdata_cart_cc)
        b1_python_cc=np.moveaxis(b1_bart_cc,-2,0)
        b1_python_cc=np.moveaxis(b1_python_cc,-1,0)
        np.save(filename_b12Dplus1,b1_python_cc)

        image_file=str.split(filename_b12Dplus1, ".npy")[0] + ".jpg"

        sl = int(b1_python_cc.shape[1]/2)

        list_images=list(np.abs(b1_python_cc[:,sl,:,:]))
        plot_image_grid(list_images,(6,6),title="BART Coil Sensitivity map for slice".format(sl),save_file=image_file)

        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=n_comp)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(n_comp)
            pca_dict[sl]=deepcopy(pca)

        with open(filename_virtualcoils, "wb") as file:
            pickle.dump(pca_dict, file)





@machine
@set_parameter("filename_volume", str, default=None, description="Volume time serie")
@set_parameter("sl", str, default=None, description="Slices to select")
def select_slices_volume(filename_volume, sl):
    filename_volume_new=filename_volume.split(".npy")[0]+"_{}.npy".format(sl)
    
    volume=np.load(filename_volume)
    slices=np.array(sl.split(",")).astype(int)
    volume_new=volume[:,slices]

    np.save(filename_volume_new, volume_new)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@set_parameter("in_phase", bool, default=False, description="Select only in phase spokes from original MRF sequence")
@set_parameter("out_phase", bool, default=False, description="Select only out of phase spokes from original MRF sequence")
@set_parameter("full_volume", bool, default=False, description="Build one volume with all spokes (weights are not used)")
@set_parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
@set_parameter("us", int, default=1, description="Undersampling")
def build_volumes_allbins(filename_kdata,filename_b1,filename_pca,filename_weights,n_comp,gating_only,dens_adj,in_phase,out_phase,full_volume,nb_rep_center_part,us):
    '''
    Build single volume for each motion phase with all spokes (for motion deformation field estimation)
    '''
    if not(gating_only):
        filename_volumes=filename_kdata.split("_kdata.npy")[0] + "_volumes_allbins.npy"
    else:
        filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_no_dcomp_volumes_allbins.npy"

    if full_volume:
        filename_volumes=str.replace(filename_volumes,"volumes_allbins","full_volume")

    print("Loading Kdata")
    kdata_all_channels_all_slices=np.load(filename_kdata)

    if dens_adj:
        npoint = kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density

    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=(str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)).replace("_no_densadj","").replace("_no_dens_adj","")

    if filename_pca is None:
        filename_pca = (str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)).replace("_no_densadj","")

    if filename_weights is None:
        filename_weights = (str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy").replace("_no_densadj","")
    
    b1_all_slices_2Dplus1_pca=np.load(filename_b1)

    if not(full_volume):
        all_weights=np.load(filename_weights)
    else:
        all_weights=np.ones(shape=(1,1,kdata_all_channels_all_slices.shape[1],kdata_all_channels_all_slices.shape[2],1))
    if gating_only:
        all_weights=(all_weights>0)*1

    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us

    file = open(filename_pca, "rb")
    pca_dict = pickle.load(file)
    file.close()

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("PCA components shape {}".format(pca_dict[0].components_.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))

    if in_phase:
        selected_spokes=np.r_[300:800,1200:1400]
        #selected_spokes=np.r_[280:580]
    elif out_phase:
        selected_spokes = np.r_[800:1200]
    else:
        selected_spokes=None

    volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights,selected_spokes,nb_rep_center_part)
    np.save(filename_volumes,volumes_allbins)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@set_parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@set_parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@set_parameter("file_deformation_map", str, default=None, description="Deformation map from bin 0 to other bins")
@set_parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@set_parameter("useGPU", bool, default=True, description="Use GPU")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("index_ref", int, default=0, description="Reference bin for deformation")
@set_parameter("interp", str, default=None, description="Registration interpolation")
@set_parameter("suffix", str, default="", description="Suffix")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
def build_volumes_singular_allbins_registered(filename_kdata, filename_b1, filename_pca, filename_weights,filename_phi,dictfile,L0,file_deformation_map,n_comp,useGPU,nb_rep_center_part,index_ref,interp,gating_only,suffix,select_first_rep):
    '''
    Build singular volumes for MRF registered to the same motion phase and averaged (first iteration of the gradient descent for motion-corrected MRF)
    Output shape L0 x nz x nx x ny
    '''
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_registered_gr{}{}.npy".format(index_ref,suffix)
    print("Loading Kdata")
    
    kdata_all_channels_all_slices = np.load(filename_kdata)

    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_pca is None:
        filename_pca = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    if ((filename_phi is None) and (dictfile is None)):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_phi = str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)



    print("Loading B1")
    b1_all_slices_2Dplus1_pca = np.load(filename_b1)
    print("Loading Weights")
    all_weights = np.load(filename_weights)

    if gating_only:
        all_weights=(all_weights>0)*1

    print("Loading Coil Compression weights")
    file = open(filename_pca, "rb")
    pca_dict = pickle.load(file)
    file.close()

    print("Loading Time Basis")
    if filename_phi not in os.listdir():
        phi = build_phi(dictfile, L0)
    else:
        phi = np.load(filename_phi)
    print("Loading Deformation Map")
    deformation_map=np.load(file_deformation_map)
    if not(index_ref==0):
        deformation_map=change_deformation_map_ref(deformation_map,index_ref)

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("virtual coils components shape {}".format(pca_dict[0].components_.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))


    if interp is None:
        interp=cv2.INTER_LINEAR

    elif interp=="nearest":
        interp=cv2.INTER_NEAREST
    
    elif interp=="cubic":
        interp=cv2.INTER_CUBIC



    volumes_allbins_registered=build_volume_singular_2Dplus1_cc_allbins_registered(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict,
                                               all_weights, phi, L0, deformation_map,useGPU,nb_rep_center_part,interp,select_first_rep)
    np.save(filename_volumes, volumes_allbins_registered)

    return




@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@set_parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@set_parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@set_parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@set_parameter("useGPU", bool, default=True, description="Use GPU")
@set_parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("us", int, default=1, description="Undersampling")


def build_volumes_singular_allbins(filename_kdata, filename_b1, filename_pca, filename_weights,filename_phi,dictfile,L0,n_comp,useGPU,dens_adj,nb_rep_center_part,gating_only,us):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)

    if dens_adj:
        npoint = kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density


    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_pca is None:
        filename_pca = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    if ((filename_phi is None) and (dictfile is None)):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_phi = str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)

    print(filename_phi)
    b1_all_slices_2Dplus1_pca = np.load(filename_b1)

    nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]
    print(kdata_all_channels_all_slices.shape)
    print(nb_slices_b1)
    nb_slices=kdata_all_channels_all_slices.shape[-2]-nb_rep_center_part+1
    npoint=int(kdata_all_channels_all_slices.shape[-1]/2)

    nb_channels=b1_all_slices_2Dplus1_pca.shape[0]
    
    file = open(filename_pca, "rb")
    pca_dict = pickle.load(file)
    file.close()

    # if nb_slices>nb_slices_b1:
    #     us_b1 = int(nb_slices / nb_slices_b1)
    #     print("B1 map on x{} coarser grid. Interpolating B1 map on a finer grid".format(us_b1))
    #     b1_all_slices_2Dplus1_pca=interp_b1(b1_all_slices_2Dplus1_pca,us=us_b1,start=0)

    #     print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
    #     pca_dict={}
    #     for sl in range(nb_slices):
    #         pca=PCAComplex(n_components_=nb_channels)
    #         pca.explained_variance_ratio_=[1]
    #         pca.components_=np.eye(nb_channels)
    #         pca_dict[sl]=deepcopy(pca)

    if (nb_slices>nb_slices_b1)or(npoint>npoint_b1):
        
        print("Regridding b1")
        new_shape=(nb_slices,npoint,npoint)
        b1_all_slices_2Dplus1_pca=interp_b1_resize(b1_all_slices_2Dplus1_pca,new_shape)

        print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=nb_channels)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(nb_channels)
            pca_dict[sl]=deepcopy(pca)

    all_weights = np.load(filename_weights)

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1


    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us


    if filename_phi not in os.listdir():
        phi = build_phi(dictfile, L0)
    else:
        phi = np.load(filename_phi)

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("virtual coils components shape {}".format(pca_dict[0].components_.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))




    volumes_allbins=build_volume_singular_2Dplus1_cc_allbins(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict,
                                               all_weights, phi, L0,useGPU,nb_rep_center_part=nb_rep_center_part)
    np.save(filename_volumes, volumes_allbins)

    return




@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@set_parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@set_parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@set_parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@set_parameter("useGPU", bool, default=True, description="Use GPU")
@set_parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("incoherent", bool, default=False, description="Use GPU")

def build_volumes_singular_allbins_3D(filename_kdata, filename_b1, filename_weights,filename_phi,dictfile,L0,n_comp,useGPU,dens_adj,nb_rep_center_part,gating_only,incoherent):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    print(incoherent)
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)
    if dens_adj:
        npoint = kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density


    if ((filename_b1 is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    if ((filename_phi is None) and (dictfile is None)):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_phi = str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)

    b1_all_slices_2Dplus1_pca = np.load(filename_b1)

    nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]
    nb_slices=kdata_all_channels_all_slices.shape[-2]
    npoint=kdata_all_channels_all_slices.shape[-1]
    npoint_image=int(npoint/2)
    nb_allspokes=kdata_all_channels_all_slices.shape[1]

    nb_channels=b1_all_slices_2Dplus1_pca.shape[0]
    

    # if nb_slices>nb_slices_b1:
    #     us_b1 = int(nb_slices / nb_slices_b1)
    #     print("B1 map on x{} coarser grid. Interpolating B1 map on a finer grid".format(us_b1))
    #     b1_all_slices_2Dplus1_pca=interp_b1(b1_all_slices_2Dplus1_pca,us=us_b1,start=0)

    #     print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
    #     pca_dict={}
    #     for sl in range(nb_slices):
    #         pca=PCAComplex(n_components_=nb_channels)
    #         pca.explained_variance_ratio_=[1]
    #         pca.components_=np.eye(nb_channels)
    #         pca_dict[sl]=deepcopy(pca)

    if (nb_slices>nb_slices_b1)or(npoint_image>npoint_b1):
        
        print("Regridding b1")
        new_shape=(nb_slices,npoint_image,npoint_image)
        b1_all_slices_2Dplus1_pca=interp_b1_resize(b1_all_slices_2Dplus1_pca,new_shape)


    all_weights = np.load(filename_weights)

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1




    if filename_phi not in os.listdir():
        phi = build_phi(dictfile, L0)
    else:
        phi = np.load(filename_phi)

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))



    radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=incoherent,mode="old")
    volumes_allbins=build_volume_singular_3D_allbins(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, radial_traj,
                                               all_weights, phi, L0,useGPU,nb_rep_center_part=nb_rep_center_part)
    np.save(filename_volumes, volumes_allbins)

    return


@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@set_parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@set_parameter("filename_weights", str, default=None, description="Weights file to simulate undersampling from binning")
@set_parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@set_parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@set_parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("useGPU", bool, default=True, description="Use GPU")
@set_parameter("full_volume", bool, default=False, description="Build full volume")
@set_parameter("in_phase", bool, default=False, description="MRF T1-FF : Select in phase spokes from original MRF sequence")
@set_parameter("out_phase", bool, default=False, description="MRF T1-FF : Select out of phase spokes from original MRF sequence")

def build_volumes_singular(filename_kdata, filename_b1, filename_pca,filename_phi,dictfile,L0,n_comp,nb_rep_center_part,useGPU,filename_weights,full_volume,in_phase,out_phase):
    '''
    Build singular volumes for MRF (no binning)
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''

    filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    print(dico_seqParams)

    nb_gating_spokes=dico_seqParams["alFree"][6]
    print(nb_gating_spokes)

    if (use_navigator_dll)and(nb_gating_spokes>0):
        meas_sampling_mode = dico_seqParams["alFree"][15]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]

    print(meas_sampling_mode)

    undersampling_factor = dico_seqParams["alFree"][9]

    nb_segments = dico_seqParams["alFree"][4]
    nb_slices = int(dico_seqParams["nb_part"])

    if meas_sampling_mode == 1:
        incoherent = False
        mode = None
    elif meas_sampling_mode == 2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode == 3:
        incoherent = True
        mode = "new"

    print(incoherent)

    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular.npy"

    if full_volume:
        filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volume_full.npy"
        if in_phase:
            filename_volumes = filename_volumes.split("_volume_full.npy")[0] + "_volume_full_ip.npy"
        elif out_phase:
            filename_volumes = filename_volumes.split("_volume_full.npy")[0] + "_volume_full_oop.npy"
    print("Loading Kdata")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_segments=kdata_all_channels_all_slices.shape[1]
    npoint=kdata_all_channels_all_slices.shape[-1]

    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None)and(not(incoherent)):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing for stack of stars reco')

    if ((filename_phi is None)and(dictfile is None)and(not(full_volume))):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_phi = str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)

    if filename_b1 is None:
        if (not(incoherent)):
            filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)
        else:
            filename_b1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b1.npy".format(n_comp)

    if filename_pca is None:
        filename_pca = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)


    b1_all_slices_2Dplus1_pca = np.load(filename_b1)


    if full_volume:
        L0=1
        window=8
        phi=np.ones((1,1))
    else:
        if filename_phi not in os.listdir():
            phi=build_phi(dictfile,L0)
        else:
            phi = np.load(filename_phi)

    if filename_weights is not None:
        print("Applying weights mask to k-space")
        all_weights=np.load(filename_weights)
        all_weights=(np.sum(all_weights,axis=0)>0)*1
        kdata_all_channels_all_slices=kdata_all_channels_all_slices*all_weights


    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    #
    print("phi shape {}".format(phi.shape))

    print("Building Singular Volumes")
    if not(incoherent):

        if in_phase:
            selected_spokes=np.r_[300:800,1200:1400]
            #selected_spokes=np.r_[280:580]
        elif out_phase:
            selected_spokes = np.r_[800:1200]
        else:
            selected_spokes=None

        print("Stack of stars - using 2D+1 reconstruction")
        file = open(filename_pca, "rb")
        pca_dict = pickle.load(file)
        file.close()
        print("virtual coils components shape {}".format(pca_dict[0].components_.shape))

        volumes_allbins=build_volume_singular_2Dplus1_cc(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict,
                                                1, phi, L0,nb_rep_center_part=nb_rep_center_part,useGPU=useGPU,selected_spokes=selected_spokes)
    
    else:
        print("Non stack of stars - using 3D reconstruction")
        cond_us = np.zeros((nb_slices, nb_segments))

        cond_us = cond_us.reshape((nb_slices, -1, 8))

        curr_start = 0
        for sl in range(nb_slices):
            cond_us[sl, curr_start::undersampling_factor, :] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % undersampling_factor

        cond_us = cond_us.flatten()
        included_spokes = cond_us
        included_spokes = (included_spokes > 0)

        radial_traj_allspokes = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                         nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        from utils_reco import correct_mvt_kdata_zero_filled
        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj_allspokes, included_spokes, 1)

        weights = weights.reshape(1, -1, 8, nb_slices)
        import math
        nb_rep = math.ceil(nb_slices / undersampling_factor)
        weights_us = np.zeros(shape=(1, 175, 8, nb_rep), dtype=weights.dtype)

        shift = 0

        for sl in range(nb_slices):
            if int(sl / undersampling_factor) < nb_rep:
                weights_us[:, shift::undersampling_factor, :, int(sl / undersampling_factor)] = weights[:,
                                                                                                shift::undersampling_factor,
                                                                                                :, sl]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        weights_us = weights_us.reshape(1, -1, nb_rep)
        weights_us = weights_us[..., None]

        
        radial_traj=Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode,nb_rep_center_part=nb_rep_center_part)
        
        volumes_allbins=build_volume_singular_3D(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, radial_traj,weights_us,phi,L0,useGPU,nb_rep_center_part)
    np.save(filename_volumes, volumes_allbins)

    return



@machine
@set_parameter("filename_kdata", str, default=None, description="MRF raw data")
@set_parameter("nbins", int, default=5, description="Number of bins")
@set_parameter("nkept", int, default=4, description="Number of bins kept")
@set_parameter("nb_gating_spokes", int, default=50, description="Gating spokes count")
@set_parameter("equal_spoke_per_bin", bool, default=False, description="Equal number of spokes per bin")
def generate_random_weights(filename_kdata, nbins,nkept,nb_gating_spokes,equal_spoke_per_bin):
    '''
    Build singular volumes for MRF (no binning)
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    filename_weights = filename_kdata.split("_kdata.npy")[0] + "_weights.npy"
    print("Loading Kdata")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_allspokes=kdata_all_channels_all_slices.shape[1]
    nb_slices=kdata_all_channels_all_slices.shape[2]

    all_weights=[]

    displacement=np.zeros((nb_slices,nb_gating_spokes))
    for sl in range(nb_slices):
        phase=np.random.uniform()*np.pi-np.pi/2
        amplitude=np.random.uniform()*0.2+0.9
        frequency=(np.random.uniform()*0.5+1)/nb_gating_spokes
        displacement[sl]=amplitude*np.sin(2*np.pi*np.arange(nb_gating_spokes)*frequency+phase)

    displacement=displacement.flatten()

    displacement_for_binning = displacement
    if not(equal_spoke_per_bin):
        max_bin = np.max(displacement_for_binning)
        min_bin = np.min(displacement_for_binning)
        bin_width = (max_bin-min_bin)/(nbins)
        min_bin = np.min(displacement_for_binning) 
        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        
        retained_categories=list(range(nbins-nkept+1,nbins+1))
    
    else:
        disp_sorted_index = np.argsort(displacement_for_binning)
        count_disp = len(disp_sorted_index)
        disp_width = int(count_disp / nbins)
        bins = []
        for j in range(1, nbins):
            bins.append(np.sort(displacement_for_binning)[j * disp_width])
        retained_categories=list(range(nbins-nkept,nbins))

    categories = np.digitize(displacement_for_binning, bins)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()
    
    print(df_groups)
    print(retained_categories)
    groups=[]
    for cat in retained_categories:
        groups.append(categories==cat)

    spoke_groups = np.argmin(np.abs(
            np.arange(0, nb_allspokes * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_allspokes * nb_slices,
                                                                              nb_allspokes / nb_gating_spokes).reshape(1,
                                                                                                                      -1)),
            axis=-1)

    spoke_groups = spoke_groups.reshape(nb_slices, nb_allspokes)
    spoke_groups[:-1, -int(nb_allspokes / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_allspokes / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()
    dico_traj_retained={}
    radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=800,nb_slices=nb_slices*1,incoherent=False,mode="old",nspoke_per_z_encoding=8)

    for j,g in enumerate(groups):
        retained_nav_spokes_index = np.argwhere(g).flatten()
        included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
        included_spokes[::int(nb_allspokes/nb_gating_spokes)]=False
        # included_spokes=included_spokes.reshape(nb_slices,nb_allspokes)
        # included_spokes=np.moveaxis(included_spokes,0,1)[None]
        
        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, 1)
        dico_traj_retained[j]=weights
        #all_weights.append(included_spokes)
    
    
    for gr in dico_traj_retained.keys():
        all_weights.append(np.expand_dims(dico_traj_retained[gr],axis=-1))
    all_weights=np.array(all_weights)
    

    print(all_weights.shape)

    np.save(filename_weights, all_weights)

    return


@machine
@set_parameter("filename_volume", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("mu", float, default=1, description="Gradient step size")
@set_parameter("mu_TV", float, default=1, description="Spatial Regularization")
@set_parameter("lambda_wav", float, default=0.5e-5, description="Lambda wavelet")
@set_parameter("lambda_LLR", float, default=0.0005, description="Lambda LLR")
@set_parameter("mu_bins", float, default=None, description="Interbin regularization")
@set_parameter("niter", int, default=None, description="Number of iterations")
@set_parameter("suffix", str, default="", description="Suffix")
@set_parameter("gamma", float, default=None, description="Gamma Correction")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@set_parameter("use_wavelet", bool, default=False, description="Wavelet regularization instead of TV")
@set_parameter("use_proximal_TV", bool, default=False, description="Proximal gradient (FISTA) instead of gradient descent")
@set_parameter("us", int, default=1, description="Undersampling")
@set_parameter("use_LLR", bool, default=False, description="LLR regularization instead of TV")
@set_parameter("block", str, default="4,10,10", description="Block size for LLR regularization")



def build_volumes_iterative_allbins(filename_volume,filename_b1,filename_weights,mu,mu_TV,mu_bins,niter,gamma,suffix,gating_only,dens_adj,nb_rep_center_part,select_first_rep,use_proximal_TV,use_wavelet,lambda_wav,us,use_LLR,lambda_LLR,block):
    filename_target=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_denoised{}.npy".format(suffix)

    if gamma is not None:
        filename_target=filename_target.split(".npy")[0]+"_gamma_{}.npy".format(str(gamma).replace(".","_"))
    
    weights_TV=np.array([1.0,0.2,0.2])
    weights_TV/=np.sum(weights_TV)
    print("Loading Volumes")
    volumes=np.load(filename_volume)
    #To fit the input expected by the undersampling function with L0=1 in our case
    if volumes.ndim==4:
        volumes=np.expand_dims(volumes,axis=1)

    volumes=volumes.astype("complex64")

    if filename_weights is None:
        filename_weights = (filename_volume.split("_volumes_allbins.npy")[0] + "_weights.npy").replace("_no_densadj","")
    if gating_only:
        filename_weights=str.replace(filename_weights,"_no_dcomp","")

    

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)

    if gating_only:
        all_weights=(all_weights>0)*1

    if nb_rep_center_part>1:
        all_weights=weights_aggregate_center_part(all_weights,nb_rep_center_part,select_first_rep)

    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us

    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint=2*volumes.shape[-1]
    nbins=volumes.shape[0]

    radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

    ntimesteps=all_weights.shape[1]


    volumes0=copy(volumes)
    volumes=mu*volumes0

    if use_wavelet:
        print("Wavelet Denoising")
        wav_level = 4
        wav_type="db4"

        lambd = lambda_wav

        alpha_denoised = []

        

        for gr in tqdm(range(nbins)):
            print("#################   Denoising Bin {}   #######################".format(gr))
            weights = all_weights[gr]
            vol_denoised_log=[volumes[gr]]
            coefs = pywt.wavedecn(volumes[gr], wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
            u, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
            u0 = u
            y = u
            t = 1
            u = pywt.threshold(y, lambd * mu)

            print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

            vol_denoised_log.append(pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)))

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u
            t = t_next

            for i in range(niter):
                u_prev = u
                x = pywt.array_to_coeffs(y, slices)
                x = pywt.waverecn(x, wav_type, mode="periodization",axes=(1,2,3))

                volumesi = undersampling_operator_singular_new(x, radial_traj_2D,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
                #volumesi = volumesi.squeeze()
                coefs = pywt.wavedecn(volumesi, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
                grad_y, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
                grad = grad_y - u0
                y = y - mu * grad

                u = pywt.threshold(y, lambd * mu)
                print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

                t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
                y = u + (t - 1) / t_next * (u - u_prev)
                t = t_next

                if (i%1==0):
                    vol_denoised_log.append(pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)))

            vol_denoised_log=np.array(vol_denoised_log)
            filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volume_denoised_gr{}{}.npy".format(gr,suffix)
            np.save(filename_target_intermediate,vol_denoised_log)

            alpha_denoised.append(u)

        alpha_denoised = np.array(alpha_denoised)
        volumes = [pywt.waverecn(pywt.array_to_coeffs(alpha, slices), wav_type, mode="periodization",axes=(1,2,3)) for alpha in
                            alpha_denoised]
        volumes = np.array(volumes)


    elif use_LLR:
        print("LLR denoising")
        blck = np.array(block.split(",")).astype(int)
        strd = blck
        lambd = lambda_LLR
        threshold = lambd * mu

        volumes_denoised = []

        for gr in tqdm(range(nbins)):
            print("#################   Denoising Bin {}   #######################".format(gr))
            weights = all_weights[gr]
            u = mu * volumes[gr]
            u0 = volumes[gr]
            y = u
            t = 1

            u = proj_LLR(u.squeeze(), strd, blck, threshold)

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u
            t = t_next
            for i in range(niter):
                u_prev = u
                if u.ndim == 3:
                    u = np.expand_dims(u, axis=0)
                volumesi = undersampling_operator_singular_new(u, radial_traj_2D,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)

                grad = volumesi - u0
                y = y - mu * grad

                u = proj_LLR(y.squeeze(), strd, blck, threshold)

                t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
                y = u + (t - 1) / t_next * (u - u_prev)
                t = t_next

            volumes_denoised.append(u)

        volumes = np.array(volumes_denoised)


    else:


        for i in tqdm(range(niter)):
            print("Correcting volumes for iteration {}".format(i))
            all_grad_norm=0
            for gr in tqdm(range(nbins)):
                volumesi=undersampling_operator_singular_new(volumes[gr],radial_traj_2D, b1_all_slices_2Dplus1_pca,weights=all_weights[gr],density_adj=dens_adj)

                grad = volumesi - volumes0[gr]
                volumes[gr] = volumes[gr] - mu * grad



                if (mu_TV is not None)and(not(mu_TV==0)):
                    print("Applying TV regularization")

                    grad_norm=np.linalg.norm(grad)
                    all_grad_norm+=grad_norm**2
                    print("grad norm {}".format(grad_norm))
                    del grad
                    grad_TV=np.zeros_like(volumes[gr])
                    for ts in tqdm(range(ntimesteps)):
                        for ind_w, w in (enumerate(weights_TV)):
                            if w > 0:
                                grad_TV[ts] += (w * grad_J_TV(volumes[gr,ts], ind_w,is_weighted=False,shift=0))

                            #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                    grad_TV_norm = np.linalg.norm(grad_TV)
                                # signals = matched_signals + mu * grad

                    print("grad_TV_norm {}".format(grad_TV_norm))

                    volumes[gr] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                    del grad_TV
                    del grad_TV_norm

            all_grad_norm=np.sqrt(all_grad_norm)
            if (mu_bins is not None)and (not(mu_bins==0)):
                grad_TV_bins=grad_J_TV(volumes,0,is_weighted=False,shift=0)
                grad_TV_bins_norm = np.linalg.norm(grad_TV_bins)

                volumes -= mu * mu_bins * grad_TV_bins/grad_TV_bins_norm*all_grad_norm
                print("grad_TV_bins norm {}".format(grad_TV_bins_norm))
                del grad_TV_bins

            if (i%5==0):
                filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_denoised_it{}{}.npy".format(i,suffix)
                np.save(filename_target_intermediate,volumes)

    volumes=np.squeeze(volumes)
    if gamma is not None:
        for gr in range(volumes.shape[0]):
            volumes[gr]=gamma_transform(volumes[gr],gamma)



    np.save(filename_target,volumes)
    #volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    #np.save(filename_volumes,volumes_allbins)

    return


@machine
@set_parameter("filename_volume", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_weights", str, default=None, description="Undersampling weights")
@set_parameter("filename_seqParams", str, default=None, description="Undersampling weights")
@set_parameter("mu", float, default=1, description="Gradient step size")
@set_parameter("mu_TV", float, default=1, description="Spatial Regularization")
@set_parameter("lambda_wav", float, default=0.5e-5, description="Lambda wavelet")
@set_parameter("lambda_LLR", float, default=0.0005, description="Lambda LLR")
@set_parameter("niter", int, default=None, description="Number of iterations")
@set_parameter("suffix", str, default="", description="Suffix")
@set_parameter("gamma", float, default=None, description="Gamma Correction")
@set_parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@set_parameter("use_wavelet", bool, default=False, description="Wavelet regularization instead of TV")
@set_parameter("use_LLR", bool, default=False, description="LLR regularization instead of TV")
@set_parameter("block", str, default="2,10,10", description="Block size for LLR regularization")

def build_volumes_iterative(filename_volume, filename_b1,filename_weights,filename_seqParams, mu, mu_TV, niter, gamma,
                                    suffix, dens_adj, use_wavelet, lambda_wav, use_LLR, lambda_LLR, block):
    filename_target = filename_volume.split(".npy")[0] + "_denoised{}.npy".format(
        suffix)

    if gamma is not None:
        filename_target = filename_target.split(".npy")[0] + "_gamma_{}.npy".format(str(gamma).replace(".", "_"))

    print(filename_target)

    weights_TV = np.array([1.0, 0.2, 0.2])
    weights_TV /= np.sum(weights_TV)
    print("Loading Volumes")
    volumes = np.load(filename_volume)
    # To fit the input expected by the undersampling function with L0=1 in our case

    volumes = volumes.astype("complex64")


    b1_all_slices_2Dplus1_pca = np.load(filename_b1)
    incoherent=False #stack of stars by default

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]
        nb_gating_spokes=dico_seqParams["alFree"][6]
        if (use_navigator_dll)or(nb_gating_spokes>0):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        undersampling_factor = dico_seqParams["alFree"][9]

        nb_segments = dico_seqParams["alFree"][4]
        nb_slices = int(dico_seqParams["nb_part"])

        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

    if filename_weights is not None:
        print("Applying weights mask to k-space")
        weights=np.load(filename_weights)
    else:
        weights=1



    print("Volumes shape {}".format(volumes.shape))

    nb_allspokes = 1400
    npoint = 2 * volumes.shape[-1]
    nbins = volumes.shape[0]

    if not(incoherent):
        radial_traj = Radial(total_nspokes=nb_allspokes, npoint=npoint)
    else:
        radial_traj = Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)

        cond_us = np.zeros((nb_slices, nb_segments))

        cond_us = cond_us.reshape((nb_slices, -1, 8))

        curr_start = 0
        for sl in range(nb_slices):
            cond_us[sl, curr_start::undersampling_factor, :] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % undersampling_factor

        cond_us = cond_us.flatten()
        included_spokes = cond_us
        included_spokes = (included_spokes > 0)

        radial_traj_allspokes = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                         nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        from utils_reco import correct_mvt_kdata_zero_filled
        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj_allspokes, included_spokes, 1)

        weights = weights.reshape(1, -1, 8, nb_slices)
        import math
        nb_rep = math.ceil(nb_slices / undersampling_factor)
        weights_us = np.zeros(shape=(1, 175, 8, nb_rep), dtype=weights.dtype)

        shift = 0

        for sl in range(nb_slices):
            if int(sl / undersampling_factor) < nb_rep:
                weights_us[:, shift::undersampling_factor, :, int(sl / undersampling_factor)] = weights[:,
                                                                                                shift::undersampling_factor,
                                                                                                :, sl]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        weights_us = weights_us.reshape(1, -1, nb_rep)
        weights_us = weights_us[..., None]

    volumes0 = copy(volumes)
    volumes = mu * volumes0

    if use_wavelet:

        wav_level = None
        wav_type = "db4"
        axes=(2,3)

        lambd = lambda_wav

        alpha_denoised = []

        vol_denoised_log = [volumes]
        coefs = pywt.wavedecn(volumes, wav_type, level=wav_level, mode="periodization", axes=axes)
        u, slices = pywt.coeffs_to_array(coefs, axes=axes)
        u0 = u
        y = u
        t = 1
        u = pywt.threshold(y, lambd * mu)
        #u=np.maximum(np.abs(y)-lambd * mu,0)/np.abs(y)*y
        print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

        vol_denoised_log.append(
            pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization", axes=axes))

        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u
        t = t_next

        for i in range(niter):
            u_prev = u
            x = pywt.array_to_coeffs(y, slices)
            x = pywt.waverecn(x, wav_type, mode="periodization", axes=axes)

            print("x.shape : {}".format(x.shape))
            if not(incoherent):
                volumesi = undersampling_operator_singular_new(x, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
            else:


                volumesi = undersampling_operator_singular(x, radial_traj,
                                                         b1_all_slices_2Dplus1_pca,
                                                               density_adj=dens_adj,weights=weights_us)
                #volumesi = x

            print("volumesi.shape : {}".format(volumesi.shape))
            # volumesi = volumesi.squeeze()
            coefs = pywt.wavedecn(volumesi, wav_type, level=wav_level, mode="periodization", axes=axes)
            grad_y, slices = pywt.coeffs_to_array(coefs, axes=axes)
            grad = grad_y - u0/mu
            y = y - mu * grad

            u = pywt.threshold(y, lambd * mu)
            #u=grad_y

            print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u + (t - 1) / t_next * (u - u_prev)
            t = t_next

            if (i % 1 == 0):
                vol_denoised_log.append(
                    pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization", axes=axes))

        vol_denoised_log = np.array(vol_denoised_log)
        filename_target_intermediate = filename_volume.split("_volumes_allbins.npy")[
                                           0] + "_volume_denoised_{}.npy".format(suffix)
        np.save(filename_target_intermediate, vol_denoised_log)

        alpha_denoised.append(u)

        alpha_denoised = np.array(alpha_denoised)
        volumes = [pywt.waverecn(pywt.array_to_coeffs(alpha, slices), wav_type, mode="periodization", axes=axes)
                   for alpha in
                   alpha_denoised]
        volumes = np.array(volumes).squeeze()
        #volumes=volumesi


    elif use_LLR:
        blck = np.array(block.split(",")).astype(int)
        

        volumes_denoised = []

        u = mu * volumes
        u0 = volumes
        y = u
        t = 1

        if (u.ndim==(len(blck)+1)):
            blck=np.array([u.shape[0]]+list(blck))

        strd = blck
        lambd = lambda_LLR
        threshold = lambd * mu

        print(blck)
        
        

        print(u.shape)
        u = proj_LLR(u.squeeze(), strd, blck, threshold)
        print(u.shape)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u
        t = t_next
        for i in range(niter):
            u_prev = u
            if u.ndim == 3:
                u = np.expand_dims(u, axis=0)
            
            if not(incoherent):
                volumesi = undersampling_operator_singular_new(u, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
            else:
                volumesi = undersampling_operator_singular(u, radial_traj,
                                                               b1_all_slices_2Dplus1_pca,
                                                               density_adj=dens_adj)

            grad = volumesi - u0/mu
            y = y - mu * grad

            u = proj_LLR(y.squeeze(), strd, blck, threshold)

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u + (t - 1) / t_next * (u - u_prev)
            t = t_next

        volumes_denoised.append(u)

        volumes = np.array(volumes_denoised).squeeze()


    else:

        for i in tqdm(range(niter)):
            print("Correcting volumes for iteration {}".format(i))
            all_grad_norm = 0
            for gr in tqdm(range(nbins)):
                volumesi = undersampling_operator_singular_new(volumes[gr], radial_traj, b1_all_slices_2Dplus1_pca,
                                                               weights=all_weights[gr], density_adj=dens_adj)

                grad = volumesi - volumes0[gr]
                volumes[gr] = volumes[gr] - mu * grad

                if (mu_TV is not None) and (not (mu_TV == 0)):
                    print("Applying TV regularization")

                    grad_norm = np.linalg.norm(grad)
                    all_grad_norm += grad_norm ** 2
                    print("grad norm {}".format(grad_norm))
                    del grad
                    grad_TV = np.zeros_like(volumes[gr])
                    for ts in tqdm(range(ntimesteps)):
                        for ind_w, w in (enumerate(weights_TV)):
                            if w > 0:
                                grad_TV[ts] += (w * grad_J_TV(volumes[gr, ts], ind_w, is_weighted=False, shift=0))

                            # grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                    grad_TV_norm = np.linalg.norm(grad_TV)
                    # signals = matched_signals + mu * grad

                    print("grad_TV_norm {}".format(grad_TV_norm))

                    volumes[gr] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                    del grad_TV
                    del grad_TV_norm

            all_grad_norm = np.sqrt(all_grad_norm)
            if (mu_bins is not None) and (not (mu_bins == 0)):
                grad_TV_bins = grad_J_TV(volumes, 0, is_weighted=False, shift=0)
                grad_TV_bins_norm = np.linalg.norm(grad_TV_bins)

                volumes -= mu * mu_bins * grad_TV_bins / grad_TV_bins_norm * all_grad_norm
                print("grad_TV_bins norm {}".format(grad_TV_bins_norm))
                del grad_TV_bins

            if (i % 5 == 0):
                filename_target_intermediate = filename_volume.split("_volumes_allbins.npy")[
                                                   0] + "_volumes_allbins_denoised_it{}{}.npy".format(i, suffix)
                np.save(filename_target_intermediate, volumes)

    volumes = np.squeeze(volumes)
    if gamma is not None:
        for gr in range(volumes.shape[0]):
            volumes[gr] = gamma_transform(volumes[gr], gamma)

    np.save(filename_target, volumes)
    # volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    # np.save(filename_volumes,volumes_allbins)

    return

@machine
@set_parameter("filename_volume", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("file_deformation", str, default=None, description="Deformation")
@set_parameter("index_ref", int, default=0, description="Registration reference")
@set_parameter("mu", float, default=2, description="Gradient step size")
@set_parameter("mu_TV", float, default=None, description="Spatial Regularization")
@set_parameter("niter", int, default=None, description="Number of iterations")
@set_parameter("suffix", str, default="", description="Suffix")
@set_parameter("gamma", float, default=None, description="Gamma Correction")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("beta", float, default=None, description="Relative importance of registered volumes vs volume of reference")
@set_parameter("interp", str, default=None, description="Registration interpolation")
@set_parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@set_parameter("lambda_wav", float, default=0.5e-5, description="Lambda wavelet")
@set_parameter("use_wavelet", bool, default=False, description="Wavelet regularization instead of TV")
@set_parameter("us", int, default=1, description="Undersampling")
@set_parameter("kept_bins",str,default=None,description="Bins to keep")

def build_volumes_iterative_allbins_registered(filename_volume,filename_b1,filename_weights,mu,mu_TV,niter,gamma,suffix,gating_only,dens_adj,nb_rep_center_part,file_deformation,index_ref,beta,interp,select_first_rep,lambda_wav,use_wavelet,us,kept_bins):
    filename_target=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_ref{}{}.npy".format(index_ref,suffix)

    if gamma is not None:
        filename_target=filename_target.split(".npy")[0]+"_gamma_{}.npy".format(str(gamma).replace(".","_"))
    
    if interp is None:
        interp=cv2.INTER_LINEAR

    elif interp=="nearest":
        interp=cv2.INTER_NEAREST
    
    elif interp=="cubic":
        interp=cv2.INTER_CUBIC


    weights_TV=np.array([1.0,0.2,0.2])
    weights_TV/=np.sum(weights_TV)
    print("Loading Volumes")
    volumes=np.load(filename_volume)
    #To fit the input expected by the undersampling function with L0=1 in our case
    #volumes=np.expand_dims(volumes,axis=1)

    volumes=volumes.astype("complex64")

    if volumes.ndim==4:
        #To fit the input expected by the undersampling function with L0=1 in our case
        volumes=np.expand_dims(volumes,axis=1)
        shift=0
    else:
        shift=1

    if filename_weights is None:
        filename_weights = (filename_volume.split("_volumes_allbins.npy")[0] + "_weights.npy").replace("_no_densadj","")
    if gating_only:
        filename_weights=str.replace(filename_weights,"_no_dcomp","")

    

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)



    if gating_only:
        print("Using weights only for gating")
        all_weights=(all_weights>0)*1

    if nb_rep_center_part>1:
        all_weights=weights_aggregate_center_part(all_weights,nb_rep_center_part,select_first_rep)


    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us


    if kept_bins is not None:
        kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
        print(kept_bins_list)
        volumes=volumes[kept_bins_list]
        all_weights=all_weights[kept_bins_list]


    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint_image=volumes.shape[-1]
    npoint=2*npoint_image
    nbins=volumes.shape[0]
    nb_slices=volumes.shape[2]

    radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

    if file_deformation is None:#identity by default
        X,Y=np.meshgrid(np.arange(npoint_image),np.arange(npoint_image))
        def_identity_x=np.tile(np.expand_dims(X,axis=(0,1)),(nbins,nb_slices,1,1))
        def_identity_y=np.tile(np.expand_dims(Y,axis=(0,1)),(nbins,nb_slices,1,1))
        deformation_map=np.stack([def_identity_x,def_identity_y],axis=0)
        

    else:
        deformation_map=np.load(file_deformation)

    nb_slices_def=deformation_map.shape[2]
    npoint_def=deformation_map.shape[-1]

    # if nb_slices>nb_slices_def:
    #     us_def = int(nb_slices / nb_slices_def)
    #     print("Deformation map on x{} coarser grid. Interpolating deformation map on a finer grid".format(us_def))
    #     deformation_map=interp_deformation(deformation_map,us=us_def,start=0)

    # nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    # npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]
    # nb_channels=b1_all_slices_2Dplus1_pca.shape[0]

    # if nb_slices>nb_slices_b1:
    #     us_b1 = int(nb_slices / nb_slices_b1)
    #     print("B1 map on x{} coarser grid. Interpolating B1 map on a finer grid".format(us_b1))
    #     b1_all_slices_2Dplus1_pca=interp_b1(b1_all_slices_2Dplus1_pca,us=us_b1,start=0)
    #     print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
    #     pca_dict={}
    #     for sl in range(nb_slices):
    #         pca=PCAComplex(n_components_=nb_channels)
    #         pca.explained_variance_ratio_=[1]
    #         pca.components_=np.eye(nb_channels)
    #         pca_dict[sl]=deepcopy(pca)

    if (nb_slices>nb_slices_def)or(npoint_image>npoint_def):
        print("Regridding deformation map")
        new_shape=(nb_slices,npoint_image,npoint_image)
        deformation_map=interp_deformation_resize(deformation_map,new_shape)

    nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]
    nb_channels=b1_all_slices_2Dplus1_pca.shape[0]

    if (nb_slices>nb_slices_b1)or(npoint_image>npoint_b1):
        print("Regridding B1")
        new_shape=(nb_slices,npoint_image,npoint_image)
        b1_all_slices_2Dplus1_pca=interp_b1_resize(b1_all_slices_2Dplus1_pca,new_shape)
        print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=nb_channels)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(nb_channels)
            pca_dict[sl]=deepcopy(pca)

    deformation_map=change_deformation_map_ref(deformation_map,index_ref)

    print("Calculating inverse deformation map")

    if file_deformation is None:#identity by default
        inv_deformation_map=deformation_map
    else:
        inv_deformation_map = np.zeros_like(deformation_map)
        for gr in tqdm(range(nbins)):
            inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr])
        
    print(volumes.shape)
    volumes_registered=np.zeros(volumes.shape[1:],dtype=volumes.dtype)
    print(volumes_registered.shape)


    if file_deformation is None:
        print("No deformation - summing volumes for all bins")
        for gr in range(nbins):
            volumes_registered+=volumes[gr].squeeze()
    else:

        print("Registering initial volumes")
        for gr in range(nbins):
            if beta is None:
                volumes_registered+=apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp=interp)
            else:
                if gr==index_ref:
                    volumes_registered+=beta*apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp=interp)
                else:
                    volumes_registered += (1-beta) * apply_deformation_to_complex_volume(volumes[gr].squeeze(),
                                                                                    deformation_map[:, gr],interp=interp)


    volumes0=copy(volumes_registered)

    volumes_registered=mu*volumes0

    print("volumes_registered.shape {}".format(volumes_registered.shape))

    print(use_wavelet)
    if use_wavelet:

        wav_level = None
        wav_type="db4"

        lambd = lambda_wav

        print("Wavelet regularization penalty {}".format(lambd))

        if volumes_registered.ndim==3:
            volumes_registered=np.expand_dims(volumes_registered,axis=0)

        coefs = pywt.wavedecn(volumes_registered, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
        u, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
        u0 = u
        y = u
        t = 1

        u = pywt.threshold(y, lambd * mu)

        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u 
        t = t_next

        for i in range(niter):
            u_prev = u
            x = pywt.array_to_coeffs(y, slices)
            x = pywt.waverecn(x, wav_type, mode="periodization",axes=(1,2,3))

        
            for gr in tqdm(range(nbins)):
                
                volumesi=apply_deformation_to_complex_volume(x,inv_deformation_map[:,gr],interp=interp)
                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)
                print("volumesi.shape {}".format(volumesi.shape))
                volumesi=undersampling_operator_singular_new(volumesi,radial_traj_2D, b1_all_slices_2Dplus1_pca,weights=all_weights[gr],density_adj=dens_adj)
                volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map[:,gr],interp=interp)

                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= beta
                    else:
                        volumesi *= (1 - beta)

                if gr==0:
                    final_volumesi=volumesi
                else:
                    final_volumesi+=volumesi
            #volumesi = volumesi.squeeze()
            coefs = pywt.wavedecn(final_volumesi, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
            grad_y, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
            grad = grad_y - u0
            y = y - mu * grad

            u = pywt.threshold(y, lambd * mu)

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u + (t - 1) / t_next * (u - u_prev)
            t = t_next

            if (i%1==0):    
                filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_ref{}_it{}{}.npy".format(index_ref,i,suffix)
                volumes_registered = pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)).squeeze()
                np.save(filename_target_intermediate,volumes_registered)

        volumes_registered = pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)).squeeze()



    else:
        for i in tqdm(range(niter)):
            print("Correcting volumes for iteration {}".format(i))
            all_grad_norm=0
            for gr in tqdm(range(nbins)):
                
                volumesi=apply_deformation_to_complex_volume(volumes_registered,inv_deformation_map[:,gr],interp=interp)
                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)
                print("volumesi.shape {}".format(volumesi.shape))
                volumesi=undersampling_operator_singular_new(volumesi,radial_traj_2D, b1_all_slices_2Dplus1_pca,weights=all_weights[gr],density_adj=dens_adj)
                volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map[:,gr],interp=interp)

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= beta
                    else:
                        volumesi *= (1 - beta)

                if gr==0:
                    final_volumesi=volumesi
                else:
                    final_volumesi+=volumesi

            grad = final_volumesi - volumes0
            volumes_registered = volumes_registered - mu * grad
            


            if (mu_TV is not None)and(not(mu_TV==0)):
                print("Applying TV regularization")
                                
                grad_norm=np.linalg.norm(grad)
                all_grad_norm+=grad_norm**2
                print("grad norm {}".format(grad_norm))
                del grad
                grad_TV=np.zeros_like(volumes_registered)
                
                for ind_w, w in (enumerate(weights_TV)):
                    if w > 0:
                        grad_TV += (w * grad_J_TV(volumes_registered, ind_w,is_weighted=False,shift=shift))

                            #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV)
                                # signals = matched_signals + mu * grad

                print("grad_TV_norm {}".format(grad_TV_norm))

                volumes_registered -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

            

            if (i%1==0):    
                filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_ref{}_it{}{}.npy".format(index_ref,i,suffix)
                np.save(filename_target_intermediate,volumes_registered)

    volumes_registered=np.squeeze(volumes_registered)
    if gamma is not None:
        volumes_registered=gamma_transform(volumes_registered)



    np.save(filename_target,volumes_registered)
    #volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    #np.save(filename_volumes,volumes_allbins)

    return

@machine
@set_parameter("filename_volume", str, default=None, description="MRF raw data")
@set_parameter("filename_b1", str, default=None, description="B1")
@set_parameter("filename_weights", str, default=None, description="Motion bin weights")
@set_parameter("file_deformation", str, default=None, description="Deformation")
@set_parameter("mu", float, default=1, description="Gradient step size")
@set_parameter("mu_TV", float, default=None, description="Spatial Regularization")
@set_parameter("niter", int, default=None, description="Number of iterations")
@set_parameter("suffix", str, default="", description="Suffix")
@set_parameter("gamma", float, default=None, description="Gamma Correction")
@set_parameter("gating_only", bool, default=False, description="Use weights only for gating")
@set_parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@set_parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@set_parameter("beta", float, default=None, description="Relative importance of registered volumes vs volume of reference")
@set_parameter("interp", str, default=None, description="Registration interpolation")
@set_parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@set_parameter("us", int, default=1, description="Undersampling")

def build_volumes_iterative_allbins_registered_allindex(filename_volume,filename_b1,filename_weights,mu,mu_TV,niter,gamma,suffix,gating_only,dens_adj,nb_rep_center_part,file_deformation,beta,interp,select_first_rep,us):
    filename_target=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_allindex{}.npy".format(suffix)

    if gamma is not None:
        filename_target=filename_target.split(".npy")[0]+"_gamma_{}.npy".format(str(gamma).replace(".","_"))
    
    weights_TV=np.array([1.0,0.2,0.2])
    weights_TV/=np.sum(weights_TV)
    print("Loading Volumes")
    volumes=np.load(filename_volume)
    
    if volumes.ndim==4:
        #To fit the input expected by the undersampling function with L0=1 in our case
        volumes=np.expand_dims(volumes,axis=1)
        shift=0
    else:
        shift=1
    
    if interp is None:
        interp=cv2.INTER_LINEAR

    elif interp=="nearest":
        interp=cv2.INTER_NEAREST
    
    elif interp=="cubic":
        interp=cv2.INTER_CUBIC

    volumes=volumes.astype("complex64")

    if filename_weights is None:
        filename_weights = (filename_volume.split("_volumes_allbins.npy")[0] + "_weights.npy").replace("_no_densadj","")
    if gating_only:
        filename_weights=str.replace(filename_weights,"_no_dcomp","")

    

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)

    if gating_only:
        all_weights=(all_weights>0)*1

    if nb_rep_center_part>1:
        all_weights=weights_aggregate_center_part(all_weights,nb_rep_center_part,select_first_rep)

    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us

    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint=2*volumes.shape[-1]
    nbins=volumes.shape[0]

    radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

    deformation_map=np.load(file_deformation)


    deformation_map_allindex=[]
    inv_deformation_map_allindex=[]
    volumes_registered_allindex=[]

    print("Extraction deformation map for all bin reference")
    for index_ref in range(nbins):
        deformation_map_allindex.append(change_deformation_map_ref(deformation_map,index_ref))

    print("Building inverse deformation map for all bin reference")
    for index_ref in range(nbins):
        deformation_map=deformation_map_allindex[index_ref]
        print("Calculating inverse deformation map")
        inv_deformation_map = np.zeros_like(deformation_map)
        for gr in tqdm(range(nbins)):
            inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr])
        inv_deformation_map_allindex.append(inv_deformation_map)

        volumes_registered=np.zeros(volumes.shape[1:],dtype=volumes.dtype)
        

        print("Registering initial volumes")
        for gr in range(nbins):

            if beta is None:
                volumes_registered+=apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp)

            else:
                if gr==index_ref:
                    volumes_registered+=beta*apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp)
                else:
                    volumes_registered += (1-beta) * apply_deformation_to_complex_volume(volumes[gr].squeeze(),
                                                                                     deformation_map[:, gr],interp)
        volumes_registered_allindex.append(volumes_registered)
    
    deformation_map_allindex=np.array(deformation_map_allindex)
    inv_deformation_map_allindex=np.array(inv_deformation_map_allindex)

    volumes_registered_allindex=np.array(volumes_registered_allindex)
    volumes0_allindex=copy(volumes_registered_allindex)
    volumes_registered_allindex=mu*volumes0_allindex


    for i in tqdm(range(niter)):
        print("Correcting volumes for iteration {}".format(i))
        for index_ref in range(nbins):
            all_grad_norm=0
            for gr in tqdm(range(nbins)):
                
                volumesi=apply_deformation_to_complex_volume(volumes_registered_allindex[index_ref],inv_deformation_map_allindex[index_ref,:,gr],interp)
                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)
                volumesi=undersampling_operator_singular_new(volumesi,radial_traj_2D, b1_all_slices_2Dplus1_pca,weights=all_weights[gr],density_adj=dens_adj)
                volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map_allindex[index_ref,:,gr],interp)

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= beta
                    else:
                        volumesi *= (1 - beta)

                if gr==0:
                    final_volumesi=volumesi
                else:
                    final_volumesi+=volumesi

            grad = final_volumesi - volumes0_allindex[index_ref]
            volumes_registered_allindex[index_ref] = volumes_registered_allindex[index_ref] - mu * grad
            


            if (mu_TV is not None)and(not(mu_TV==0)):
                print("Applying TV regularization")
                                
                grad_norm=np.linalg.norm(grad)
                all_grad_norm+=grad_norm**2
                print("grad norm {}".format(grad_norm))
                del grad
                grad_TV=np.zeros_like(volumes_registered_allindex[index_ref])
                
                for ind_w, w in (enumerate(weights_TV)):
                    if w > 0:
                        grad_TV += (w * grad_J_TV(volumes_registered_allindex[index_ref], ind_w,is_weighted=False,shift=shift))

                            #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV)
                                # signals = matched_signals + mu * grad

                print("grad_TV_norm {}".format(grad_TV_norm))

                volumes_registered_allindex[index_ref] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

        

        if (i%5==0):    
            filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_allindex_it{}{}.npy".format(i,suffix)
            np.save(filename_target_intermediate,volumes_registered_allindex)

    volumes_registered_allindex=np.squeeze(volumes_registered_allindex)
    if gamma is not None:
        volumes_registered_allindex=gamma_transform(volumes_registered_allindex)



    np.save(filename_target,volumes_registered_allindex)
    #volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    #np.save(filename_volumes,volumes_allbins)

    return


@machine
@set_parameter("sequence_file", str, default="mrf_sequence_adjusted.json", description="Sequence File")
@set_parameter("reco", float, default=4, description="Recovery (s)")
@set_parameter("min_TR_delay", float, default=1.14, description="TR delay (ms)")
@set_parameter("dictconf", str, default="mrf_dictconf_Dico2_Invivo_overshoot.json", description="Dictionary grid")
@set_parameter("dictconf_light", str, default="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot.json", description="Coarse dictionary grid (clustering)")
@set_parameter("inversion", bool, default=True, description="Use initial inversion")
@set_parameter("TI", float, default=None, description="Inversion time (ms)")
def generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,inversion,TI):
    with open(sequence_file,"r") as file:
        sequence_config=json.load(file)

    TR_list,FA_list,TE_list=load_sequence_file(sequence_file,reco,min_TR_delay/1000)

    new_sequence_file=str.split(sequence_file,".json")[0]+"_{}.json".format(np.round(min_TR_delay,2))
    print(new_sequence_file)
    if TI is not None:
        new_sequence_file = str.replace(new_sequence_file,".json","_TI{}.json".format("_".join(str.split(str(TI),"."))))

    print(new_sequence_file)

    if inversion:
        generate=generate_epg_dico_T1MRFSS
    else:
        generate=generate_epg_dico_T1MRFSS_NoInv
    generate(new_sequence_file,dictconf,FA_list,TE_list,reco,min_TR_delay/1000,TI=TI)
    generate(new_sequence_file,dictconf_light,FA_list,TE_list,reco,min_TR_delay/1000,TI=TI)
    return




@machine
@set_parameter("filemap", str, default=None, description="MRF maps (.pkl)")
@set_parameter("fileseq", str, default="mrf_sequence_adjusted.json", description="Sequence File")
@set_parameter("spacing", [float,float,float], default=[1,1,5], description="Voxel size")
@set_parameter("reorient", bool, default=True, description="Reorient to match usual orientation")
@set_parameter("filename", str, default=None, description=".dat file for adding geometry if necessary")
def generate_dixon_volumes_for_segmentation(filemap,fileseq,spacing,reorient,filename):
    gen_mode="other"

    if filename is not None :
        reorient=False

    with open(fileseq) as f:
        sequence_config = json.load(f)

    sequence_config["TE"]=[2.39,3.45]
    sequence_config["TR"]=list(np.array(sequence_config["TE"])+10000)
    sequence_config["B1"]=[3.0,3.0]

    nrep=2
    rep=nrep-1
    TR_total = np.sum(sequence_config["TR"])

    Treco = TR_total-np.sum(sequence_config["TR"])
    Treco=10000
    ##other options
    sequence_config["T_recovery"]=Treco
    sequence_config["nrep"]=nrep
    sequence_config["rep"]=rep

    seq=T1MRFSS_NoInv(**sequence_config)

    #seq=T1MRF(**sequence_config)

    gen_mode="other"

    import pickle
    with open(filemap,"rb") as file:
        all_maps=pickle.load(file)

    #print(all_maps)
    mask=all_maps[0][1]
    map_rebuilt=all_maps[0][0]
    map_rebuilt["attB1"]=np.ones_like(map_rebuilt["attB1"])
    #map_rebuilt["attB1"]=1/map_rebuilt["attB1"]
    norm=all_maps[0][4]
    phase=all_maps[0][3]


    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in map_rebuilt.keys()]
    map_for_sim = dict(zip(list(map_rebuilt.keys()), values_simu))

                # predict spokes
    m = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True,gen_mode=gen_mode)

    m.buildParamMap()
    m.build_ref_images(seq,norm=norm,phase=phase)

    if reorient:
        volume_ip=np.flip(np.moveaxis(np.abs(m.images_series[0]),0,-1),axis=(1,2))
        volume_oop=np.flip(np.moveaxis(np.abs(m.images_series[1]),0,-1),axis=(1,2))
    else:
        volume_ip=np.abs(m.images_series[0])
        volume_oop=np.abs(m.images_series[1])
    
    
    split=str.split(filemap,"CF_iterative_2Dplus1_MRF_map.pkl")

    if len(split)==1:
        split=str.split(filemap,"MRF_map.pkl")
    file_ip=split[0]+"ip.mha"
    file_oop=split[0]+"oop.mha"

    # split=str.split(filemap,"/")
    # folder_path="/".join(split[:-1])
    # file_ip=folder_path+"/vol_ip.mha"
    # file_oop=folder_path+"/vol_oop.mha"
    # print(file_ip)

    #spacing=[1, 1,5]
    if filename is None:
        io.write(file_ip, volume_ip, tags={"spacing": spacing})
        io.write(file_oop, volume_oop, tags={"spacing": spacing})

    else:
        print("Getting geometry from {}".format(filename))
        geom,is3D=get_volume_geometry(filename)
        if is3D:
            volume_ip=np.flip(np.moveaxis(volume_ip,0,-1),axis=(1,2))
            volume_oop=np.flip(np.moveaxis(volume_oop,0,-1),axis=(1,2))
        else:
            volume_ip=np.moveaxis(volume_ip,0,2)[:,::-1]
            volume_oop=np.moveaxis(volume_oop,0,2)[:,::-1]
        
        
        volume_ip = io.Volume(volume_ip, **geom)
        volume_oop = io.Volume(volume_oop, **geom)
        io.write(file_ip,volume_ip)
        io.write(file_oop,volume_oop)
    


    return




@machine
@set_parameter("fileseg", str, default=None, description="Segmentation (.nii or .nii.gz)")
def generate_mask_roi_from_segmentation(fileseg):

    file_maskROI=str.split(fileseg,".nii")[0]+"_maskROI.npy"
    img = nib.load(fileseg)
    data = np.array(img.dataobj)
    maskROI=np.moveaxis(np.flip(data,axis=(1,2)),-1,0)



    np.save(file_maskROI,maskROI)

    return


@machine
@set_parameter("filemap", str, default=None, description="Maps (.pkl)")
@set_parameter("fileroi", str, default=None, description="ROIs (.npy)")
@set_parameter("adj_wT1", bool, default=True, description="Filter water T1 value for FF")
@set_parameter("fat_threshold", float, default=0.7, description="FF threshold for water T1 filtering")
def getROIresults(filemap,fileroi,adj_wT1,fat_threshold):

    file_results_ROI=str.split(filemap,".pkl")[0]+"_ROI_results.csv"

    import pickle
    with open(filemap,"rb") as file:
        all_maps=pickle.load(file)
    mask=all_maps[0][1]
    map_=all_maps[0][0]
    maskROI=np.load(fileroi)

    results=get_ROI_values(map_,mask,maskROI,adj_wT1=adj_wT1, fat_threshold=0.7,kept_keys=None,min_ROI_count=15,return_std=True)

    print(results)
    results.to_csv(file_results_ROI)

    return

def task(ts):
        #global fs_hat
    global max_slices
    global f_list
    global npoint
    global data_chopt
    npoint_list = np.expand_dims(np.arange(npoint), axis=(0, 1))
    fun_correl_matrix = -np.abs(np.sum(np.expand_dims(data_chopt[ts, :max_slices].conj(), axis=0) * np.exp(
            2 * 1j * np.pi * f_list * npoint_list / npoint), axis=-1))

        # cost_padded=np.array([cost[0]]+cost+[cost[-1]])
        # cost=np.maximum(cost_padded[1:-1]-0.5*(cost_padded[:-2]+cost_padded[2:]),0)

        # x = minimize(fun_correl, x0=(f_min+f_max)/2, bounds=[(f_min, f_max)], tol=1e-8)

        # f_opt=x.x[0]
    np.save("./fshat/fshat_{}.npy".format(ts),f_list[np.argmin(fun_correl_matrix, axis=0)].squeeze())

def task_Ashat(ts):
    global nb_channels
    global fs_hat
    global data
    global max_slices
    global npoint
    As_hat_ts = np.zeros((nb_channels, max_slices))
    kdata_with_pt_corrected_ts=np.zeros((nb_channels,max_slices,npoint),dtype=data.dtype)
    for ch in tqdm(range(nb_channels)):
        for sl in range(max_slices):
            f_opt = fs_hat[ts, sl]
                # f_opt=fs[ts,sl]
            scalar_product = np.sum(
                    data[ch, ts, sl].conj() * np.exp(2 * 1j * np.pi * f_opt * np.arange(npoint) / npoint))
            A_opt = np.abs(scalar_product)
            phase = -np.angle(scalar_product)
                # A_opt=As[ts,sl]*npoint
            kdata_with_pt_corrected_ts[ch, sl] = data[ch, ts, sl] - A_opt / npoint * np.exp(
                    2 * 1j * np.pi * f_opt * np.arange(npoint) / npoint) * np.exp(1j * phase)
            As_hat_ts[ch,sl]=A_opt
    np.save("./fshat/Ashat_{}.npy".format(ts), As_hat_ts)
    np.save("./fshat/kdata_pt_corrected_{}.npy".format(ts), kdata_with_pt_corrected_ts)


def task_Ashat_filtered(ch):
    global nb_allspokes
    global As_hat
    global max_slices
    As_hat_filtered_ch = np.zeros((nb_allspokes, max_slices))
    As_hat_normalized_ch = np.zeros((nb_allspokes, max_slices))
    for sl in tqdm(range(max_slices)):
        signal = As_hat[ch, :, sl]
        min_=np.min(signal)
        max_=np.max(signal)
        signal = (signal - min_) / (max_ - min_)
        As_hat_normalized_ch[:, sl] = signal
            # mean=np.mean(signal)
            # std=np.std(signal)
            # ind=np.argwhere(signal<(mean-std)).flatten()
            # signal[ind]=signal[ind-1]
        signal_filtered = savgol_filter(signal, 41, 3)
        signal_filtered = lowess(signal_filtered, np.arange(len(signal_filtered)), frac=0.1)[:, 1]
        #As_hat_filtered_ch[:, sl] = (max_-min_)*signal_filtered+min_
        As_hat_filtered_ch[:, sl] = signal_filtered
    np.save("./fshat/Ashat_filtered_ch{}.npy".format(ch), As_hat_filtered_ch)
    np.save("./fshat/As_hat_normalized_ch{}.npy".format(ch), As_hat_normalized_ch)

toolbox = Toolbox("script_recoInVivo_3D_machines", description="Reading Siemens 3D MRF data and performing image series reconstruction")
toolbox.add_program("build_kdata", build_kdata)
toolbox.add_program("build_coil_sensi", build_coil_sensi)
toolbox.add_program("build_volumes", build_volumes)
toolbox.add_program("build_mask", build_mask)
toolbox.add_program("build_data_for_grappa_simulation", build_data_for_grappa_simulation)
toolbox.add_program("calib_and_estimate_kdata_grappa", calib_and_estimate_kdata_grappa)
toolbox.add_program("build_data_autofocus", build_data_autofocus)
toolbox.add_program("aggregate_kdata", aggregate_kdata)
toolbox.add_program("build_maps", build_maps)
toolbox.add_program("generate_image_maps", generate_image_maps)
toolbox.add_program("generate_movement_gif", generate_movement_gif)
toolbox.add_program("generate_matchedvolumes_allgroups", generate_matchedvolumes_allgroups)
toolbox.add_program("build_data_nacq", build_data_nacq)
toolbox.add_program("calculate_displacement_weights", calculate_displacement_weights)
toolbox.add_program("coil_compression", coil_compression)
toolbox.add_program("coil_compression_bart", coil_compression_bart)
toolbox.add_program("build_volumes_allbins", build_volumes_allbins)
toolbox.add_program("build_volumes_singular_allbins_registered", build_volumes_singular_allbins_registered)
toolbox.add_program("build_volumes_singular_allbins", build_volumes_singular_allbins)
toolbox.add_program("build_volumes_singular", build_volumes_singular)
toolbox.add_program("build_volumes_iterative_allbins", build_volumes_iterative_allbins)
toolbox.add_program("build_mask_from_singular_volume", build_mask_from_singular_volume)
toolbox.add_program("build_mask_full_from_mask", build_mask_full_from_mask)
toolbox.add_program("select_slices_volume", select_slices_volume)
toolbox.add_program("generate_dictionaries", generate_dictionaries)
toolbox.add_program("extract_singular_volume_allbins", extract_singular_volume_allbins)
toolbox.add_program("extract_allsingular_volumes_bin", extract_allsingular_volumes_bin)
toolbox.add_program("build_kdata_pilot_tone", build_kdata_pilot_tone)
toolbox.add_program("build_volumes_iterative_allbins_registered", build_volumes_iterative_allbins_registered)
toolbox.add_program("build_volumes_iterative_allbins_registered_allindex", build_volumes_iterative_allbins_registered_allindex)
toolbox.add_program("getTR", getTR)
toolbox.add_program("getGeometry", getGeometry)
toolbox.add_program("convertArrayToImage", convertArrayToImage)
toolbox.add_program("plot_deformation", plot_deformation)
toolbox.add_program("build_navigator_images", build_navigator_images)
toolbox.add_program("generate_random_weights", generate_random_weights)
toolbox.add_program("build_volumes_singular_allbins_3D", build_volumes_singular_allbins_3D)
toolbox.add_program("build_volumes_iterative", build_volumes_iterative)
toolbox.add_program("generate_dixon_volumes_for_segmentation", generate_dixon_volumes_for_segmentation)
toolbox.add_program("generate_mask_roi_from_segmentation", generate_mask_roi_from_segmentation)
toolbox.add_program("getROIresults", getROIresults)

if __name__ == "__main__":
    toolbox.cli()





# python script_recoInVivo_3D_machines.py build_maps --filename-volume data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_volumes_singular_allbins_registered.npy --filename-mask data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_mask.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --dictfile mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict --dictfile-light mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict --optimizer-config opt_config_iterative_singular.json --file-deformation data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF.dat
# python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4.npy --file-model data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_vxm_model_weights.h5
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins_registered --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-pca data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_virtualcoils_16.pkl --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --file-deformation-map data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10 --n-comp 16