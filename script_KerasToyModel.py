
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from utils_mrf import *
import json
from finufft import nufft1d1,nufft1d2
from scipy import signal,interpolate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
from movements import *
from dictoptimizers import *
from toy_model_keras import *
import tensorflow as tf
## Random map simulation

useGPU=False

dictfile = "mrf175.dict"
#dictfile = "mrf175_CS.dict"
dictfile = "mrf175_SimReco2.dict"

with open("mrf_sequence.json") as f:
    sequence_config = json.load(f)


seq = T1MRF(**sequence_config)

window = 8 #corresponds to nspoke by image
size=(256,256)

file_matlab_paramMap = "./data/Phantom1/paramMap.mat"

###### Building Map
#m = MapFromFile("TestPhantomV1",image_size=size,file=file_matlab_paramMap,rounding=True)

with open("mrf_dictconf_SimReco2.json") as f:
    dict_config = json.load(f)

dict_config["ff"]=np.arange(0.,1.05,0.05)

window = 8 #corresponds to nspoke by image
region_size=16 #size of the regions with uniform values for params in pixel number (square regions)
size=(256,256)

predictions=[]

file_matlab_paramMap = "./data/paramMap.mat"

###### Building Map
m = RandomMap("TestRandom",dict_config,image_size=size,region_size=region_size,mask_reduction_factor=1/4)

m.buildParamMap()

#m.plotParamMap(save=True)

##### Simulating Ref Images
m.build_ref_images(seq)

ntimesteps=175
nspoke=8
npoint = 2*m.images_series.shape[1]

radial_traj=Radial(total_nspokes=ntimesteps*nspoke,npoint=npoint)
kdata = m.generate_kdata(radial_traj,useGPU=useGPU)

volumes = simulate_radial_undersampled_images(kdata,radial_traj,m.image_size,density_adj=True,useGPU=useGPU)
mask = build_mask_single_image(kdata,radial_traj,m.image_size,useGPU=useGPU)#Not great - lets make both simulate_radial_.. and build_mask_single.. have kdata as input and call generate_kdata upstream
plt.imshow(mask)
mask=m.mask
#optimizer = SimpleDictSearch(mask=mask,niter=10,seq=seq,trajectory=radial_traj,split=500,pca=True,threshold_pca=15,log=False,useAdjPred=False)


callbacks_list = []
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        self.x = x_test
        self.y = y_test

    def on_train_batch_end(self, epoch, logs=None):
        print('Evaluating Model...')
        x = self.x.reshape(1,-1)
        print(x.shape)
        pred = self.model.predict(x)
        Y_pred=optimizer.paramDict["output_scaler"].inverse_transform(pred.reshape(1,-1))

        global predictions

        predictions.append(Y_pred[0])

        #print("=====Parameters comparison===========")
        # print("Index : {}".format(i))
        # print("wT1 : {} vs {}".format(Y_pred[0][0], self.y[0]))
        # print("fT1 : {} vs {}".format(Y_pred[0][1], self.y[1]))
        # print("attB1 : {} vs {}".format(Y_pred[0][2], self.y[2]))
        # print("df : {} vs {}".format(Y_pred[0][3], self.y[3]))
        # print("ff : {} vs {}".format(Y_pred[0][4], self.y[4]))

        #print('Model Evaluation: ', self.model.evaluate(self.x_test))


FF_list = list(np.arange(0., 1.05, 0.05))
keys, signal = read_mrf_dict(dictfile, FF_list)
Y_TF = np.array(keys)
real_signal = signal.real
imag_signal = signal.imag
X_TF = np.concatenate((real_signal, imag_signal), axis=1)

i=np.random.choice(X_TF.shape[0])
perf_callback = PerformancePlotCallback(X_TF[i],Y_TF[i])


callbacks_list.append(tensorboard_callback)
callbacks_list.append(stop_early)
callbacks_list.append(perf_callback)

model = build_and_compile_model_simple
fitting_opt = {
"batch_size":256*4,"shuffle":True,
    "validation_split":0.2,
    "verbose":1, "epochs":300,"callbacks":callbacks_list

}



optimizer = ToyNN(model,fitting_opt,mask=mask)
optimizer.fit_and_set(dictfile)

predictions=np.array(predictions)

plt.figure()
j=4
plt.plot(predictions[:,j])
plt.axhline(y=Y_TF[i,j],c="r",linestyle="dashed")

Y_pred=optimizer.paramDict["output_scaler"].inverse_transform(optimizer.paramDict["model"].predict(X_TF[i].reshape(1,-1)))


print("=====Parameters comparison===========")
print("Index : {}".format(i))
print("wT1 : {} vs {}".format(Y_pred[0][0],Y_TF[i,0]))
print("fT1 : {} vs {}".format(Y_pred[0][1],Y_TF[i,1]))
print("attB1 : {} vs {}".format(Y_pred[0][2],Y_TF[i,2]))
print("df : {} vs {}".format(Y_pred[0][3],Y_TF[i,3]))
print("ff : {} vs {}".format(Y_pred[0][4],Y_TF[i,4]))



all_maps_adj=optimizer.search_patterns(dictfile,volumes)

plt.close("all")

maskROI=buildROImask(m.paramMap)
regression_paramMaps_ROI(m.paramMap, all_maps_adj[0][0], m.mask > 0, all_maps_adj[0][1] > 0,maskROI=maskROI,
                             title="ROI Orig vs Toy Keras", proj_on_mask1=True, adj_wT1=True, fat_threshold=0.7)

compare_paramMaps(m.paramMap, all_maps_adj[0][0], m.mask > 0, all_maps_adj[0][1] > 0,adj_wT1=True,fat_threshold=0.7,proj_on_mask1=True)