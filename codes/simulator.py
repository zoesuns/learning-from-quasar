import os
import h5py
import numpy as np
from astropy.convolution import convolve,Gaussian1DKernel


data_dir="../data/"

# load forest data
forest_hdf5=h5py.File(data_dir+"PZforest_LLS_DLA_free.hdf5",'r')
wv_flux_forest=forest_hdf5["wavelength"][:]
forest_dataset=forest_hdf5["forest"][:]
nForest=len(forest_dataset)

# load wing profiles
hfile=h5py.File(data_dir+"wing_profiles.h5",'r')
wv_wing=hfile["wavelength"][:]
xHIv_arr=[]
for key in hfile.keys():
    if 'xHIv=' in key:
        xHIv_arr.append(float(key.split('xHIv=')[-1]))
xHIv_arr=np.array(xHIv_arr)
nWing=10000

# split forest to test the impact of LSS
forest_train=np.load(data_dir+"forest_dataset_train.npy")
len_forest_train=len(forest_train)
forest_rest=np.load(data_dir+"forest_dataset_rest.npy")
len_forest_rest=len(forest_rest)


class Simulator:
    def __init__(self,param,dataset="train",SNR=10):
        '''
        input param=(xHIv,wingPos) #pytorch\n",
        return flux at wv=np.linspace(-4000,8000,1201) [km/s]\n",
        '''
        self.param = np.array(param)
        self.xHIv = float(param[0])
        self.wingPos = param[1]
        self.raw_wv = np.linspace(-4000,8000,1201)
        self.smsize = None
        self.binsize = None
        self.SNR = SNR
    
        if dataset=="train":
            self.forest_dataset_size=len_forest_train
            randi=np.random.randint(self.forest_dataset_size),
            self.forest_component=forest_train[randi]
        if dataset=="rest":
            self.forest_dataset_size=len_forest_rest
            randi=np.random.randint(self.forest_dataset_size),
            self.forest_component=forest_rest[randi]

        i=np.argmin(np.abs(xHIv_arr-self.xHIv))
        key="xHIv={:5.3f}".format(xHIv_arr[i])
        randi=np.random.randint(nWing)
        self.noise=np.random.normal(scale=1/SNR,size=len(self.raw_wv))
        
        if randi>=len(hfile[key]):
            self.wing_component = np.ones_like(self.forest_component) # no damping wing
            self.clean_spec = self.forest_component

        else:
            wing=hfile[key+"/wing_"+str(randi)][:]
            ipos=int((self.wingPos+4000)/10) #move it to the starting point
            self.wing_component = np.zeros_like(self.forest_component)
            self.wing_component[:ipos] = wing[-ipos:]
            self.clean_spec = self.forest_component*self.wing_component

        self.noisy_spec = self.clean_spec+self.noise

    def gaussian_smoothing(self,smsize=30):
        self.smsize = smsize
        gauss = Gaussian1DKernel(stddev=smsize/10)
        final=convolve(self.clean_spec, gauss, boundary='extend')
        outwv=convolve(wv_flux_forest, gauss, boundary='extend')
        return self.raw_wv[::int(smsize/10)],final[::int(smsize/10)]
    def binning(self,binsize=500):
        self.binsize=binsize
        npixbin=int(binsize/10)
        _reshaped = self.clean_spec[1:].reshape(-1, npixbin)
        binned_spec = np.mean(_reshaped, axis=1)
        tmp=self.raw_wv[::npixbin]
        return (tmp[1:]+tmp[:-1])/2, binned_spec
