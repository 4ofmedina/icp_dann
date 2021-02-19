import numpy as np
import glob, os
from scipy import signal
import peakdetect

## parameters for windowing:
stride = 600  # 25 in samples
win = 300  # in samples
norm = 1  # 1 for normalization of the windows on, 0 for off

## Acquistion times:
dt = 0.026  # acq. time  blood flow (BF)
dtI = 0.0025  # acq. time ICP for TBI (400Hz) (!! BESS has 200Hz !!)

def prepareData(path):
    BFWins = []
    meanICPS = []
    ICPwins = []
    patientsBF = []
    patientsICPS = []

    i = 0
    fname = []
    for f in sorted(glob.glob(path)):
        ## Load the data
        rCBF, ICP, timeB, _ = loadHoB(f)
        ## Windowing of the data with the parameters defined above     
        winsBF, ICPmean, winsICP, winsIndex, _ = windowing(win, stride, 1, rCBF, ICP, 1)
        print(winsBF.shape)
        print(winsICP.shape)
        print(ICPmean.shape)

        tmp = f.split('/')[-1].split('.')[0]
        fname.append(tmp)
        print(fname[-1])

        ## create folder results and save the subject specific data in a.npz file
        indexpath = "./results/"
        if not os.path.exists(indexpath):
            os.mkdir(indexpath)
        
        if path == "./data_TBI/T*.txt":
            indexpath = "./results/source"
        else:
            indexpath = "./results/target"
        
        if not os.path.exists(indexpath):
            os.mkdir(indexpath)
        
        np.savez(indexpath+"/" + fname[-1] + "_" + str(win) + "_" + str(stride) + "_" + str(norm) + ".npz", 
                 winsIndex = winsIndex, winsBF = winsBF, ICPmean = ICPmean, winsICP = winsICP, subjID = fname[-1])

        ## Check if loading the data worked:
        #subjectData = np.load(indexpath + fname[-1] + "_" + str(win) + "_" + str(stride) + "_" + str(norm) + ".npz")
        #print(sorted(subjectData.files))
            
        # Here you can stack/append the dat for all subjects directlz together to use them, if you wish
        #if i == 0:
        #    i += 1
        #    BFWins = winsBF
        #    meanICPS = ICPmean
        #    ICPwins = winsICP
        #else:
        #    BFWins = np.vstack((BFWins, winsBF))
        #    meanICPS = np.hstack((meanICPS, ICPmean))
        #    ICPwins = np.vstack((ICPwins, winsICP))

        #patientsBF.append(winsBF)
        #patientsICPS.append(ICPmean)


    return  patientsBF, patientsICPS

def loadHoB(fname):
    X = np.loadtxt(fname, skiprows=1)
    timeB = X[:,0] # time vector blood flow
    timeI = X[:,3] # time vector ICP (!! here is a rounding issue, time not exact see dtI below, therefore we will correct the time vector later !!)
    rCBF = X[:,2]*1e9 # raw blood flow data, 2 = BFI blood flow index, then multiply as well with 1e9, 3 = rCBF relative changes of BFI to a baseline BFI/<BFI>_baseline
    ICP = X[:,4] # raw ICP data
    marks = X[:,-1]
	
	# Make sure files which start with a time of exact 0 can be read as well
    if timeB[0,] == 0:
        timeB[0,] = 0.0001

    # Find start end end points for ICP and BF
    lengthB = np.where(timeB == 0)[0][0]
    begI = np.where(timeI >= 0)[0][0]
    endI = np.where(timeI >= timeB[lengthB-1])[0][0]
       
    # Use only relevant data (timeB>=0)
    ICP = ICP[begI:endI+5]
    timeI = timeI[begI:endI+5]
    rCBF = rCBF[:lengthB]
    timeB = timeB[:lengthB]
      
    Fs=1/dt # Sampling rate BF; Fs/2 = nyquist
    FsI = 1/dtI # Sampling rate ICP; Fs/2 = nyquist
    
    # Create new corrected time vector for the ICP (see above)
    timeInew = np.arange(0,len(ICP)*dtI,dtI)

    # Filter ICP with 7 Hz
    bd, ad, = signal.butter(7, 7/(FsI/2), btype="low")
    icpfilt = signal.filtfilt(bd,ad,ICP)
    # Downsample ICP data to blood flow data
    #ICPdec = np.interp(timeB,timeInew,icpfilt)
    # Instead we keep the sampling rate of the ICP and we adjust the stride and the window size for the sampling rate
    ICPdec = icpfilt

    # Filter blood flow with 7Hz low pass filter
    bd2, ad2, = signal.butter(7, 7/(Fs/2), btype="low")
    rCBF = signal.filtfilt(bd2, ad2, rCBF)

    return rCBF, ICPdec, timeB, marks


def windowing(size, stride, lookback, CBF, ICP, norm):
    windowsBF = []
    windowsICP = []
    meanICP = []
    windowsindex = []

    # Adjusted stride and window size for IC, e.g. stride * FsICP/FsBF (Fs sampling rate = 1/dt)
    strideicp = int(stride*dt/dtI)
    sizeicp = int(size*dt/dtI)

    i = 0
    reject = 0
    accept = 0
    bf = np.copy(CBF)
    icp = np.copy(ICP)
    
    selectVal = False
    # In the while we obtain the windows and calculate the mean ICP for the ICP windows
    while int((i+lookback)*stride+size) < len(bf)-size:
        
        win_lookf = []
        win_lookpre = []
        win_lookicp = []
        snr = 0

        for j in range(lookback):
            preData = bf[(i+j)*stride:(i+j)*stride+size]  # getting BF window
            
            # If norm = 1, we get normalized windows
            if norm == 1:
                fData = normalize(preData)
            else:
                fData = preData

            win_lookf.append(fData)
            win_lookicp.append(icp[(i+j)*strideicp:(i+j)*strideicp+sizeicp]) # getting ICP window
        if True:
            windowsBF.append(win_lookf)  # blood flow windows
            windowsICP.append(win_lookicp)  # ICP windows
            windowsindex.append(i+1)
            # Taking out non relevant values/ JF: do we need this?
            if np.mean(win_lookicp) > 40:
                meanICP.append(40)
            elif np.mean(win_lookicp) < 1:
                meanICP.append(1)
            else:
                meanICP.append(np.mean(win_lookicp)) # mean ICP value
            accept += 1
        else:
            reject += 1
        i += 1

    return np.array(windowsBF), np.array(meanICP), np.array(windowsICP), np.array(windowsindex), accept


def normalize(data):
    ## normalization method to have 0 mean and std of 1
    
    # get peaks of teh window
    maxt, mint = peakdetect.peakdet(data, .3)

    minmed = np.median(data[mint.astype(int)])
    maxmed = np.median(data[maxt.astype(int)])
    if (maxmed - minmed) == 0:
        maxmed = np.max(data)
        minmed = np.min(data)

    ret = (data - minmed)/(maxmed - minmed)
    return (ret - np.mean(ret))/np.std(ret)



if __name__ == "__main__":

    print('running')
    patientsBF, patientsICPS = prepareData("./data_TBI/T*.txt")
    patientsBF, patientsICPS = prepareData("./data_KIDS_TBI/k*.txt")
    print('done')

   
        

    

