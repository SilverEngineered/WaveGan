import wave,struct
import numpy as np
import os
from matplotlib import pyplot as plt
src="data"
second_data=[]
first_pass=True
file_count=0
dir_path = os.path.dirname(os.path.realpath(__file__))
for each in os.listdir(src):
    print(os.path.join(src,each))
    waveFile=wave.open(os.path.join(src,each))
    length=waveFile.getnframes()
    per_file_data_in_seconds=[]
    alldata=[]
    for i in range(0,length):
        waveData=waveFile.readframes(1)
        binary_data=struct.unpack("<h",waveData)
        alldata.append(binary_data)
        if(i%16384==0):
            per_file_data_in_seconds.append(alldata)
            alldata=[]
    per_file_data_in_seconds=per_file_data_in_seconds[1:]
    waveFile.close()
    if first_pass:
        second_data=np.array(per_file_data_in_seconds)
        first_pass=False
    elif len(per_file_data_in_seconds)>0:
        second_data=np.concatenate((np.array(second_data),np.array(per_file_data_in_seconds)),axis=0)
alldata=np.array(alldata)
second_data=np.array(second_data)
print("Done Reading!")
for j in range(len(second_data)):
    x=[]
    #print(str(j) + " Out of: " + str(len(second_data)))
    signal=second_data[j]
    for i in range(len(signal)):
        x.append(wave.struct.pack('h',signal[i][0])) # transform to binary
    file_path=os.path.join("formatteddata",str(file_count) + ".wav")
    file=wave.open(file_path, 'wb')
    file.setparams((1, 2, 16384, 44100, 'NONE', 'noncompressed'))
    x=np.array(x)
    file.writeframes(x)
    file.close()
    file_count+=1
