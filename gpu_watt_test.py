from pynvml import *
from pynvml.smi import nvidia_smi

nvmlInit()

inst = nvidia_smi.getInstance()
result = inst.DeviceQuery("-q Power Readings")

print(result)

