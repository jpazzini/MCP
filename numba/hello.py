from numba import cuda

# check if any cuda gpu is detected
print(cuda.gpus)

# select the device_id to assign it to numba
cuda.select_device(0)

print('all done')
