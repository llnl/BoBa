# Profiling cuda runs using nvidia tools

-  First, grab an allocation. You might want a longer allocation (try 5x longer) than normal since profiling has a significant overhead.
```
lalloc 1 -W 200 --atsdisable
```
- Load the nsight compute module 
```
ml nsight-compute
```
- Then, try to build and run the study of interest
```
make clean && make test_boba_scaling BOBA_CUDA=1 BOBA_PROFILING=1
```
- Now run the produced executable with extra arguments
```
lrun -1 ncu -f -o boba_gpu_perf --set full --target-processes all ./*.out
```
- Open a second terminal on a CPU cluster (such as `dane`). Load the nsight module there
```
ml nsight-compute
```
- Navigate to the working directory where you performed the above runs.
- Open the file `boba_gpu_perf.ncu-rep` with the program `ncu-ui`
```
ncu-ui boba_gpu_perf.log
```
