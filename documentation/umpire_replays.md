
# How to profile umpire allocations

- When installing Umpire, enable tools.  This is likely done by default in BoBa, but double check this if you are having issues.
```
  -DUMPIRE_ENABLE_TOOLS=On \
```

- Build your Umpire-enabled example for CUDA
```
make clean && make -j 2 example_explicit BOBA_CUDA=1 BOBA_UMPIRE=1
```
or for HIP
```
make clean && make -j 2 example_explicit BOBA_HIP=1 BOBA_UMPIRE=1
```

- Set an environment variable to enable replays, `UMPIRE_REPLAY=ON`, and your CUDA example
```
UMPIRE_REPLAY=ON SCHEME=tensor_train lrun -1 ./example_explicit_cuda_ump.out -D 3 -n 300
```
or your HIP example:
```
UMPIRE_REPLAY=ON SCHEME=tensor_train ./example_explicit_hip_ump.out -D 3 -n 300
```

- This should produce a ``.replay`` file.  In Umpire, we need to find the replay executable and run it on the replay file. Note that if you used cuda/hip you should probably use the matching Umpire executable:
```
 ./install/umpire_cuda/bin/replay -d -i umpire.*.0.replay
 ```
 or
 ```
 ./install/umpire_hip/bin/replay -d -i umpire.*.0.replay
```
- This will generate a ultra file.
- Note: you may need to use the ``-r`` option when running replay. This will refresh or "recompile" the replay binary.
```
replay2133111.ult
```
Run `pydv` on the ultra file (this path may be system-dependent).
```
/usr/gapps/pydv/pdv replay2133111.ult
```
Now that you have pydv open, check out the [PyDV documentation](https://pydv.readthedocs.io/en/latest/getting_started.html) to plot the quantities of interest.  For example, you can plot the memory and memory roof.
```
curve 9 11 12
```
