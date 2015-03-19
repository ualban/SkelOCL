# SkelOCL

Welcome to Skel OCL! A prototype skeleton library for Erlang that exploits GP-GPUs via OpenCL.
Skel OCL is implemented as an Erlang NIF library.

The only supported platform is Linux.

Erlang and OpenCL are required.

-----------HOW TO BUILD SKEL OCL-----------

1.Set the Makefile:
- Set the ERL_INCLUDE variable to the directory containing Erlang's include files
- Set the OPENCL_LIB variable to the OpenCL library directory
- Set the OPENCL_INCLUDE variable to the directory containing CL.h

2.Once the variables are set up, just launch make.
The default target builds both the erlang part (into ebin/) and the NIF .so (skel_ocl.so).

3.Make sure Erlang finds the skel_ocl/ebin directory.
To do that, one way is to set the ERL_LIBS env variable to the skel_ocl root directory (the one to which this readme belongs).
If the root directory is the current:
export ERL_LIBS=`pwd`

4.We need also to make skel_ocl.so available to the loader, since Erlang VM will need it at runtime.
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH


-----------USAGE EXAMPLE--------------

Refer to example/ for a simple usage example. The module usageExample shows how to use the mapLL skeleton applying a square kernel to the elements of a list of float() elements. The kernel is defined in a .cl file using OpenCL C.


Running the example from Erlang shell :

Erlang R16B01 (erts-5.10.2) [source] [64-bit] [smp:2:2] [async-threads:10] [kernel-poll:false]

Eshell V5.10.2  (abort with ^G)
1> skel_ocl:cl_init().
ok
2> c(usageExample).
{ok,usageExample}      
3> usageExample:exampleFun(1024).
[2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,22.0,24.0,
 26.0,28.0,30.0,32.0,34.0,36.0,38.0,40.0,42.0,44.0,46.0,48.0,
 50.0,52.0,54.0,56.0,58.0|...]

NOTE: Using the default make target, Skel OCL is built so to look for NVIDIA GPUs during initialization (skel_ocl:cl_init()).
In case no NVIDIA GPUs are found, initialization fails.
This can be avoided building Skel OCL with make DEFAULT. Using the DEFAULT target, the behavior is the following: look for the first OpenCL platform available and prefer GPU devices over CPU.
