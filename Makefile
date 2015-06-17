#ERLANG/OPENCL PATH - set these
#ERL_INCLUDE = ...
#OPENCL_LIB = ...
#OPENCL_INCLUDE = ...

#titanic
#ERL_INCLUDE = /usr/lib/erlang/erts-5.10.4/include/
#OPENCL_LIB = /opt/AMDAPPSDK-2.9-1/lib/x86_64
#OPENCL_INCLUDE = /opt/AMDAPPSDK-2.9-1/include/

#UBUNTU VM
#ERL_INCLUDE = /usr/local/lib/erlang/erts-5.10.4/include/
#OPENCL_LIB = /opt/AMDAPP/lib/x86_64
#OPENCL_INCLUDE = /opt/AMDAPP/include/

#pianosa
#ERL_INCLUDE =  /home/albanese/erlang/lib/erlang/erts-6.2/include/
#OPENCL_LIB = /usr/lib64
#OPENCL_INCLUDE = /opt/intel/opencl-1.2-3.1.1.11385/include/CL/

OPENCLFLAGS = -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -I$(OPENCL_INCLUDE) -L$(OPENCL_LIB) -lOpenCL
LIBFLAGS = -fpic -shared -std=c++11
NVIDIA = -DNVIDIA

C_SRC= c_src


.PHONY: all clean erl_clean so_clean


all:	erl NVIDIA

clean:	erl_clean so_clean


#----------------ERLANG targets---------------------
erl:
	erlc -o ebin src/*.erl

erl_clean:
	cd ebin && rm *.beam


#---------------- SKEL OCL shared library targets----------------

SO_NAME = skel_ocl.so

SRCS= $(C_SRC)/OCL2.cpp $(C_SRC)/skel_ocl.cpp $(C_SRC)/myTime.c

DEP_SRCS = $(SRCS) $(C_SRC)/errors.cpp

#FORCE NVIDIA PLATFORM (GPU)
NVIDIA:	$(DEP_SRCS)
	g++ $(NVIDIA) -I$(ERL_INCLUDE) $(LIBFLAGS) $(SRCS) -o $(SO_NAME) $(OPENCLFLAGS)

#skeleton total running time on stderr (start-end time)
NVIDIA_TIME_FUN: $(DEP_SRCS)
	g++ -DTIME_FUN $(NVIDIA) -I$(ERL_INCLUDE) $(LIBFLAGS) $(SRCS) -o $(SO_NAME) $(OPENCLFLAGS)

#skeleton running time on stderr (detailed)
NVIDIA_TIME:	$(DEP_SRCS)
	g++ -DTIME -I $(C_SRC) $(NVIDIA) -I$(ERL_INCLUDE) $(LIBFLAGS) $(SRCS) -o $(SO_NAME)$(OPENCLFLAGS)

#DEBUG prints on stderr
NVIDIA_DEBUG:	$(DEP_SRCS)
	g++ -DDEBUG -I $(C_SRC) $(NVIDIA) -I$(ERL_INCLUDE) $(LIBFLAGS) $(SRCS) -o $(SO_NAME) $(OPENCLFLAGS)

#Use the first platform available (GPU preferred)
DEFAULT:	$(DEP_SRCS)
	g++ -I$(ERL_INCLUDE) $(LIBFLAGS) $(SRCS) -o $(SO_NAME) $(OPENCLFLAGS)

so_clean:
	rm skel_ocl.so

