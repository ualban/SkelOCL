# make skel_ocl.so available to the loader
export LD_LIBRARY_PATH=$SKEL_OCL_DIR:$LD_LIBRARY_PATH

# make skel_ocl available to erlang
export ERL_LIBS=$SKEL_OCL_DIR