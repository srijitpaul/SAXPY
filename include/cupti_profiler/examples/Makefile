all: saxpy

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include
CXXARGS = -arch=sm_60 -std=c++11  
CXXARGS += -Xcompiler -DNDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti

saxpy: saxpy.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) saxpy.cu -o saxpy
clean:
	rm -f *.o saxpy
