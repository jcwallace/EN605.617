CXX = g++
NVCC = nvcc

CLBLAS_DIR = clBLAS

CXXFLAGS += -O3 -Wall -std=c++11
NVFLAGS += -std=c++11 -O3 -arch=sm_35 -Xcompiler -Wall

INCLUDES = -I/usr/include/nvidia-396/
INCLUDES += -I$(CLBLAS_DIR)/include
LDFLAGS = -L$(CLBLAS_DIR)/lib64
LDFLAGS += -lOpenCL -lcuda -lcudart -lcublas

CPPSOURCES = project.cpp clBLAS.cpp
GPUSOURCES = cuBLAS.cu myBLAS.cu

OBJS = $(CPPSOURCES:%.cpp=%.cpp.o)
OBJS += $(GPUSOURCES:%.cu=%.cu.o)
BIN = gemm

all: build run

build: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $(LDFLAGS) -o $(BIN)

%.cpp.o: %.cpp *.h
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $< -o $@

%.cu.o: %.cu *.h
	$(NVCC) -c $(NVFLAGS) $(INCLUDES) $< -o $@
        
run:
	./$(BIN)
        
clean:
	rm -f $(BIN)
	rm -f *.o
