CXX = g++
NVCC = nvcc

CXXFLAGS += -O3 -Wall -std=c++11
NVFLAGS += -std=c++11 -O3 -arch=sm_35 -Xcompiler -Wall

INCLUDES = -I/usr/include/nvidia-396/
LDFLAGS = -lOpenCL -lcuda -lcudart -lcublas

CPPSOURCES = project.cpp myBLAS.cpp
GPUSOURCES = cuBLAS.cu myBLAS.cu

OBJS = $(CPPSOURCES:%.cpp=%.cpp.o)
OBJS += $(GPUSOURCES:%.cu=%.cu.o)
BIN = gemm

all: build run

build: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $(LDFLAGS) -o $(BIN)

%.cpp.o: %.cpp *.h
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

%.cu.o: %.cu *.h
	$(NVCC) -c $(NVFLAGS) $(INCLUDES) $< -o $@
        
run:
	./$(BIN)
        
clean:
	rm -f $(BIN)
	rm -f *.o
