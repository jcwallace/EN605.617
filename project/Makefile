CXX = g++
NVCC = nvcc

CXXFLAGS += -O3 -Wall -std=c++11
NVFLAGS += -03 -arch=sm_35 -Xcompiler -Wall

INCLUDES = -I/usr/include/nvidia-396/cuda/
LDFLAGS = -lOpenCL -lcuda -lcudart -lcublas

CPPSOURCES = project.cpp
GPUSOURCES = cublas.cu

OBJS = $(CPPSOURCES:%.cpp=%.cpp.o)
OBJS += $(GPUSOURCES:%.cu=%.cu.o)
BIN = gemm

all: build run

build: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $(LDFLAGS) -o $(BIN)

%.cpp.o: %.cpp *.h
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

%.o: %.cu *.h
	$(NVCC) -c $(NVFLAGS) $(INCLUDES) $< -o $@
        
run:
	./$(BIN)
        
clean:
	rm -f $(BIN)
	rm -f *.o
