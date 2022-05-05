NVCC = nvcc

NVFLAGS += -03 -arch=sm_35 -Xcompiler -Wall

LDFLAGS += -lOpenCL -lcuda -lcudart

SOURCES = project.cu

OBJS = $(SOURCES:%.cpp=%.o)

BIN = gemm

all: build run

build: $(BIN)
	$(NVCC) $(SOURCES) -std=c++11 $(LDFLAGS) -o $(BIN)
        
run:
	./$(BIN)
        
clean:
	rm -f $(BIN)
	rm -f *.o
