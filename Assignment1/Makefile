CC=g++
NVCC=		nvcc
LINK=		nvcc
#NVCC=		/usr/local/cuda-11.7/bin/nvcc
#LINK=		/usr/local/cuda-11.7/bin/nvcc
#NVCC=		/usr/local/cuda-13.1/bin/nvcc
#LINK=		/usr/local/cuda-13.1/bin/nvcc

#CFLAGS= -g -pedantic -W -Wall -L/usr/lib
#CFLAGS= -O2
CFLAGS= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

PROBLEM_SIZE := 10000
RECTANGULAR_PROBLEM_SIZE_N := 50
RECTANGULAR_PROBLEM_SIZE_M := 20000

#CUDA_ARCH_FLAGS			= -gencode arch=compute_35,code=sm_35	# GTX 680 is 3.5, a Kepler
CUDA_ARCH_FLAGS		= -gencode=arch=compute_60,code=sm_60	# GTX 1080 is 6.0, a Pascal
#CUDA_ARCH_FLAGS		= -gencode arch=compute_70,code=sm_70	# V100 is 7.0, a Volta
#CUDA_ARCH_FLAGS		= -gencode arch=compute_75,code=sm_75	# RTX 2080 Super is 7.5, a Turing
#CUDA_ARCH_FLAGS		= -gencode arch=compute_86,code=sm_86	# RTX 3090 is 8.6, a Ampere
#CUDA_ARCH_FLAGS		= -gencode arch=compute_90,code=sm_90	# H100s and H200s are 9.0, a Hopper

#NVCCFLAGS	= -g -G -DWITH_MY_DEBUG
NVCCFLAGS_REGS			= --use_fast_math $(CUDA_ARCH_FLAGS) --extra-device-vectorization -DWITH_REGISTERS
NVCCFLAGS_NOREGS		= --use_fast_math $(CUDA_ARCH_FLAGS) --extra-device-vectorization
#NVCCFLAGS	= --use_fast_math $(CUDA_ARCH_FLAGS) --extra-device-vectorization
#NVCCFLAGS	= --use_fast_math --resource-usage $(CUDA_ARCH_FLAGS) --extra-device-vectorization

INCPATH       = -I.

#SOURCES		= matrixAddUp.cu

OBJECTS_REGS		=	matrixAddUpRegs.o \
		  				main.o

OBJECTS_NOREGS		=	matrixAddUpNoRegs.o \
		  				main.o

TARGET_REGS 		= matrixAddUpRegs.out
TARGET_NOREGS 		= matrixAddUpNoRegs.out

EXEC_REGS=matrixNormsRegs.out
EXEC_NOREGS=matrixNormsNoRegs.out

all: mainRegs mainNoRegs

#main: mainRegs mainNoRegs
#	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) -lefence
#	$(NVCC) $(OBJECTS) -o $(TARGET) $(NVCCFLAGS) -I$(INCPATH) 

mainRegs: $(OBJECTS_REGS) Makefile
	$(NVCC) $(OBJECTS_REGS) -o $(TARGET_REGS) $(NVCCFLAGS_REGS) -I$(INCPATH) 

mainNoRegs: $(OBJECTS_NOREGS) Makefile
	$(NVCC) $(OBJECTS_NOREGS) -o $(TARGET_NOREGS) $(NVCCFLAGS_NOREGS) -I$(INCPATH) 

main.o: main.cpp Makefile
	$(CC) main.cpp $(CFLAGS) -c $(INCPATH) $<

#%.o: %.cpp Makefile
#	$(CC) $(CFLAGS) -c $(INCPATH) $<

matrixAddUpRegs.o: matrixAddUp.cu Makefile
	$(NVCC) matrixAddUp.cu -o matrixAddUpRegs.o -c $(NVCCFLAGS_REGS) -I$(INCPATH)

matrixAddUpNoRegs.o: matrixAddUp.cu Makefile
	$(NVCC) matrixAddUp.cu -o matrixAddUpNoRegs.o -c $(NVCCFLAGS_NOREGS) -I$(INCPATH)

install:

test: all
	@echo "   ";
	@echo "==================================================================================================================================";
	@echo "========================== Testing Without Registers, size "$(PROBLEM_SIZE)"x"$(PROBLEM_SIZE)" ==========================";
	@echo "==================================================================================================================================";
	for blockSize in 32 64 128 256 512 1024 ; do \
		./$(TARGET_NOREGS) -n $(PROBLEM_SIZE) -m $(PROBLEM_SIZE) -t -x $$blockSize -X $$blockSize -y $$blockSize -Y $$blockSize; \
	done
	@echo "   "
	@echo "=================================================================================================================================="
	@echo "========================== Testing With Registers, size "$(PROBLEM_SIZE)"x"$(PROBLEM_SIZE)" ==========================";
	@echo "=================================================================================================================================="
	for blockSize in 32 64 128 256 512 1024 ; do \
		./$(TARGET_REGS) -n $(PROBLEM_SIZE) -m $(PROBLEM_SIZE) -t -x $$blockSize -X $$blockSize -y $$blockSize -Y $$blockSize; \
	done

test-efence: all
	@echo "   ";
	@echo "==================================================================================================================================";
	@echo "========================== Testing Non Square cases With Electric Fence and Without Registers, sizes "$(RECTANGULAR_PROBLEM_SIZE_N)"x"$(RECTANGULAR_PROBLEM_SIZE_N)" and transposed ==========================";
	@echo "==================================================================================================================================";
	for blockSize in 32 64 128 256 512 1024 ; do \
		LD_PRELOAD=libefence.so ./$(TARGET_NOREGS) -n $(RECTANGULAR_PROBLEM_SIZE_N) -m $(RECTANGULAR_PROBLEM_SIZE_M) -t -x $$blockSize -X $$blockSize -y $$blockSize -Y $$blockSize; \
		LD_PRELOAD=libefence.so ./$(TARGET_NOREGS) -n $(RECTANGULAR_PROBLEM_SIZE_M) -m $(RECTANGULAR_PROBLEM_SIZE_N) -t -x $$blockSize -X $$blockSize -y $$blockSize -Y $$blockSize; \
	done
	@echo "   "
	@echo "=================================================================================================================================="
	@echo "========================== Testing Non Square cases With Electric Fence and With Registers, sizes "$(RECTANGULAR_PROBLEM_SIZE_N)"x"$(RECTANGULAR_PROBLEM_SIZE_N)" and transposed ==========================";
	@echo "=================================================================================================================================="
	for blockSize in 32 64 128 256 512 1024 ; do \
		LD_PRELOAD=libefence.so ./$(TARGET_REGS) -n $(RECTANGULAR_PROBLEM_SIZE_N) -m $(RECTANGULAR_PROBLEM_SIZE_M) -t -x $$blockSize -X $$blockSize -y $$blockSize -Y $$blockSize; \
		LD_PRELOAD=libefence.so ./$(TARGET_REGS) -n $(RECTANGULAR_PROBLEM_SIZE_M) -m $(RECTANGULAR_PROBLEM_SIZE_N) -t -x $$blockSize -X $$blockSize -y $$blockSize -Y $$blockSize; \
	done



clean:
	rm -f *.o ${TARGET_REGS} ${TARGET_NOREGS}
