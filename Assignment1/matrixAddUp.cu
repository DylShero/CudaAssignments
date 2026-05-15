///// Created by Jose Mauricio Refojo - 2018-01-23		Last changed: 2026-02-16
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------


#include <getopt.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>      // std::setprecision
#include <sys/time.h>
#include <time.h>
#include <vector>

using namespace std;

void cudaLastErrorCheck (const char *message) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		cout << "(Cuda error " << message << "): " << cudaGetErrorString( err) << ")" << endl;
		exit(EXIT_FAILURE);
	}
}



// TODO: Write a kernel that reduces a single precision vector in the global memory into a single value in the global memory performing the addition:
// TODO: in the registers if the compiler passes the flag -DWITH_REGISTERS or in the global memory if not
__global__ void reduceVectorSinglePrecision(const float* inputVector, float* finalValue, int size) {
    extern __shared__ float sdataFloat[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data into shared memory (use 0.0f if out of bounds)
    if (i < size) {
    	sdataFloat[tid] = inputVector[i];
	} else {
    	sdataFloat[tid] = 0.0f;
	}
    __syncthreads();

    //Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataFloat[tid] += sdataFloat[tid + s];
        }
        __syncthreads();
    }

    //Thread 0 of each block adds its sum to the final global value
    if (tid == 0) {
        atomicAdd(finalValue, sdataFloat[0]);
    }
}

// TODO: Write a kernel that adds together the absolute value of each element of each row of a single precision matrix into a single precision vector of size n performing the addition:
// TODO: in the registers if the compiler passes the flag -DWITH_REGISTERS or in the global memory if not
__global__ void addMatrixRowsSinglePrecision(const float* matrix, float* outVector, int rows, int columns) {
    extern __shared__ float sdataFloatRow[];
    
    int rowIdx = blockIdx.x; //Each block is assigned one row
    int tid = threadIdx.x;

    if (rowIdx < rows) {
        float sum = 0.0f;
        
        //Grid-stride loop to handle matrices with more columns than threads per block
        for (int colIdx = tid; colIdx < columns; colIdx += blockDim.x) {
            sum += fabsf(matrix[rowIdx * columns + colIdx]);
        }
        
        //Store thread's partial sum into shared memory
        sdataFloatRow[tid] = sum;
        __syncthreads();

        //Perform parallel reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdataFloatRow[tid] += sdataFloatRow[tid + s];
            }
            __syncthreads();
        }

        //Thread 0 writes the reduced row sum to the output vector
        if (tid == 0) {
            outVector[rowIdx] = sdataFloatRow[0];
        }
    }
}
// TODO: Write a kernel that adds together the absolute value of each element of each column of a single precision matrix into a single precision vector of size m performing the addition in the registers
// TODO: in the registers if the compiler passes the flag -DWITH_REGISTERS or in the global memory if not
__global__ void addMatrixColsSinglePrecision(const float* matrix, float* outVector, int rows, int columns) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (colIdx < columns) {
#ifdef WITH_REGISTERS
        float sum = 0.0f;
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            sum += fabsf(matrix[rowIdx * columns + colIdx]);
        }
        outVector[colIdx] = sum;
#else
        outVector[colIdx] = 0.0f; // Initialize to prevent garbage values
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            outVector[colIdx] += fabsf(matrix[rowIdx * columns + colIdx]);
        }
#endif
    }
}



// TODO: Write a kernel that reduces a double precision vector in the global memory into a double value in the global memory performing the addition:
// TODO: in the registers if the compiler passes the flag -DWITH_REGISTERS or in the global memory if not
__global__ void reduceVectorDoublePrecision(const double* inputVector, double* finalValue, int size) {
    extern __shared__ double sdataDouble[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data into shared memory (use 0.0 if out of bounds)
    sdataDouble[tid] = (i < size) ? inputVector[i] : 0.0;
    __syncthreads();

    //Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdataDouble[tid] += sdataDouble[tid + s];
        }
        __syncthreads();
    }

    //Thread 0 of each block adds its sum to the final global value
    if (tid == 0) {
        atomicAdd(finalValue, sdataDouble[0]);
    }
}

// TODO: Write a kernel that adds together the absolute value of each element of each row of a double precision matrix into a double precision vector of size n performing the addition:
// TODO: in the registers if the compiler passes the flag -DWITH_REGISTERS or in the global memory if not
__global__ void addMatrixRowsDoublePrecision(const double* matrix, double* outVector, int rows, int columns) {
    extern __shared__ double sdataDoubleRow[];
    
    int rowIdx = blockIdx.x; //Each block is assigned exactly one row
    int tid = threadIdx.x;

    if (rowIdx < rows) {
        double sum = 0.0;
        
        //Grid-stride loop to handle matrices with more columns than threads per block
        for (int colIdx = tid; colIdx < columns; colIdx += blockDim.x) {
            sum += fabs(matrix[rowIdx * columns + colIdx]);
        }
        
        //Store thread's partial sum into shared memory
        sdataDoubleRow[tid] = sum;
        __syncthreads();

        //Perform parallel reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdataDoubleRow[tid] += sdataDoubleRow[tid + s];
            }
            __syncthreads();
        }

        //Thread 0 writes the reduced row sum to the output vector
        if (tid == 0) {
            outVector[rowIdx] = sdataDoubleRow[0];
        }
    }
}

// TODO: Write a kernel that adds together the absolute value of each element of each column of a double precision matrix into a double precision vector of size m performing the addition in the registers
// TODO: in the registers if the compiler passes the flag -DWITH_REGISTERS or in the global memory if not
__global__ void addMatrixColsDoublePrecision(const double* matrix, double* outVector, int rows, int columns) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (colIdx < columns) {
#ifdef WITH_REGISTERS
        double sum = 0.0;
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            sum += fabs(matrix[rowIdx * columns + colIdx]);
        }
        outVector[colIdx] = sum;
#else
        outVector[colIdx] = 0.0;
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            outVector[colIdx] += fabs(matrix[rowIdx * columns + colIdx]);
        }
#endif
    }
}


extern int cudaMatrixAddUp (
	std::vector< float >  &matrixFloat1d,
	std::vector< double >  &matrixDouble1d,
	int rows,int columns,
	float &totalRowsFloat,			float &totalColumnsFloat,
	double &totalRowsDouble,		double &totalColumnsDouble,
	double &timeAddRowsFloatGpu,	double &timeAddColumnsFloatGpu,
	double &timeReduceRowsFloatGpu, double &timeReduceColumnsFloatGpu,
	double &timeAddRowsDoubleGpu,	double &timeAddColumnsDoubleGpu,
	double &timeReduceRowsDoubleGpu,double &timeReduceColumnsDoubleGpu,
	int &blockSizeSinglePrecisionRow,int &blockSizeSinglePrecisionColumn,int &blockSizeDoublePrecisionRow,int &blockSizeDoublePrecisionColumn,
	bool verbose,unsigned int printPrecision, int numberOfStreams) {

	cudaError_t err;
	//Allocate the variables in the global memory

	cudaHostRegister(matrixFloat1d.data(), matrixFloat1d.size() * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(matrixDouble1d.data(), matrixDouble1d.size() * sizeof(double), cudaHostRegisterDefault);

	//Create streams
	cudaStream_t* streamsFloat = new cudaStream_t[numberOfStreams];
    cudaStream_t* streamsDouble = new cudaStream_t[numberOfStreams];
    for (int i = 0; i < numberOfStreams; ++i) {
        cudaStreamCreate(&streamsFloat[i]);
        cudaStreamCreate(&streamsDouble[i]);
    }


	float *matrixFloat_gpu, *rowsFloat_gpu, *columnsFloat_gpu, *totalRowsFloat_gpu, *totalColumnsFloat_gpu;
    cudaMalloc(&matrixFloat_gpu, sizeof(float) * (rows * columns));
    cudaMalloc(&rowsFloat_gpu, sizeof(float) * rows);
    cudaMalloc(&columnsFloat_gpu, sizeof(float) * columns);
    cudaMalloc(&totalRowsFloat_gpu, sizeof(float));
    cudaMalloc(&totalColumnsFloat_gpu, sizeof(float));

    double *matrixDouble_gpu, *rowsDouble_gpu, *columnsDouble_gpu, *totalRowsDouble_gpu, *totalColumnsDouble_gpu;
    cudaMalloc(&matrixDouble_gpu, sizeof(double) * (rows * columns));
    cudaMalloc(&rowsDouble_gpu, sizeof(double) * rows);
    cudaMalloc(&columnsDouble_gpu, sizeof(double) * columns);
    cudaMalloc(&totalRowsDouble_gpu, sizeof(double));
    cudaMalloc(&totalColumnsDouble_gpu, sizeof(double));



	

	// ************************ Compute set up ************************
	// TODO: Compute the execution configuration (theads and blocks) :

	// TODO: Number Of Threads per block for the single precision row wise operation from the variable blockSizeSinglePrecisionRow
	dim3 dimBlockSingleRow(blockSizeSinglePrecisionRow);
	// TODO: Number Of Threads per block for the single precision column wise operation from the variable blockSizeSinglePrecisionColumn
	dim3 dimBlockSingleCol(blockSizeSinglePrecisionColumn);
	// TODO: Number Of Threads per block for the double precision row wise operation from the variable blockSizeDoublePrecisionRow
	dim3 dimBlockDoubleRow(blockSizeDoublePrecisionRow);
	// TODO: Number Of Threads per block for the double precision column wise operation from the variable blockSizeDoublePrecisionColumn
	dim3 dimBlockDoubleCol(blockSizeDoublePrecisionColumn);

	dim3 dimGridSingleCol((columns + dimBlockSingleCol.x - 1) / dimBlockSingleCol.x);
    dim3 dimGridDoubleCol((columns + dimBlockDoubleCol.x - 1) / dimBlockDoubleCol.x);


cudaEvent_t totalFloatStart, totalFloatEnd;
    cudaEventCreate(&totalFloatStart); cudaEventCreate(&totalFloatEnd);
    cudaEventRecord(totalFloatStart, 0); //Start timer

    int rowsPerStream = rows / numberOfStreams;
    /*
    //Depth first loop
    for (int i = 0; i < numberOfStreams; ++i) {
        int rowOffset = i * rowsPerStream;
        int currentRows = (i == numberOfStreams - 1) ? (rows - rowOffset) : rowsPerStream;
        int elemOffset = rowOffset * columns;

        //Async Transfer
        cudaMemcpyAsync(&matrixFloat_gpu[elemOffset], &matrixFloat1d[elemOffset], currentRows * columns * sizeof(float), cudaMemcpyHostToDevice, streamsFloat[i]);

        //Kernel Launch 
        dim3 dimGridChunk((currentRows + dimBlockSingleRow.x - 1) / dimBlockSingleRow.x);
        addMatrixRowsSinglePrecision<<<dimGridChunk, dimBlockSingleRow, dimBlockSingleRow.x * sizeof(float), streamsFloat[i]>>>(
            &matrixFloat_gpu[elemOffset], &rowsFloat_gpu[rowOffset], currentRows, columns);
    }
            */

    
    //Breadth first loop
    for (int i = 0; i < numberOfStreams; ++i) {
        int rowOffset = i * rowsPerStream;
        int currentRows = (i == numberOfStreams - 1) ? (rows - rowOffset) : rowsPerStream;
        int elemOffset = rowOffset * columns;
        cudaMemcpyAsync(&matrixFloat_gpu[elemOffset], &matrixFloat1d[elemOffset], currentRows * columns * sizeof(float), cudaMemcpyHostToDevice, streamsFloat[i]);
    }
    for (int i = 0; i < numberOfStreams; ++i) {
        int rowOffset = i * rowsPerStream;
        int currentRows = (i == numberOfStreams - 1) ? (rows - rowOffset) : rowsPerStream;
        int elemOffset = rowOffset * columns;
        dim3 dimGridChunk((currentRows + dimBlockSingleRow.x - 1) / dimBlockSingleRow.x);
        addMatrixRowsSinglePrecision<<<dimGridChunk, dimBlockSingleRow, dimBlockSingleRow.x * sizeof(float), streamsFloat[i]>>>(
            &matrixFloat_gpu[elemOffset], &rowsFloat_gpu[rowOffset], currentRows, columns);
    }
    

    //Wait for stream chunks to finish before reducing
    cudaDeviceSynchronize();

    //Reduce Rows
    cudaMemset(totalRowsFloat_gpu, 0, sizeof(float));
    int reduceGridSize = (rows + dimBlockSingleRow.x - 1) / dimBlockSingleRow.x;
    reduceVectorSinglePrecision<<<reduceGridSize, dimBlockSingleRow, dimBlockSingleRow.x * sizeof(float)>>>(rowsFloat_gpu, totalRowsFloat_gpu, rows);

    //Column Wise Addition 
    addMatrixColsSinglePrecision<<<dimGridSingleCol, dimBlockSingleCol>>>(matrixFloat_gpu, columnsFloat_gpu, rows, columns);
    
    //Reduce Columns
    cudaMemset(totalColumnsFloat_gpu, 0, sizeof(float));
    int reduceGridSizeCol = (columns + dimBlockSingleCol.x - 1) / dimBlockSingleCol.x;
    reduceVectorSinglePrecision<<<reduceGridSizeCol, dimBlockSingleCol, dimBlockSingleCol.x * sizeof(float)>>>(columnsFloat_gpu, totalColumnsFloat_gpu, columns);

    //Async Transfers back to Host
    cudaMemcpyAsync(&totalRowsFloat, totalRowsFloat_gpu, sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(&totalColumnsFloat, totalColumnsFloat_gpu, sizeof(float), cudaMemcpyDeviceToHost, 0);

    cudaEventRecord(totalFloatEnd, 0);
    cudaEventSynchronize(totalFloatEnd);
    
    float totalFloatPipelineTime;
    cudaEventElapsedTime(&totalFloatPipelineTime, totalFloatStart, totalFloatEnd);
    if (verbose) cout << "OVERALL Single Precision Pipeline Time: " << totalFloatPipelineTime * 0.001 << " seconds (Using " << numberOfStreams << " streams)" << endl;



    //Double precision add
    cudaEvent_t totalDoubleStart, totalDoubleEnd;
    cudaEventCreate(&totalDoubleStart); cudaEventCreate(&totalDoubleEnd);
    cudaEventRecord(totalDoubleStart, 0);

    for (int i = 0; i < numberOfStreams; ++i) {
        int rowOffset = i * rowsPerStream;
        int currentRows = (i == numberOfStreams - 1) ? (rows - rowOffset) : rowsPerStream;
        int elemOffset = rowOffset * columns;

        cudaMemcpyAsync(&matrixDouble_gpu[elemOffset], &matrixDouble1d[elemOffset], currentRows * columns * sizeof(double), cudaMemcpyHostToDevice, streamsDouble[i]);

        dim3 dimGridChunk((currentRows + dimBlockDoubleRow.x - 1) / dimBlockDoubleRow.x);
        addMatrixRowsDoublePrecision<<<dimGridChunk, dimBlockDoubleRow, dimBlockDoubleRow.x * sizeof(double), streamsDouble[i]>>>(
            &matrixDouble_gpu[elemOffset], &rowsDouble_gpu[rowOffset], currentRows, columns);
    }

    cudaDeviceSynchronize();

    cudaMemset(totalRowsDouble_gpu, 0, sizeof(double));
    int reduceGridSizeDoubleRow = (rows + dimBlockDoubleRow.x - 1) / dimBlockDoubleRow.x;
    reduceVectorDoublePrecision<<<reduceGridSizeDoubleRow, dimBlockDoubleRow, dimBlockDoubleRow.x * sizeof(double)>>>(rowsDouble_gpu, totalRowsDouble_gpu, rows);

    addMatrixColsDoublePrecision<<<dimGridDoubleCol, dimBlockDoubleCol>>>(matrixDouble_gpu, columnsDouble_gpu, rows, columns);

    cudaMemset(totalColumnsDouble_gpu, 0, sizeof(double));
    int reduceGridSizeDoubleCol = (columns + dimBlockDoubleCol.x - 1) / dimBlockDoubleCol.x;
    reduceVectorDoublePrecision<<<reduceGridSizeDoubleCol, dimBlockDoubleCol, dimBlockDoubleCol.x * sizeof(double)>>>(columnsDouble_gpu, totalColumnsDouble_gpu, columns);

    cudaMemcpyAsync(&totalRowsDouble, totalRowsDouble_gpu, sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(&totalColumnsDouble, totalColumnsDouble_gpu, sizeof(double), cudaMemcpyDeviceToHost, 0);

    cudaEventRecord(totalDoubleEnd, 0);
    cudaEventSynchronize(totalDoubleEnd);
    
    float totalDoublePipelineTime;
    cudaEventElapsedTime(&totalDoublePipelineTime, totalDoubleStart, totalDoubleEnd);
    if (verbose) cout << "OVERALL Double Precision Pipeline Time: " << totalDoublePipelineTime * 0.001 << " seconds (Using " << numberOfStreams << " streams)" << endl;


    
    //Unpin Host Memory
    cudaHostUnregister(matrixFloat1d.data());
    cudaHostUnregister(matrixDouble1d.data());

    //Destroy Streams
    for (int i = 0; i < numberOfStreams; ++i) {
        cudaStreamDestroy(streamsFloat[i]);
        cudaStreamDestroy(streamsDouble[i]);
    }
    delete[] streamsFloat;
    delete[] streamsDouble;

    //Free Device Memory
    cudaFree(matrixFloat_gpu);
    cudaFree(rowsFloat_gpu);
    cudaFree(columnsFloat_gpu);
    cudaFree(totalRowsFloat_gpu);
    cudaFree(totalColumnsFloat_gpu);
    cudaFree(matrixDouble_gpu);
    cudaFree(rowsDouble_gpu);
    cudaFree(columnsDouble_gpu);
    cudaFree(totalRowsDouble_gpu);
    cudaFree(totalColumnsDouble_gpu);

    cudaDeviceReset();
    return 0;
}


// Choose card to use - will find all the cards and choose the one with more multi-processors
int chooseCudaCard (bool verbose) {
	int i,numberOfDevices,best,bestNumberOfMultiprocessors;
	int numberOfCUDAcoresForThisCC=0;
	struct cudaDeviceProp x;

	if ( cudaGetDeviceCount(&numberOfDevices)!=cudaSuccess ) {
		cout << "No CUDA-enabled devices were found " << endl;
	}
	cout << "***************************************************" << endl;
	cout << "Found " << numberOfDevices << " CUDA-enabled devices" << endl;
	best=-1;
	bestNumberOfMultiprocessors=-1;
	for (i=0;i<numberOfDevices;i++) {
		cudaGetDeviceProperties(&x, i);
		if (verbose) {
			cout << "========================= IDENTITY DATA ==================================" << endl;
			cout << "GPU model name: " << x.name << endl;
			if (x.integrated==1) {
				cout << "GPU The device is an integrated (motherboard) GPU" << endl;
			} else {
				cout << "GPU The device is NOT an integrated (motherboard) GPU - i.e. it is a discrete device" << endl;
			}
			cout << "GPU pciBusID: " << x.pciBusID << endl;
			cout << "GPU pciDeviceID: " << x.pciDeviceID << endl;
			cout << "GPU pciDomainID: " << x.pciDomainID << endl;
			if (x.tccDriver==1) {
				cout << "the device is a Tesla one using TCC driver" << endl;
			} else {
				cout << "the device is NOT a Tesla one using TCC driver" << endl;
			}
			cout << "========================= COMPUTE DATA ==================================" << endl;
			cout << "GPU Compute capability: " << x.major << "." << x.minor << endl;
		}
		switch (x.major) {
			case 1:	// Tesla / T10
				numberOfCUDAcoresForThisCC=8;
				break;
			case 2:	// Fermi
				switch (x.minor) {
					case 0: // 2.0
						numberOfCUDAcoresForThisCC=32;
						break;
					case 1: // 2.1
						numberOfCUDAcoresForThisCC=48;
						break;
					default: // Unknown
						numberOfCUDAcoresForThisCC=0;
						break;
				}
				break;
			case 3:	// Kepler
				numberOfCUDAcoresForThisCC=192;
				break;
			case 5:	// Maxwell
				numberOfCUDAcoresForThisCC=128;
				break;
			case 6:	// Pascal
				switch (x.minor) {
					case 0: // GP100, 64 cuda cores per SM - 7.0 should be prefered over 7.1
						numberOfCUDAcoresForThisCC=64;
						break;
					case 1: // GP102, GP104, GP106, GP107, 128 cuda cores per SM
					case 2: // GP10B, Pascal Tegra cards  - still 128 cuda cores per SM
						numberOfCUDAcoresForThisCC=128;
						break;
					default: // Unknown - 6.2 is the GP10B on Jetson TX2, DRIVE PX 2
						numberOfCUDAcoresForThisCC=0;
						break;
				}
				break;
			case 7:	// Volta is 7.0 and 7.2, 64 cuda cores per SM, Turing is 7.5 - also has 64 cuda cores per SM
				numberOfCUDAcoresForThisCC=64;
				break;
			case 8:	// Ampere 8.x, with x < 9, has 64 cuda cores per SM, but Ada Lovelace (8.9) has 128 cuda cores per SM
				switch (x.minor) {
					case 0: // The GA100 in the A100 is an Ampere (8.0) with  has 64 cuda cores per SM
						numberOfCUDAcoresForThisCC=64;
						break;
					case 6: // The Geforce 3000 series is an Ampere (8.6) with 128 cuda cores per SM
						numberOfCUDAcoresForThisCC=128;
						break;
					case 9: // The Geforce 4000 series are Ada Lovelace (8.9) with 128 cuda cores per SM
						numberOfCUDAcoresForThisCC=128;
						break;
					default: // Unknown - 6.2 is the GP10B on Jetson TX2, DRIVE PX 2
						numberOfCUDAcoresForThisCC=64;
						break;
				}
				break;
			case 9:	// Hopper (G100 is 9.0) and Grace Hopper both have 128 cuda cores per SM
				numberOfCUDAcoresForThisCC=128;
				break;
			case 10: // Blackwell has 128 cuda cores per SM
			case 12: // Blackwell has 128 cuda cores per SM
				numberOfCUDAcoresForThisCC=128;
				break;
			default: // Unknown
				numberOfCUDAcoresForThisCC=0;
				break;
		}
		if (x.multiProcessorCount>bestNumberOfMultiprocessors*numberOfCUDAcoresForThisCC) {
			best=i;
			bestNumberOfMultiprocessors=x.multiProcessorCount*numberOfCUDAcoresForThisCC;
		}
		if (verbose) {
			int clockRateValue;
			cudaDeviceGetAttribute(&clockRateValue,cudaDevAttrClockRate,i);
			cout << "GPU Clock frequency in hertzs: " << clockRateValue << endl;
			//cout << "GPU Clock frequency in hertzs: %" << x.clockRate << endl; // REMOVED IN CUDA 13.0+!!
			cout << "GPU number of multi-processors: " << x.multiProcessorCount << endl;
			cout << "GPU maximum number of threads per multi-processor: " << x.maxThreadsPerMultiProcessor << endl;
			cout << "GPU Maximum size of each dimension of a grid: " << x.maxGridSize[0]<<","<<x.maxGridSize[1]<<","<<x.maxGridSize[2] << endl;
			cout << "GPU Maximum size of each dimension of a block: " << x.maxThreadsDim[0]<<","<<x.maxThreadsDim[1]<<","<<x.maxThreadsDim[2] << endl;
			cout << "GPU Maximum number of threads per block: " << x.maxThreadsPerBlock << endl;
			cout << "GPU Maximum pitch in bytes allowed by memory copies: " << (unsigned int)(x.memPitch) << endl;
			int computeModeValue;
			cudaDeviceGetAttribute(&computeModeValue,cudaDevAttrComputeMode,i);
			cout << "GPU Compute mode is: " << computeModeValue << endl;
			//cout << "GPU Compute mode is: " << x.computeMode << endl; // REMOVED IN CUDA 13.0+!!
			cout << "========================= MEMORY DATA ==================================" << endl;
			cout << "GPU total global memory: " << (size_t)(x.totalGlobalMem) << " bytes" << endl;
			int memoryClockRateValue;
			cudaDeviceGetAttribute(&memoryClockRateValue,cudaDevAttrMemoryClockRate,i);
			cout << "GPU peak memory clock frequency in kilohertz: " << memoryClockRateValue << endl;
			//cout << "GPU peak memory clock frequency in kilohertz: " << x.memoryClockRate << endl; // REMOVED IN CUDA 13.0+!!

			cout << "GPU memory bus width: " << x.memoryBusWidth << " bits" << endl;
			cout << "GPU L2 cache size: " << x.l2CacheSize << " bytes" << endl;
			cout << "GPU 32-bit registers available per block: " << x.regsPerBlock << endl;
			cout << "GPU Shared memory available per block in bytes:" << (int)(x.sharedMemPerBlock) << endl;
			cout << "GPU Alignment requirement for textures: " << (int)(x.textureAlignment) << endl;
			cout << "GPU Constant memory available on device in bytes: " << (int)(x.totalConstMem) << endl;
			cout << "GPU Warp size in threads: " << x.warpSize << endl;
			cout << "GPU maximum 1D texture size: " << x.maxTexture1D << endl;
			cout << "GPU maximum 2D texture size: " << x.maxTexture2D[0] << "," << x.maxTexture2D[1] << endl;
			cout << "GPU maximum 3D texture size: " << x.maxTexture3D[0] << "," << x.maxTexture3D[1] << "," << x.maxTexture3D[2] << endl;
			cout << "GPU maximum 1D layered texture dimensions: " << x.maxTexture1DLayered[0] << "," << x.maxTexture1DLayered[1] << endl;
			cout << "GPU maximum 2D layered texture dimensions: " << x.maxTexture2DLayered[0] << "," << x.maxTexture2DLayered[1] << "," << x.maxTexture2DLayered[2] << endl;
			cout << "GPU surface alignment: " << (int)(x.surfaceAlignment) << endl;
			if (x.canMapHostMemory==1) {
				cout << "GPU The device can map host memory into the CUDA address space" << endl;
			} else {
				cout << "GPU The device can NOT map host memory into the CUDA address space" << endl;
			}
			if (x.ECCEnabled==1) {
				cout << "GPU memory has ECC support" << endl;
			} else {
				cout << "GPU memory does not have ECC support" << endl;
			}
			if (x.unifiedAddressing==1) {
				cout << "GPU The device shares an unified address space with the host" << endl;
			} else {
				cout << "GPU The device DOES NOT share an unified address space with the host" << endl;
			}
			cout << "========================= EXECUTION DATA ==================================" << endl;
			if (x.concurrentKernels==1) {
				cout << "GPU Concurrent kernels are allowed" << endl;
			} else {
				cout << "GPU Concurrent kernels are NOT allowed" << endl;
			}

			int kernelTimeoutValue;
			cudaDeviceGetAttribute(&kernelTimeoutValue,cudaDevAttrKernelExecTimeout,i);
			if (kernelTimeoutValue==1) {
				cout << "GPU There is a run time limit for kernels executed in the device" << endl;
			} else {
				cout << "GPU There is NOT a run time limit for kernels executed in the device" << endl;
			}
			//if (x.kernelExecTimeoutEnabled==1) {														// REMOVED IN CUDA 13.0+!!
			//	cout << "GPU There is a run time limit for kernels executed in the device" << endl;		// REMOVED IN CUDA 13.0+!!
			//} else {																					// REMOVED IN CUDA 13.0+!!
			//	cout << "GPU There is NOT a run time limit for kernels executed in the device" << endl;	// REMOVED IN CUDA 13.0+!!
			//}																							// REMOVED IN CUDA 13.0+!!

			if (x.asyncEngineCount==1) {
				cout << "GPU The device can concurrently copy memory between host and device while executing a kernel" << endl;
			} else if (x.asyncEngineCount==2) {
				cout << "GPU The device can concurrently copy memory between host and device in both directions and execute a kernel at the same time" << endl;
			} else {
				cout << "GPU the device is NOT capable of concurrently memory copying" << endl;
			}
		}
	}
	// set the best device
	if (best>=0) {
		cudaGetDeviceProperties(&x, best);
		cout << "Choosing " << x.name << endl;
		cudaSetDevice(best);
	}

	// We return the number of devices, in case we want to use more than one
	cout << "***************************************************" << endl;
	return (numberOfDevices);
}


