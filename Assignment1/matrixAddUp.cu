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
	bool verbose,unsigned int printPrecision) {

	cudaError_t err;
	// Allocate the variables in the global memory

	// ************************ Single Precision allocation ************************
	// Cuda Timing
	cudaEvent_t allocateFloatGpuStart, allocateFloatGpuEnd;
	float allocateFloatGpuElapsedTime,allocateFloatGpuTime;
	cudaEventCreate(&allocateFloatGpuStart);
	cudaEventCreate(&allocateFloatGpuEnd);
	cudaEventRecord(allocateFloatGpuStart, 0); // We use 0 here because it is the "default" stream

	// TODO: Allocate a matrix of single precision values of size rows*columns
	float *matrixFloat_gpu;
	err = cudaMalloc(&matrixFloat_gpu, sizeof(float) * (rows * columns));
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating matrixFloat_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// 

	// TODO: Allocate a vector of single precision values of size rows
	float *rowsFloat_gpu;
	err = cudaMalloc(&rowsFloat_gpu, sizeof(float) * rows);
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating rowsFloat_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	

	// TODO: Allocate a vector of single precision values of size columns
	float *columnsFloat_gpu;
	err = cudaMalloc(&columnsFloat_gpu, sizeof(float) * columns);
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating columnsFloat_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// TODO: Allocate one single precision value for the reduced rowsFloat_gpu vector
	float *totalRowsFloat_gpu;
	err = cudaMalloc(&totalRowsFloat_gpu, sizeof(float));
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating totalRowsFloat_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// TODO: Allocate one single precision value for the reduced columnsFloat_gpu vector
	float *totalColumnsFloat_gpu;
	err = cudaMalloc(&totalColumnsFloat_gpu, sizeof(float));


	// Cuda Timing
	cudaEventRecord(allocateFloatGpuEnd, 0);
	cudaEventSynchronize(allocateFloatGpuStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(allocateFloatGpuEnd); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&allocateFloatGpuElapsedTime, allocateFloatGpuStart, allocateFloatGpuEnd);
	allocateFloatGpuTime=(double)(allocateFloatGpuElapsedTime)*0.001;
	cudaLastErrorCheck("allocateFloat");

	if (verbose) cout << "allocateFloatGpuTime: " << allocateFloatGpuTime << endl;

	// ************************ Double precision allocation ************************
	// Cuda Timing
	cudaEvent_t allocateDoubleGpuStart, allocateDoubleGpuEnd;
	float allocateDoubleGpuElapsedTime,allocateDoubleGpuTime;
	cudaEventCreate(&allocateDoubleGpuStart);
	cudaEventCreate(&allocateDoubleGpuEnd);
	cudaEventRecord(allocateDoubleGpuStart, 0); // We use 0 here because it is the "default" stream

	// TODO: Allocate a matrix of double precision values of size rows*columns
	double *matrixDouble_gpu;
	err = cudaMalloc(&matrixDouble_gpu, sizeof(double) * (rows * columns));
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating matrixDouble_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// TODO: Allocate a vector of double precision values of size rows
	double *rowsDouble_gpu;
	err = cudaMalloc(&rowsDouble_gpu, sizeof(double) * rows);
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating rowsDouble_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// TODO: Allocate a vector of double precision values of size columns
	double *columnsDouble_gpu;
	err = cudaMalloc(&columnsDouble_gpu, sizeof(double) * columns);
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating columnsDouble_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// TODO: Allocate one double precision value for the reduced rowsDouble_gpu vector
	double *totalRowsDouble_gpu;
	err = cudaMalloc(&totalRowsDouble_gpu, sizeof(double));
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating totalRowsDouble_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// TODO: Allocate one double precision value for the reduced columnsDouble_gpu vector
	double *totalColumnsDouble_gpu;
	err = cudaMalloc(&totalColumnsDouble_gpu, sizeof(double));
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Allocating totalColumnsDouble_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Cuda Timing
	cudaEventRecord(allocateDoubleGpuEnd, 0);
	cudaEventSynchronize(allocateDoubleGpuStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(allocateDoubleGpuEnd); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&allocateDoubleGpuElapsedTime, allocateDoubleGpuStart, allocateDoubleGpuEnd);
	allocateDoubleGpuTime=(float)(allocateDoubleGpuElapsedTime)*0.001;
	cudaLastErrorCheck("allocateDouble_rows_kernel");

	if (verbose) cout << "allocateDoubleGpuTime: " << allocateDoubleGpuTime << endl;

	// ************************ Single precision transfer to device ************************
	// Cuda Timing
	cudaEvent_t transferFloatGpuStart, transferFloatGpuEnd;
	float transferFloatGpuElapsedTime,transferFloatGpuTime;
	cudaEventCreate(&transferFloatGpuStart);
	cudaEventCreate(&transferFloatGpuEnd);
	cudaEventRecord(transferFloatGpuStart, 0); // We use 0 here because it is the "default" stream

	// TODO: Copy the single precision matrix (matrixFloat1d) into the global memory of the GPU (matrixFloat_gpu)
	err = cudaMemcpy(matrixFloat_gpu, matrixFloat1d.data(), sizeof(float) * (rows * columns), cudaMemcpyHostToDevice);
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Error copying matrixFloat1d to matrixFloat_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Cuda Timing
	cudaEventRecord(transferFloatGpuEnd, 0);
	cudaEventSynchronize(transferFloatGpuStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(transferFloatGpuEnd); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&transferFloatGpuElapsedTime, transferFloatGpuStart, transferFloatGpuEnd);
	transferFloatGpuTime=(float)(transferFloatGpuElapsedTime)*0.001;
	if (verbose) cout << "transferFloatGpuTime: " << transferFloatGpuTime << endl;

	// ************************ Double transfer to device ************************
	// Cuda Timing
	cudaEvent_t transferDoubleGpuStart, transferDoubleGpuEnd;
	float transferDoubleGpuElapsedTime,transferDoubleGpuTime;
	cudaEventCreate(&transferDoubleGpuStart);
	cudaEventCreate(&transferDoubleGpuEnd);
	cudaEventRecord(transferDoubleGpuStart, 0); // We use 0 here because it is the "default" stream

	// TODO: Copy the double precision matrix (matrixDouble1d) into the global memory of the GPU (matrixDouble_gpu)
	err = cudaMemcpy(matrixDouble_gpu, matrixDouble1d.data(), sizeof(double) * (rows * columns), cudaMemcpyHostToDevice);
	if (cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n", "Error copying matrixDouble1d to matrixDouble_gpu", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Cuda Timing
	cudaEventRecord(transferDoubleGpuEnd, 0);
	cudaEventSynchronize(transferDoubleGpuStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(transferDoubleGpuEnd); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&transferDoubleGpuElapsedTime, transferDoubleGpuStart, transferDoubleGpuEnd);
	transferDoubleGpuTime=(double)(transferDoubleGpuElapsedTime)*0.001;

	if (verbose) cout << "transferDoubleGpuTime: " << transferDoubleGpuTime << endl;

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

	// TODO: Number Of blocks for the single precision row wise operation from the variables blockSizeSinglePrecisionRow and rows
	dim3 dimGridSingleRow((rows / dimBlockSingleRow.x) + (!(rows % dimBlockSingleRow.x) ? 0 : 1));
	// TODO: Number Of blocks for the single precision column wise operation from the variables blockSizeSinglePrecisionColumn and columns
	dim3 dimGridSingleCol((columns / dimBlockSingleCol.x) + (!(columns % dimBlockSingleCol.x) ? 0 : 1));
	// TODO: Number Of blocks for the double precision row wise operation from the variables blockSizeSinglePrecisionRow and rows
	dim3 dimGridDoubleRow((rows / dimBlockDoubleRow.x) + (!(rows % dimBlockDoubleRow.x) ? 0 : 1));
	// TODO: Number Of blocks for the double precision column wise operation from the variables blockSizeSinglePrecisionColumn and columns
	dim3 dimGridDoubleCol((columns / dimBlockDoubleCol.x) + (!(columns % dimBlockDoubleCol.x) ? 0 : 1));


	if (verbose) {
		// TODO: Print the number of threads per block and number of blocks for each one of the four cases
		printf("Single Precision Row-wise: %d threads/block, %d blocks\n", dimBlockSingleRow.x, dimGridSingleRow.x);
        printf("Single Precision Col-wise: %d threads/block, %d blocks\n", dimBlockSingleCol.x, dimGridSingleCol.x);
        printf("Double Precision Row-wise: %d threads/block, %d blocks\n", dimBlockDoubleRow.x, dimGridDoubleRow.x);
        printf("Double Precision Col-wise: %d threads/block, %d blocks\n", dimBlockDoubleCol.x, dimGridDoubleCol.x);
	}

	// ************************ Single precision compute ************************
	// ************************ 
	// Add row wise:

	// --- Time the Add Kernel ---
    cudaEvent_t addRowsFloatGpuStart, addRowsFloatGpuEnd;
    cudaEventCreate(&addRowsFloatGpuStart);
    cudaEventCreate(&addRowsFloatGpuEnd);
    cudaEventRecord(addRowsFloatGpuStart, 0);

    // Call the kernel that adds the rows
    addMatrixRowsSinglePrecision<<<rows, dimBlockSingleRow, dimBlockSingleRow.x * sizeof(float)>>>(matrixFloat_gpu, rowsFloat_gpu, rows, columns);

    cudaEventRecord(addRowsFloatGpuEnd, 0);
    cudaEventSynchronize(addRowsFloatGpuEnd);
    
    float addRowsFloatGpuElapsedTime;
    cudaEventElapsedTime(&addRowsFloatGpuElapsedTime, addRowsFloatGpuStart, addRowsFloatGpuEnd);
    timeAddRowsFloatGpu = addRowsFloatGpuElapsedTime * 0.001;
    if (verbose) cout << "Add Matrix Rows Float Gpu Time: " << timeAddRowsFloatGpu << endl;


    // --- Time the Reduce Kernel ---
    cudaEvent_t reduceRowsFloatGpuStart, reduceRowsFloatGpuEnd;
    cudaEventCreate(&reduceRowsFloatGpuStart);
    cudaEventCreate(&reduceRowsFloatGpuEnd);
    cudaEventRecord(reduceRowsFloatGpuStart, 0);

    // Set the target value to 0 before accumulating
    cudaMemset(totalRowsFloat_gpu, 0, sizeof(float));
    // Calculate grid size for the 1D vector reduction
    int reduceGridSize = (rows + dimBlockSingleRow.x - 1) / dimBlockSingleRow.x;
    
    // Call the kernel that reduces the vector
    reduceVectorSinglePrecision<<<reduceGridSize, dimBlockSingleRow, dimBlockSingleRow.x * sizeof(float)>>>(rowsFloat_gpu, totalRowsFloat_gpu, rows);

    cudaEventRecord(reduceRowsFloatGpuEnd, 0);
    cudaEventSynchronize(reduceRowsFloatGpuEnd);
    
    float reduceRowsFloatGpuElapsedTime;
    cudaEventElapsedTime(&reduceRowsFloatGpuElapsedTime, reduceRowsFloatGpuStart, reduceRowsFloatGpuEnd);
    timeReduceRowsFloatGpu = reduceRowsFloatGpuElapsedTime * 0.001;
    if (verbose) cout << "Reduce Vector Float Gpu Time: " << timeReduceRowsFloatGpu << endl;

	// ************************ 
	// Add column wise:

	// --- Time the Add Kernel ---
    cudaEvent_t addColsFloatGpuStart, addColsFloatGpuEnd;
    cudaEventCreate(&addColsFloatGpuStart);
    cudaEventCreate(&addColsFloatGpuEnd);
    cudaEventRecord(addColsFloatGpuStart, 0);

    // Call the kernel that adds the columns
    addMatrixColsSinglePrecision<<<dimGridSingleCol, dimBlockSingleCol>>>(matrixFloat_gpu, columnsFloat_gpu, rows, columns);

    cudaEventRecord(addColsFloatGpuEnd, 0);
    cudaEventSynchronize(addColsFloatGpuEnd);
    
    float addColsFloatGpuElapsedTime;
    cudaEventElapsedTime(&addColsFloatGpuElapsedTime, addColsFloatGpuStart, addColsFloatGpuEnd);
    timeAddColumnsFloatGpu = addColsFloatGpuElapsedTime * 0.001;
    if (verbose) cout << "Add Matrix Cols Float Gpu Time: " << timeAddColumnsFloatGpu << endl;


    // --- Time the Reduce Kernel ---
    cudaEvent_t reduceColsFloatGpuStart, reduceColsFloatGpuEnd;
    cudaEventCreate(&reduceColsFloatGpuStart);
    cudaEventCreate(&reduceColsFloatGpuEnd);
    cudaEventRecord(reduceColsFloatGpuStart, 0);

    // Set the target value to 0 before accumulating
    cudaMemset(totalColumnsFloat_gpu, 0, sizeof(float));
    // Calculate grid size for the 1D vector reduction (using columns instead of rows)
    int reduceGridSizeCol = (columns + dimBlockSingleCol.x - 1) / dimBlockSingleCol.x;
    
    // Call the kernel that reduces the vector
    reduceVectorSinglePrecision<<<reduceGridSizeCol, dimBlockSingleCol, dimBlockSingleCol.x * sizeof(float)>>>(columnsFloat_gpu, totalColumnsFloat_gpu, columns);

    cudaEventRecord(reduceColsFloatGpuEnd, 0);
    cudaEventSynchronize(reduceColsFloatGpuEnd);
    
    float reduceColsFloatGpuElapsedTime;
    cudaEventElapsedTime(&reduceColsFloatGpuElapsedTime, reduceColsFloatGpuStart, reduceColsFloatGpuEnd);
    timeReduceColumnsFloatGpu = reduceColsFloatGpuElapsedTime * 0.001;
    if (verbose) cout << "Reduce Vector Cols Float Gpu Time: " << timeReduceColumnsFloatGpu << endl;

	// ************************ Double compute ************************
	// ************************ 
	// Add row wise:

	// Cuda Timing
	// --- Time the Add Kernel ---
    cudaEvent_t addRowsDoubleGpuStart, addRowsDoubleGpuEnd;
    cudaEventCreate(&addRowsDoubleGpuStart);
    cudaEventCreate(&addRowsDoubleGpuEnd);
    cudaEventRecord(addRowsDoubleGpuStart, 0);

    // Call the kernel that adds the rows
    addMatrixRowsDoublePrecision<<<rows, dimBlockDoubleRow, dimBlockDoubleRow.x * sizeof(double)>>>(matrixDouble_gpu, rowsDouble_gpu, rows, columns);

    cudaEventRecord(addRowsDoubleGpuEnd, 0);
    cudaEventSynchronize(addRowsDoubleGpuEnd);
    
    float addRowsDoubleGpuElapsedTime;
    cudaEventElapsedTime(&addRowsDoubleGpuElapsedTime, addRowsDoubleGpuStart, addRowsDoubleGpuEnd);
    timeAddRowsDoubleGpu = addRowsDoubleGpuElapsedTime * 0.001;
    if (verbose) cout << "Add Matrix Rows Double Gpu Time: " << timeAddRowsDoubleGpu << endl;


    // --- Time the Reduce Kernel ---
    cudaEvent_t reduceRowsDoubleGpuStart, reduceRowsDoubleGpuEnd;
    cudaEventCreate(&reduceRowsDoubleGpuStart);
    cudaEventCreate(&reduceRowsDoubleGpuEnd);
    cudaEventRecord(reduceRowsDoubleGpuStart, 0);

    // Set the target value to 0 before accumulating
    cudaMemset(totalRowsDouble_gpu, 0, sizeof(double));
    // Calculate grid size for the 1D vector reduction
    int reduceGridSizeDoubleRow = (rows + dimBlockDoubleRow.x - 1) / dimBlockDoubleRow.x;
    
    // Call the kernel that reduces the vector
    reduceVectorDoublePrecision<<<reduceGridSizeDoubleRow, dimBlockDoubleRow, dimBlockDoubleRow.x * sizeof(double)>>>(rowsDouble_gpu, totalRowsDouble_gpu, rows);

    cudaEventRecord(reduceRowsDoubleGpuEnd, 0);
    cudaEventSynchronize(reduceRowsDoubleGpuEnd);
    
    float reduceRowsDoubleGpuElapsedTime;
    cudaEventElapsedTime(&reduceRowsDoubleGpuElapsedTime, reduceRowsDoubleGpuStart, reduceRowsDoubleGpuEnd);
    timeReduceRowsDoubleGpu = reduceRowsDoubleGpuElapsedTime * 0.001;
    if (verbose) cout << "Reduce Vector Rows Double Gpu Time: " << timeReduceRowsDoubleGpu << endl;

	// ************************ 
	// Add vertically:

	// --- Time the Add Kernel ---
    cudaEvent_t addColsDoubleGpuStart, addColsDoubleGpuEnd;
    cudaEventCreate(&addColsDoubleGpuStart);
    cudaEventCreate(&addColsDoubleGpuEnd);
    cudaEventRecord(addColsDoubleGpuStart, 0);

    // Call the kernel that adds the columns
    addMatrixColsDoublePrecision<<<dimGridDoubleCol, dimBlockDoubleCol>>>(matrixDouble_gpu, columnsDouble_gpu, rows, columns);

    cudaEventRecord(addColsDoubleGpuEnd, 0);
    cudaEventSynchronize(addColsDoubleGpuEnd);
    
    float addColsDoubleGpuElapsedTime;
    cudaEventElapsedTime(&addColsDoubleGpuElapsedTime, addColsDoubleGpuStart, addColsDoubleGpuEnd);
    timeAddColumnsDoubleGpu = addColsDoubleGpuElapsedTime * 0.001;
    if (verbose) cout << "Add Matrix Cols Double Gpu Time: " << timeAddColumnsDoubleGpu << endl;


    // --- Time the Reduce Kernel ---
    cudaEvent_t reduceColsDoubleGpuStart, reduceColsDoubleGpuEnd;
    cudaEventCreate(&reduceColsDoubleGpuStart);
    cudaEventCreate(&reduceColsDoubleGpuEnd);
    cudaEventRecord(reduceColsDoubleGpuStart, 0);

    // Set the target value to 0 before accumulating
    cudaMemset(totalColumnsDouble_gpu, 0, sizeof(double));
    // Calculate grid size for the 1D vector reduction
    int reduceGridSizeDoubleCol = (columns + dimBlockDoubleCol.x - 1) / dimBlockDoubleCol.x;
    
    // Call the kernel that reduces the vector
    reduceVectorDoublePrecision<<<reduceGridSizeDoubleCol, dimBlockDoubleCol, dimBlockDoubleCol.x * sizeof(double)>>>(columnsDouble_gpu, totalColumnsDouble_gpu, columns);

    cudaEventRecord(reduceColsDoubleGpuEnd, 0);
    cudaEventSynchronize(reduceColsDoubleGpuEnd);
    
    float reduceColsDoubleGpuElapsedTime;
    cudaEventElapsedTime(&reduceColsDoubleGpuElapsedTime, reduceColsDoubleGpuStart, reduceColsDoubleGpuEnd);
    timeReduceColumnsDoubleGpu = reduceColsDoubleGpuElapsedTime * 0.001;
    if (verbose) cout << "Reduce Vector Cols Double Gpu Time: " << timeReduceColumnsDoubleGpu << endl;

	// ************************ Single precision transfer back to host ************************

	// Cuda Timing
	cudaEvent_t transferBackFloatGpuStart, transferBackFloatGpuEnd;
	float transferBackFloatGpuElapsedTime,transferBackFloatGpuTime;
	cudaEventCreate(&transferBackFloatGpuStart);
	cudaEventCreate(&transferBackFloatGpuEnd);
	cudaEventRecord(transferBackFloatGpuStart, 0); // We use 0 here because it is the "default" stream

	// TODO: Copy totalRowsFloat_gpu into totalRowsFloat
	err = cudaMemcpy(&totalRowsFloat, totalRowsFloat_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaSuccess != err) {
        printf("(Cuda error %s): %s\n", "Error copying totalRowsFloat_gpu to host", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	// TODO: Copy totalColumnsFloat_gpu into totalColumnsFloat
	err = cudaMemcpy(&totalColumnsFloat, totalColumnsFloat_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaSuccess != err) {
        printf("(Cuda error %s): %s\n", "Error copying totalColumnsFloat_gpu to host", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


	// Cuda Timing
	cudaEventRecord(transferBackFloatGpuEnd, 0);
	cudaEventSynchronize(transferBackFloatGpuStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(transferBackFloatGpuEnd); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&transferBackFloatGpuElapsedTime, transferBackFloatGpuStart, transferBackFloatGpuEnd);
	transferBackFloatGpuTime=(float)(transferBackFloatGpuElapsedTime)*0.001;
	cudaLastErrorCheck("transferBackFloat_rows_kernel");
	if (verbose) cout << "transferBackFloatGpuTime: " << transferBackFloatGpuTime << endl;

	// ************************ Double precision transfer back to host ************************

	// Cuda Timing
	cudaEvent_t transferBackDoubleGpuStart, transferBackDoubleGpuEnd;
	float transferBackDoubleGpuElapsedTime,transferBackDoubleGpuTime;
	cudaEventCreate(&transferBackDoubleGpuStart);
	cudaEventCreate(&transferBackDoubleGpuEnd);
	cudaEventRecord(transferBackDoubleGpuStart, 0); // We use 0 here because it is the "default" stream

	// TODO: Copy totalRowsDouble_gpu into totalRowsDouble
	err = cudaMemcpy(&totalRowsDouble, totalRowsDouble_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaSuccess != err) {
        printf("(Cuda error %s): %s\n", "Error copying totalRowsDouble_gpu to host", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	// TODO: Copy totalColumnsDouble_gpu into totalColumnsDouble
	err = cudaMemcpy(&totalColumnsDouble, totalColumnsDouble_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaSuccess != err) {
        printf("(Cuda error %s): %s\n", "Error copying totalColumnsDouble_gpu to host", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Cuda Timing
	cudaEventRecord(transferBackDoubleGpuEnd, 0);
	cudaEventSynchronize(transferBackDoubleGpuStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(transferBackDoubleGpuEnd); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&transferBackDoubleGpuElapsedTime, transferBackDoubleGpuStart, transferBackDoubleGpuEnd);
	transferBackDoubleGpuTime=(double)(transferBackDoubleGpuElapsedTime)*0.001;
	cudaLastErrorCheck("transferBackDouble_rows_kernel");
	if (verbose) cout << "transferBackDoubleGpuTime: " << transferBackDoubleGpuTime << endl;

	// Free the memory
	// TODO: Free matrixFloat_gpu,rowsFloat_gpu, columnsFloat_gpu, totalRowsFloat_gpu and totalColumnsFloat_gpu
	// TODO: Free matrixDouble_gpu,rowsDouble_gpu, columnsDouble_gpu, totalRowsDouble_gpu and totalColumnsDouble_gpu
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


