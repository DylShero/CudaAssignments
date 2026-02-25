///// Created by Jose Mauricio Refojo - 2018-01-23		Last changed: 2026-02-16
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include <getopt.h>
#include <iomanip>      // std::setprecision
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

#include "matrixAddUp.h"

using namespace std;

int blockSizeSinglePrecisionRow, blockSizeSinglePrecisionColumn, blockSizeDoublePrecisionRow, blockSizeDoublePrecisionColumn;
long int seed;
unsigned int columns,rows,printPrecision;
bool verbose,timing,cpu,gpu;

extern inline int cudaMatrixAddUp (std::vector< float > &,std::vector< double > &, int,int,float&,float&,double &,double &,double &,double &,double &,double &, int &,int &,int &,int &,bool,unsigned int);
//extern inline int add_vectors(void);
extern inline int		chooseCudaCard			(bool verbose);

void	generateRandomMatrix	(std::vector< float > &,std::vector< float* > &, unsigned int , unsigned int);
int		parseArguments		(int argc, char **argv);
void	printUsage		(void);


template<class T> T addVectorCpu  (std::vector< T > &vector) {
		T total=0;
		unsigned int ui;

		for (ui=0;ui<vector.size();ui++) {
			total+=vector[ui];
		}
//		cout << "matrixAddUp::main::addVectorCpu total= " << total << endl;
		return total;
}

template<class T> void addRowsCpu  (std::vector< T* > const &matrix,std::vector< T > &vector) {
//		T total=0;
		unsigned int ui,uj;

		// Make sure that the size is correct
		if (vector.size()!=rows) {
			try {
				vector.resize(rows);
			} catch (std::bad_alloc const&) {;
				cout << "matrixAddUp::main::addRowsCpu vector memory allocation fail!" << endl;	exit(1);
			}
		}
		for (ui=0;ui<rows;ui++) {
			vector[ui]=0;
			for (uj=0;uj<columns;uj++) {
				vector[ui]+=fabs(matrix[ui][uj]);
//				total+=fabs(matrix[ui][uj]);
			}
		}
//		cout << "matrixAddUp::main::addRowsCpu total= " << total << endl;
		return;
}

template<class T> void addColumnsCpu  (std::vector< T* > const &matrix,std::vector< T > &vector) {
//		T total=0;
		unsigned int ui,uj;

		// Make sure that the size is correct
		if (vector.size()!=columns) {
			try {
				vector.resize(columns);
			} catch (std::bad_alloc const&) {;
				cout << "matrixAddUp::main::addColumnsCpu vector memory allocation fail!" << endl;	exit(1);
			}
		}
		for (uj=0;uj<columns;uj++) {
			vector[uj]=0;
			for (ui=0;ui<rows;ui++) {
				vector[uj]+=fabs(matrix[ui][uj]);
//				total+=fabs(matrix[ui][uj]);
			}
		}
//		cout << "matrixAddUp::main::addRowsCpu total= " << total << endl;
		return;
}

template<class T> void generateRandomMatrix (std::vector< T > &matrix1dFloat,std::vector< T* > &matrixPFloat, unsigned int numberOfRows, unsigned int numberOfColumns) {
	unsigned int ui,uj;
	try {
		matrixPFloat.resize(numberOfRows);
		matrix1dFloat.resize(numberOfRows*numberOfColumns);
		matrixPFloat.resize(numberOfRows);
	} catch (std::bad_alloc const&) {
		cout << "generateRandomMatrix memory allocation fail!" << endl;	exit(1);
	}

	for (ui=0;ui<numberOfRows;ui++) {
		matrixPFloat[ui]=(&(matrix1dFloat[ui*numberOfColumns]));
	}

	for (ui=0;ui<numberOfRows;ui++) {
		for (uj=0;uj<numberOfColumns;uj++) {
			matrixPFloat[ui][uj]=((float)(drand48())*60.0)-30.0;
		}
	}
	if (verbose) {
		cout << "MatrixP is:" << endl;
		for (ui=0;ui<numberOfRows;ui++) {
			for (uj=0;uj<numberOfColumns;uj++) {
				cout << matrixPFloat[ui][uj];
				if (uj<numberOfColumns-1) {
					cout << " ,";
				}
			}
			cout << endl;
		}
	}
}

int main(int argc, char *argv[]) {
	// Default values
	columns=10;
	rows=10;
	printPrecision=10;
	verbose=false;
	timing=false;
	cpu=true;
	gpu=true;
	seed=1234567;

	// ************************ Compute set up ************************
	// Compute the execution configuration (number of blocks for each row and column operation)
	blockSizeSinglePrecisionRow = 32;
	blockSizeSinglePrecisionColumn = 32;
	blockSizeDoublePrecisionRow = 32;
	blockSizeDoublePrecisionColumn = 32;

	// Set the default seed
	srand48(seed);

	parseArguments(argc, argv);

	if (gpu) {
		chooseCudaCard(verbose);
	}

	// =====================================================================================
	// Float version
	std::vector< float* >	matrixPFloat;
	std::vector< float >	matrix1dFloat;
	std::vector< float >	addedRowsCpuFloat;
	std::vector< float >	addedColumnsCpuFloat;

	generateRandomMatrix < float > (matrix1dFloat,matrixPFloat,rows,columns);

//	float totalAddedRowsFloatCpu,totalAddedColumnsFloatCpu;
	float totalAddedRowsVectorFloatCpu,totalAddedColumnsVectorFloatCpu;
	float totalAddedRowsFloatGpu,totalAddedColumnsFloatGpu;

	double timeAddRowsFloatCpu,timeAddColumnsFloatCpu;
	double timeAddRowsFloatGpu,timeAddColumnsFloatGpu;
	double timeAddRowsVectorFloatCpu,timeAddColumnsVectorFloatCpu;

	if (cpu) {
		struct timeval addRowsCpuStart, addRowsCpuEnd;
		gettimeofday(&addRowsCpuStart, NULL);
		addRowsCpu < float > (matrixPFloat,addedRowsCpuFloat);
		gettimeofday(&addRowsCpuEnd, NULL);
		timeAddRowsFloatCpu =	(addRowsCpuEnd.tv_sec + addRowsCpuEnd.tv_usec*0.000001) -
								(addRowsCpuStart.tv_sec + addRowsCpuStart.tv_usec*0.000001);

		struct timeval addColumnsCpuStart, addColumnsCpuEnd;
		gettimeofday(&addColumnsCpuStart, NULL);
		addColumnsCpu < float > (matrixPFloat,addedColumnsCpuFloat);
		gettimeofday(&addColumnsCpuEnd, NULL);
		timeAddColumnsFloatCpu =(addColumnsCpuEnd.tv_sec + addColumnsCpuEnd.tv_usec*0.000001) -
								(addColumnsCpuStart.tv_sec + addColumnsCpuStart.tv_usec*0.000001);

		struct timeval addRowsVectorCpuStart, addRowsVectorCpuEnd;
		gettimeofday(&addRowsVectorCpuStart, NULL);
		totalAddedRowsVectorFloatCpu = addVectorCpu < float > (addedRowsCpuFloat);
		gettimeofday(&addRowsVectorCpuEnd, NULL);
		timeAddRowsVectorFloatCpu = (addRowsVectorCpuEnd.tv_sec   + addRowsVectorCpuEnd.tv_usec*0.000001) -
									(addRowsVectorCpuStart.tv_sec + addRowsVectorCpuStart.tv_usec*0.000001);

		struct timeval addColumnsVectorCpuStart, addColumnsVectorCpuEnd;
		gettimeofday(&addColumnsVectorCpuStart, NULL);
		totalAddedColumnsVectorFloatCpu = addVectorCpu < float > (addedColumnsCpuFloat);
		gettimeofday(&addColumnsVectorCpuEnd, NULL);
		timeAddColumnsVectorFloatCpu = 	(addColumnsVectorCpuEnd.tv_sec 		+ addColumnsVectorCpuEnd.tv_usec*0.000001) -
										(addColumnsVectorCpuStart.tv_sec	+ addColumnsVectorCpuStart.tv_usec*0.000001);
	}


	// =====================================================================================
	// Double version
	std::vector< double* >	matrixPDouble;
	std::vector< double >	matrix1dDouble;
	std::vector< double >	addedRowsCpuDouble;
	std::vector< double >	addedColumnsCpuDouble;

	// Also seed to the same defaut seed value
	srand48(seed);

	generateRandomMatrix < double > (matrix1dDouble,matrixPDouble,rows,columns);

	double totalAddedRowsVectorDoubleCpu,totalAddedColumnsVectorDoubleCpu;
	double totalAddedRowsDoubleGpu,totalAddedColumnsDoubleGpu;

	double timeAddRowsDoubleCpu,timeAddColumnsDoubleCpu;
	double timeAddRowsDoubleGpu,timeAddColumnsDoubleGpu;
	double timeAddRowsVectorDoubleCpu,timeAddColumnsVectorDoubleCpu;

	if (cpu) {
		struct timeval addRowsCpuStart, addRowsCpuEnd;
		gettimeofday(&addRowsCpuStart, NULL);
		addRowsCpu < double > (matrixPDouble,addedRowsCpuDouble);
		gettimeofday(&addRowsCpuEnd, NULL);
		timeAddRowsDoubleCpu = 	(addRowsCpuEnd.tv_sec + addRowsCpuEnd.tv_usec*0.000001)-
								(addRowsCpuStart.tv_sec + addRowsCpuStart.tv_usec*0.000001);

		struct timeval addColumnsCpuStart, addColumnsCpuEnd;
		gettimeofday(&addColumnsCpuStart, NULL);
		addColumnsCpu < double > (matrixPDouble,addedColumnsCpuDouble);
		gettimeofday(&addColumnsCpuEnd, NULL);
		timeAddColumnsDoubleCpu =	(addColumnsCpuEnd.tv_sec + addColumnsCpuEnd.tv_usec*0.000001) -
									(addColumnsCpuStart.tv_sec + addColumnsCpuStart.tv_usec*0.000001);

		struct timeval addRowsVectorCpuStart, addRowsVectorCpuEnd;
		gettimeofday(&addRowsVectorCpuStart, NULL);
		totalAddedRowsVectorDoubleCpu = addVectorCpu < double > (addedRowsCpuDouble);
		gettimeofday(&addRowsVectorCpuEnd, NULL);
		timeAddRowsVectorDoubleCpu = (addRowsVectorCpuEnd.tv_sec	+ addRowsVectorCpuEnd.tv_usec*0.000001) -
									 (addRowsVectorCpuStart.tv_sec	+ addRowsVectorCpuStart.tv_usec*0.000001);

		struct timeval addColumnsVectorCpuStart, addColumnsVectorCpuEnd;
		gettimeofday(&addColumnsVectorCpuStart, NULL);
		totalAddedColumnsVectorDoubleCpu = addVectorCpu < double > (addedColumnsCpuDouble);
		gettimeofday(&addColumnsVectorCpuEnd, NULL);
		timeAddColumnsVectorDoubleCpu = (addColumnsVectorCpuEnd.tv_sec 		+ addColumnsVectorCpuEnd.tv_usec*0.000001) -
										(addColumnsVectorCpuStart.tv_sec	+ addColumnsVectorCpuStart.tv_usec*0.000001);
	}



	if (gpu) {
		cudaMatrixAddUp (	matrix1dFloat, matrix1dDouble,
							rows,columns,
							totalAddedRowsFloatGpu,totalAddedColumnsFloatGpu,totalAddedRowsDoubleGpu,totalAddedColumnsDoubleGpu,
							timeAddRowsFloatGpu,timeAddColumnsFloatGpu,timeAddRowsDoubleGpu,timeAddColumnsDoubleGpu,
							blockSizeSinglePrecisionRow,blockSizeSinglePrecisionColumn,blockSizeDoublePrecisionRow,blockSizeDoublePrecisionColumn,
							verbose,printPrecision);
	}

	if (cpu&&gpu) {
		cout << endl;
		if (verbose) {
	//		cout << "totalAddedRowsFloatCpu              = " << std::fixed << std::setprecision(printPrecision)  << totalAddedRowsFloatCpu <<
	//				"\t totalAddedRowsFloatGpu           = " << totalAddedRowsFloatGpu <<
	//				"\t difference=" << fabs(totalAddedRowsFloatCpu-totalAddedRowsFloatGpu) << endl;
	//		cout << "totalAddedColumnsFloatCpu           = " << std::fixed << std::setprecision(printPrecision)  << totalAddedColumnsFloatCpu <<
	//				"\t totalAddedColumnsFloatGpu        = " << totalAddedColumnsFloatGpu <<
	//				"\t difference=" << fabs(totalAddedColumnsFloatCpu-totalAddedColumnsFloatGpu) << endl;
			cout << "totalAddedRowsVectorFloatCpu        = " << std::fixed << std::setprecision(printPrecision)  << totalAddedRowsVectorFloatCpu <<
					"\t totalAddedRowsFloatGpu           = " << totalAddedRowsFloatGpu <<
					"\t difference=" << fabs(totalAddedRowsVectorFloatCpu-totalAddedRowsFloatGpu)<< endl;
			cout << "totalAddedColumnsVectorFloatCpu     = " << std::fixed << std::setprecision(printPrecision)  << totalAddedColumnsVectorFloatCpu <<
					"\t totalAddedColumnsFloatGpu        = " << totalAddedColumnsFloatGpu <<
					"\t difference=" << fabs(totalAddedColumnsVectorFloatCpu-totalAddedColumnsFloatGpu)<< endl;
			cout << endl;
	//		cout << "totalAddedRowsDoubleCpu             = " << std::fixed << std::setprecision(printPrecision) << totalAddedRowsDoubleCpu <<
	//				"\t totalAddedRowsDoubleGpu          = " << totalAddedRowsDoubleGpu <<
	//				"\t difference=" << fabs(totalAddedRowsDoubleCpu-totalAddedRowsDoubleGpu) << endl;
	//		cout << "totalAddedColumnsDoubleCpu          = " << std::fixed << std::setprecision(printPrecision) << totalAddedColumnsDoubleCpu <<
	//				"\t totalAddedColumnsDoubleGpu       = " << totalAddedColumnsDoubleGpu <<
	//				"\t difference=" << fabs(totalAddedColumnsDoubleCpu-totalAddedColumnsDoubleGpu) << endl;
			cout << "totalAddedRowsVectorDoubleCpu       = " << std::fixed << std::setprecision(printPrecision)  << totalAddedRowsVectorDoubleCpu <<
					"\t totalAddedRowsDoubleGpu          = " << totalAddedRowsDoubleGpu <<
					"\t difference=" << fabs(totalAddedRowsVectorDoubleCpu-totalAddedRowsDoubleGpu) << endl;
			cout << "totalAddedColumnsVectorDoubleCpu    = " << std::fixed << std::setprecision(printPrecision)  << totalAddedColumnsVectorDoubleCpu <<
					"\t totalAddedColumnsDoubleGpu       = " << totalAddedColumnsDoubleGpu <<
					"\t difference=" << fabs(totalAddedColumnsVectorDoubleCpu-totalAddedColumnsDoubleGpu) << endl;
			cout << endl;
		}

		if (timing) {
			cout << "Block size for single precision row operations: " << blockSizeSinglePrecisionRow << endl;
			cout << "addRowsFloatCpu took       :    " << timeAddRowsFloatCpu << " seconds, " <<
					"addRowsFloatGpu took       :    " << timeAddRowsFloatGpu << " seconds, speedup was:" << timeAddRowsFloatCpu/timeAddRowsFloatGpu << endl;
			cout << "Block size for double precision row operations: " << blockSizeSinglePrecisionColumn << endl;
			cout << "addColumnsFloatCpu took    :    " << timeAddColumnsFloatCpu << " seconds, " <<
					"addColumnsFloatGpu took    :    " << timeAddColumnsFloatGpu << " seconds, speedup was:" << timeAddColumnsFloatCpu/timeAddColumnsFloatGpu << endl;
			cout << "addRowsVectorFloatCpu took :    " << timeAddRowsVectorFloatCpu << " seconds, " <<
					"addRowsVectorFloatGpu took :    " << timeAddRowsFloatGpu << " seconds, speedup was:" << timeAddRowsVectorFloatCpu/timeAddRowsFloatGpu << endl;
			cout << endl;
			cout << "Block size for single precision column operations: " << blockSizeDoublePrecisionRow << endl;
			cout << "addRowsDoubleCpu took      :    " << timeAddRowsDoubleCpu << " seconds, " <<
					"addRowsDoubleGpu took      :    " << timeAddRowsDoubleGpu << " seconds, speedup was:" << timeAddRowsDoubleCpu/timeAddRowsDoubleGpu << endl;
			cout << "Block size for double precision column operations: " << blockSizeDoublePrecisionColumn << endl;
			cout << "addColumnsDoubleCpu took   :    " << timeAddColumnsDoubleCpu << " seconds, " <<
					"addColumnsDoubleGpu took   :    " << timeAddColumnsDoubleGpu << " seconds, speedup was:" << timeAddColumnsDoubleCpu/timeAddColumnsDoubleGpu << endl;
			cout << "addRowsVectorDoubleCpu took:    " << timeAddRowsVectorDoubleCpu << " seconds, " <<
					"addRowsVectorDoubleGpu took:    " << timeAddRowsDoubleGpu << " seconds, speedup was:" << timeAddRowsVectorDoubleCpu/timeAddRowsDoubleGpu << endl;
		}
	} else {
		if (cpu) {
//			cout << "totalAddedRowsDoubleCpu=" << std::fixed << std::setprecision(printPrecision) << totalAddedRowsDoubleCpu << endl;
//			cout << "totalAddedColumnsDoubleCpu=" << std::fixed << std::setprecision(printPrecision) << totalAddedColumnsDoubleCpu << endl;
			if (timing) {
				cout << "timeAddRowsDoubleCpu took:     " << timeAddRowsDoubleCpu << " seconds" << endl;
				cout << "timeAddColumnsDoubleCpu took:  " << timeAddColumnsDoubleCpu << " seconds" << endl;
			}
		}

		if (gpu) {
			cout << "totalAddedRowsDoubleGpu=" << std::fixed << std::setprecision(printPrecision) << totalAddedRowsDoubleGpu << endl;
			cout << "totalAddedColumnsDoubleGpu=" << std::fixed << std::setprecision(printPrecision) << totalAddedColumnsDoubleGpu << endl;

			if (timing) {
				cout << "timeAddRowsDoubleGpu took:    " << timeAddRowsDoubleGpu << " seconds" << endl;
				cout << "timeAddColumnsDoubleGpu took: " << timeAddColumnsDoubleGpu << " seconds" << endl;
			}
		}
	}

	matrixPFloat.clear();
	matrix1dFloat.clear();
	addedRowsCpuFloat.clear();
	addedColumnsCpuFloat.clear();

	matrixPDouble.clear();
	matrix1dDouble.clear();
	addedRowsCpuDouble.clear();
	addedColumnsCpuDouble.clear();

	return 0;
}





int parseArguments (int argc, char *argv[]) {
	int c;

	while ((c = getopt (argc, argv, "hcgn:m:p:rtx:X:y:Y:v")) != -1) {
		switch(c) {
			case 'c':
				cpu=false; break;	 //Skip the CPU test
			case 'g':
				gpu=false; break;	 //Skip the GPU test
			case 'h':
				printUsage(); break;
			case 'n':
				rows = atoi(optarg); break;
			case 'm':
				columns = atoi(optarg); break;
			case 'p':
				printPrecision = atoi(optarg); break;
			case 'r':
				struct timeval myRandom;
				gettimeofday(&myRandom, NULL);
				seed = (long int)(myRandom.tv_usec);
				break;
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
			case 'x':
				blockSizeSinglePrecisionRow = atoi(optarg);
				break;
			case 'X':
				blockSizeSinglePrecisionColumn = atoi(optarg);
				break;
			case 'y':
				blockSizeDoublePrecisionRow = atoi(optarg);
				break;
			case 'Y':
				blockSizeDoublePrecisionColumn = atoi(optarg);
				break;
			default:
				fprintf(stderr, "Invalid option given\n");
				return -1;
		}
	}
	return 0;
}

void printUsage () {
	cout << "matrixAddUp program" << endl;
	cout << "by: Jose Mauricio Refojo <jose@tchpc.tcd.ie>" << endl;
	cout << "This program will create a matrix of the specified sizes and compute the added rows and columns" << endl;
	cout << "usage:" << endl;
	cout << "matrixAddUp.out [options]" << endl;
	cout << "      -c           : will skip the CPU test" << endl;
	cout << "      -g           : will skip the GPU test" << endl;
	cout << "      -h           : will show this usage" << endl;
	cout << "      -n   size    : will set the number of rows to size (default 10)" << endl;
	cout << "      -m   size    : will set the number of columns to size (default 10)" << endl;
	cout << "      -p   size    : will set the print out precision to size (default 10)" << endl;
	cout << "      -r           : will set a random seed to the random number generator" << endl;
	cout << "      -t           : will output the amount of time that it took to generate each addUp" << endl;
	cout << "      -x   size    : will set the Block size for single precision row operations to size (default 10)" << endl;
	cout << "      -X   size    : will set the Block size for double precision row operations to size (default 10)" << endl;
	cout << "      -y   size    : will set the Block size for single precision column operations to size (default 10)" << endl;
	cout << "      -Y   size    : will set the Block size for double precision column operations to size (default 10)" << endl;
	cout << "      -v           : will activate the verbose mode" << endl;
	cout << "     " << endl;
}
