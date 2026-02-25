///// Created by Jose Mauricio Refojo - 2018-01-23		Last changed: 2026-02-16
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

//#ifdef __cplusplus
//extern "C" {
	//int cudaMatrixAddUp (std::vector< std::vector< float > > &);
	int		add_vectors				(void);

	int		cudaMatrixAddUp		(	std::vector< float >  &matrixFloat1d,std::vector< double >  &matrixDouble1d,
									int rows,int columns,
									float &totalRowsFloat,float &totalColumnsFloat,double &totalRowsDouble,double &totalColumnsDouble,
									double &timeAddRowsFloatGpu,double &timeAddColumnsFloatGpu,double &timeAddRowsDoubleGpu,double &timeAddColumnsDoubleGpu,
									int &blockSizeSinglePrecisionRow,int &blockSizeSinglePrecisionColumn,int &blockSizeDoublePrecisionRow,int &blockSizeDoublePrecisionColumn,
									bool verbose,unsigned int printPrecision=4);

	void	cudaLastErrorCheck		(const char *message);
	inline int		chooseCudaCard			(bool verbose=false);
//}
//#endif
