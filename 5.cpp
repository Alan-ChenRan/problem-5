#include<stdio.h>
#include<iostream>
#include<mpi.h>
#include <stdlib.h>
#include <ctime> 
#include<algorithm>
#include<math.h>
#include<assert.h>
#include<malloc.h>
//#include<sys/time.h>
#define N 100000
#define For(i,j,n) for(int i=j;i<=n;++i)
using namespace std;
int ProcRank, n, ProcNum;
int Size, I, J, K; // Size of matricies 
int BlockSize; // Sizes of matrix blocks on current process 
int GridSize;// Size of virtual processor grid 

int* pAMatrix; // The first argument of matrix multiplication 
int* pBMatrix; // The second argument of matrix multiplication 
int* pCMatrix; // The result matrix 

int* pAblock; // Initial block of matrix A on current process 
int* pBblock; // Initial block of matrix B on current process 
int* pCblock, * pC; // Block of result matrix C on current process

int Na = 0, Nb = 0;
int pos(int x, int y) { return y + Size * x; }
int get_num(FILE * fd)
{
	char t[1];
	int sum = 0; 
	fread(t, 1, 1, fd);
	while (t[0] < '0' || t[0]>'9')fread(t, 1, 1, fd);
	while (t[0] >= '0' && t[0] <= '9')sum = sum * 10 + t[0] - '0', fread(t, 1, 1, fd);
	return sum;
}
void Reada(int * a, FILE* fa)
{
	For(i, 0, Size - 1)For(j, 0, Size - 1)a[pos(i, j)] = get_num(fa);
}

char buf[20];
int L = 0;
void Writebuf(int x)
{
	L = 0;
	if (x == 0) { buf[0] = '0'; buf[1] = '\0'; L = 1; return; }
	while (x > 0)
	{
		buf[L] = x % 10 + '0'; x /= 10; ++L;
	}
	For(i, 0, L / 2) { char t = buf[i]; buf[i] = buf[L - 1 - i]; buf[L - 1 - i] = t; }
	buf[L] = ' '; ++L;
	buf[L] = '\0';
}
void Printing(int* c, FILE * fc)
{
	For(i, 0, Size - 1)
	{
		For(j, 0, Size - 1)
		{
			Writebuf(c[pos(i, j)]);
			fwrite(buf, 1, L, fc);
		}
		fwrite("\n", 1, 1, fc);
	}
}
void init(FILE* fa, FILE* fb)
{
	pAMatrix = new int[n * n];
	pBMatrix = new int[n * n];
	pCMatrix = new int[n * n];
	Reada(pAMatrix, fa);
	Reada(pBMatrix, fb);
}
void ProcessInitialization()
{
	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	BlockSize = Size / GridSize;
	pAblock = new int[BlockSize * BlockSize];
	pBblock = new int[BlockSize * BlockSize];
	pCblock = new int[BlockSize * BlockSize];
	pC = new int[BlockSize * BlockSize];
	for (int i = 0; i < BlockSize * BlockSize; i++)pAblock[i] = 0, pBblock[i] = 0, pCblock[i] = 0;
}
void DataDistribution()
{
	int* pSendNum; // the number of elements sent to the process 
	int* pSendInd; // the index of the first data element sent to the process 
		// Alloc memory for temporary objects 
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];
	// Define the disposition of the matrix rows for current process 
	For(i, 0, GridSize * GridSize - 1)pSendNum[i] = BlockSize * BlockSize;
	For(i, GridSize * GridSize, ProcNum - 1)pSendNum[i] = 0;
	pSendInd[0] = 0;
	For(i, 1, ProcNum - 1)pSendInd[i] = pSendInd[i - 1] + pSendNum[i];
	int* pA = new int[Size * Size];
	int* pB = new int[Size * Size];
	if (ProcRank == 0)
	{
		For(I1, 0, GridSize - 1)For(J1, 0, GridSize - 1)
		{
			int K1 = I1 * GridSize + J1;
			int x = I1 * BlockSize, y = J1 * BlockSize;
			int S = pSendInd[K1];
			For(i, 0, BlockSize - 1)For(j, 0, BlockSize - 1)
			{
				pA[S + i * BlockSize + j] = pAMatrix[(x + i) * Size + y + j];
				pB[S + i * BlockSize + j] = pBMatrix[(x + i) * Size + y + j];
			}
		}
	}
	// Scatter the rows 
	MPI_Scatterv(pA, pSendNum, pSendInd, MPI_INT, pAblock, pSendNum[ProcRank], MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(pB, pSendNum, pSendInd, MPI_INT, pBblock, pSendNum[ProcRank], MPI_INT, 0, MPI_COMM_WORLD);
	// Free the memory 
	delete[] pSendNum; delete[] pSendInd;
	delete[] pA; delete[] pB;
}
void A1()
{
	if (ProcRank < GridSize * GridSize)
	{
		int t = ProcRank + J * GridSize * GridSize;
		if (t != ProcRank)MPI_Send(pAblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD);
	}
	else
	{
		int t = I * GridSize + J;
		MPI_Recv(pAblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}
void A2()
{
	if (J == K)
	{
		For(j, 0, GridSize - 1)if (j != J)
		{
			int t = K * GridSize * GridSize + I * GridSize + j;
			MPI_Send(pAblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		int t = K * GridSize * GridSize + I * GridSize + K;
		MPI_Recv(pAblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}
void B1()
{
	if (ProcRank < GridSize * GridSize)
	{
		int t = ProcRank + I * GridSize * GridSize;
		if (t != ProcRank)MPI_Send(pBblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD);
	}
	else
	{
		int t = I * GridSize + J;
		MPI_Recv(pBblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}
void B2()
{
	if (I == K)
	{
		For(i, 0, GridSize - 1)if (i != I)
		{
			int t = K * GridSize * GridSize + i * GridSize + J;
			MPI_Send(pBblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		int t = K * GridSize * GridSize + K * GridSize + J;
		MPI_Recv(pBblock, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}
void Block_delivery()
{
	if (ProcRank < GridSize * GridSize || J == K)A1();
	MPI_Barrier(MPI_COMM_WORLD);
	A2();
	MPI_Barrier(MPI_COMM_WORLD);
	if (ProcRank < GridSize * GridSize || I == K)B1();
	MPI_Barrier(MPI_COMM_WORLD);
	B2();
	MPI_Barrier(MPI_COMM_WORLD);
}
void ParallelResultCalculation()
{
	For(i, 0, BlockSize - 1)For(k, 0, BlockSize - 1)For(j, 0, BlockSize - 1)
	{

		pCblock[i * BlockSize + j] = pCblock[i * BlockSize + j] +
			pAblock[i * BlockSize + k] * pBblock[k * BlockSize + j];
	}
}
void ProcessTermination()
{
	if (ProcRank == 0)
	{
		cout << "0---------" << endl;
		cout << "pAMatrix:-------" << endl;
		for (int i = 0; i < Size; ++i)
		{
			for (int j = 0; j < Size; ++j)cout << pAMatrix[i * Size + j] << " "; cout << endl;
		}
		cout << "pBMatrix:-------" << endl;
		for (int i = 0; i < Size; ++i)
		{
			for (int j = 0; j < Size; ++j)cout << pBMatrix[i * Size + j] << " "; cout << endl;
		}
		cout << "pCMatrix:-------" << endl;
		for (int i = 0; i < Size; ++i)
		{
			for (int j = 0; j < Size; ++j)cout << pCMatrix[i * Size + j] << " "; cout << endl;
		}
	}
}
void ResultCollection_0()
{
	if (ProcRank < GridSize * GridSize)
	{
		For(k, 1, GridSize - 1)
		{
			int t = k * GridSize * GridSize + I * GridSize + J;
			MPI_Recv(pC, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			For(i, 0, BlockSize * BlockSize - 1)pCblock[i] += pC[i];
		}
	}
	else
	{
		int t = I * GridSize + J;
		For(i, 0, BlockSize * BlockSize - 1)pC[i] = pCblock[i];
		MPI_Send(pC, BlockSize * BlockSize, MPI_INT, t, 0, MPI_COMM_WORLD);
	}
}
void ResultCollection()
{
	int* pReceiveNum; // Number of elements, that current process sends 
	int* pReceiveInd; /* Index of the first element from current process
	in result vector */
	//Alloc memory for temporary objects 
	pReceiveNum = new int[ProcNum];
	pReceiveInd = new int[ProcNum];
	//Define the disposition of the result vector block of current processor 
	For(i, 0, GridSize * GridSize - 1)pReceiveNum[i] = BlockSize * BlockSize;
	For(i, GridSize * GridSize, ProcNum - 1)pReceiveNum[i] = 0;
	pReceiveInd[0] = 0;
	For(i, 1, ProcNum - 1)pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i];
	//Gather the whole result vector on every processor 
	int* p = new int[Size * Size];
	
	MPI_Allgatherv(pCblock, pReceiveNum[ProcRank], MPI_INT, p, pReceiveNum, pReceiveInd
		, MPI_INT, MPI_COMM_WORLD);
	if (ProcRank == 0)
	{
		For(I1, 0, GridSize - 1)For(J1, 0, GridSize - 1)
		{
			int K1 = I1 * GridSize + J1;
			int x = I1 * BlockSize, y = J1 * BlockSize;
			int S = pReceiveInd[K1];
			For(i, 0, BlockSize - 1)For(j, 0, BlockSize - 1)
				pCMatrix[(x + i) * Size + y + j] = p[S + i * BlockSize + j];
		}
	}
	delete[] pReceiveNum;
	delete[] pReceiveInd;
	delete[] p;
}
void TestResult()
{
	if (ProcRank == 0)
	{
		cout << "Test:   =============" << endl;
		double* p = new double[Size * Size];
		For(i, 0, Size * Size - 1)p[i] = 0;
		For(i, 0, Size - 1)
		{

			For(j, 0, Size - 1)
			{
				double temp = 0;
				For(k, 0, Size - 1)temp += pAMatrix[i * Size + k] * pBMatrix[k * Size + j];
				p[i * Size + j] += temp;
			}
		}
		cout << "Test A*B=" << endl;
		for (int i = 0; i < Size; ++i)
		{
			for (int j = 0; j < Size; ++j)cout << p[i * Size + j] << " "; cout << endl;
		}
		
		delete[] p;
	}
}

int main(int argc, char** argv)
{
	FILE* fa=NULL, *fb = NULL, *fc = NULL;
	srand((unsigned)time(NULL));
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

	GridSize = (int)pow(ProcNum, 1.0 / 3);
	int t = ProcRank;
	K = t / (GridSize * GridSize);
	t %= (GridSize * GridSize);
	I = t / GridSize;
	t %= GridSize;
	J = t;

	if (ProcRank == 0)
	{
		fa= fopen(argv[1], "r");
		if (fa == NULL) { printf("file a error\n"); return 1; }
		fb = fopen(argv[2], "r");
		if (fb == NULL) { printf("file b error\n"); return 1; }
		fc = fopen(argv[3], "w");
		Na = get_num(fa);
		Nb = get_num(fb);
		n = max(Na, Nb); Size = n;
		init(fa, fb);
	}
	ProcessInitialization();
	DataDistribution();
	MPI_Barrier(MPI_COMM_WORLD);

	Block_delivery();
	MPI_Barrier(MPI_COMM_WORLD);
	ParallelResultCalculation();
	MPI_Barrier(MPI_COMM_WORLD);
	ResultCollection_0();
	ResultCollection();
	ProcessTermination();
	TestResult();
	if (ProcRank == 0)
	{
		Printing(pCMatrix, fc);
		fflush(fc); fclose(fa); fclose(fb); fclose(fc);
	}

	MPI_Finalize();
	return 0;
}








