#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <time.h>
#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <float.h>
#include <vector>
#include <unordered_map>
#include <math.h>

#include "cuda_profiler_api.h"
#include "mSTOMP.h"


using std::vector;
using std::unordered_map;
using std::make_pair;


// These parameters must be tuned for a specific architecture

static const unsigned int WORK_SIZE = 512;
static const unsigned int AMT_UNROLL = 4;
static const unsigned int TILE_HEIGHT_ADJUSTMENT = 4;

// Volta (V100)
//static const unsigned int AMT_UNROLL = 2;
//static const unsigned int TILE_HEIGHT_ADJUSTMENT = 4;

//Pascal (P100)
//static const unsigned int AMT_UNROLL = 16;
//static const unsigned int TILE_HEIGHT_ADJUSTMENT = 2;

// Kepler (K80/K40/K20)
// on Kepler, these parameters do not affect the runtime as much because the bottleneck
// is elsewhere
//static const unsigned int AMT_UNROLL = 4;
//static const unsigned int TILE_HEIGHT_ADJUSTMENT = 4;

//This macro checks return value of the CUDA runtime call and exits
//the application if the call failed.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//This kernel computes a sliding mean with specified window size and a corresponding prefix sum array (A)
template<class DTYPE>
__global__ void sliding_mean(DTYPE* pref_sum,  size_t window, size_t size, DTYPE* means)
{
    const DTYPE coeff = 1.0 / (DTYPE) window;
    size_t a = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.x * blockDim.x + threadIdx.x + window;

    if (a == 0)
        means[a] = pref_sum[window - 1] * coeff;

	if (a < size-1)
        means[a + 1] = (pref_sum[b] - pref_sum[a]) * coeff;
    }

// This kernel computes the reciprical sliding standard deviation with specified window size, the corresponding means of each element, and the prefix squared sum at each element
// We actually compute the multiplicative inverse of the standard deviation, as this saves us from needing to do a division in the main kernel
template<class DTYPE>
__global__ void sliding_std(DTYPE* cumsumsqr, unsigned int window, unsigned int size, DTYPE* means, DTYPE* stds) 
{
    const DTYPE coeff = 1 / (DTYPE) window;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x * blockDim.x + threadIdx.x + window;
    DTYPE sq;
    if (a == 0) {
        sq = ((cumsumsqr[window - 1] * coeff) - (means[a] * means[a]));
    }
    else if (b < size + window) {
        sq = (((cumsumsqr[b - 1] - cumsumsqr[a - 1]) * coeff) - (means[a] * means[a]));
    }
    if (sq<EPSILON)
		sq = EPSILON;
	stds[a] = 1 / sqrt(sq); 
}

template<class DTYPE>
void compute_statistics(const DTYPE *T, DTYPE *means, DTYPE *stds, size_t n, size_t m, cudaStream_t s)
{
    square<DTYPE> sqr;
    dim3 grid(ceil(n / (double) WORK_SIZE), 1,1);
    dim3 block(WORK_SIZE, 1, 1);
    
    DTYPE *scratch;
    cudaMalloc(&scratch, sizeof(DTYPE) * n);
    gpuErrchk(cudaPeekAtLastError());
    
    thrust::device_ptr<const DTYPE> dev_ptr_T = thrust::device_pointer_cast(T);
    thrust::device_ptr<DTYPE> dev_ptr_scratch = thrust::device_pointer_cast(scratch);


    // Compute prefix sum in scratch
    thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum to compute sliding mean
    sliding_mean<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means);
    gpuErrchk(cudaPeekAtLastError());
    // Compute prefix sum of squares in scratch
    thrust::transform_inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, sqr,thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum of squares to compute the sliding standard deviation
    sliding_std<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means, stds);
    gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize(s);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(scratch);
    gpuErrchk(cudaPeekAtLastError());
}

template<class DTYPE>
__global__ void elementwise_multiply_inplace(const DTYPE* A, DTYPE *B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] *= A[tid];
    }
} 

template<>
__global__ void elementwise_multiply_inplace(const cuDoubleComplex* A, cuDoubleComplex* B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] = cuCmul(A[tid], B[tid]);
    }
}

template<>
__global__ void elementwise_multiply_inplace(const cuFloatComplex* A, cuFloatComplex* B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
       B[tid] = cuCmulf(A[tid], B[tid]);
    }
}


// A is input unaligned sliding dot products produced by ifft
// out is the computed vector of distances
template<class DTYPE>
__global__ void normalized_aligned_dot_products(const DTYPE* A, const DTYPE divisor,
                                                const unsigned int m, const unsigned int n,
                                                DTYPE* QT)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) {
        QT[a] = A[a + m - 1] / divisor;
    }
}

template<class DTYPE>
__global__ void populate_reverse_pad(const DTYPE *Q, DTYPE *Q_reverse_pad, const int window_size, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < window_size) {
        Q_reverse_pad[tid] = Q[window_size - 1 - tid];
    }else if(tid < size){ 
        Q_reverse_pad[tid] = 0;
    }
}

template<class DTYPE, class CUFFT_DTYPE>
void sliding_dot_products_and_distance_profile(DTYPE* T, DTYPE* Q, DTYPE *QT, const int size, const int window_len, cudaStream_t s)
{        
    const int n = size - window_len + 1;
    const int cufft_data_size = size / 2 + 1;
    dim3 grid(ceil(n / (double) WORK_SIZE), 1, 1);
    dim3 block(WORK_SIZE, 1, 1);

    cufftHandle fft_plan, ifft_plan;    
    DTYPE *Q_reverse_pad;
    CUFFT_DTYPE *Tc, *Qc;
    cufftPlan1d(&fft_plan, size, CUFFT_FORWARD_PLAN, 1);
    cufftPlan1d(&ifft_plan, size, CUFFT_REVERSE_PLAN, 1);
    cufftSetStream(fft_plan, s);
    cufftSetStream(ifft_plan,s);
    cudaMalloc(&Q_reverse_pad, sizeof(DTYPE) * size);
    cudaMalloc(&Tc, sizeof(CUFFT_DTYPE) * cufft_data_size);
    cudaMalloc(&Qc, sizeof(CUFFT_DTYPE) * cufft_data_size);
    
    // Compute the FFT of the time series
    CUFFT_FORWARD__(fft_plan, T, Tc);
    gpuErrchk(cudaPeekAtLastError());

    // Reverse and zero pad the query
    populate_reverse_pad<DTYPE><<<dim3(ceil(size / (double) WORK_SIZE),1,1), block, 0, s>>>(Q, Q_reverse_pad, window_len, size);
    gpuErrchk(cudaPeekAtLastError());
    
    // Compute the FFT of the query
    CUFFT_FORWARD__(fft_plan, Q_reverse_pad, Qc);
    gpuErrchk(cudaPeekAtLastError());
    
    elementwise_multiply_inplace<<<dim3(ceil(cufft_data_size / (double) WORK_SIZE), 1, 1), block, 0, s>>>(Tc, Qc, cufft_data_size);
    gpuErrchk(cudaPeekAtLastError());

    // Compute the ifft
    // Use the space for the query as scratch space as we no longer need it
    CUFFT_REVERSE__(ifft_plan, Qc, Q_reverse_pad);
    gpuErrchk(cudaPeekAtLastError());
    
    normalized_aligned_dot_products<DTYPE><<<grid, block, 0, s>>>(Q_reverse_pad, size, window_len, n, QT);
    gpuErrchk(cudaPeekAtLastError());
    
    cudaFree(Q_reverse_pad);
    cudaFree(Tc);
    cudaFree(Qc);
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
} 





//Atomically updates the MP/idxs using a single 64-bit integer. We lose a small amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a critical section and dedicated locks.
__device__ inline void MPatomicMax(volatile unsigned long long int* __restrict__ address, float val, unsigned int idx)
{
    mp_entry loc, loctest;
    loc.floats[0] = val;
    loc.ints[1] = idx;
    loctest.ulong = *address;
    while (loctest.floats[0] < val){
        loctest.ulong = atomicCAS((unsigned long long int*) address, loctest.ulong,  loc.ulong);
    }
}

template<class DTYPE, unsigned int BLOCKSZ, unsigned int tile_height>
__device__ inline void initialize_tile_memory(const double* __restrict__ T, const double* __restrict__ means, const double* __restrict__ inv_stds,
                                              DTYPE* __restrict__ A_low, DTYPE* __restrict__ A_high, DTYPE* __restrict__ B_low,
                                              DTYPE* __restrict__ B_high, DTYPE* __restrict__ mean_x, DTYPE* __restrict__ mean_y,
                                              DTYPE* __restrict__ inv_std_x, DTYPE* __restrict__ inv_std_y, const unsigned int n,
                                              const unsigned int m, const unsigned int x, const unsigned int y)
{
    // Update the other cached values to reflect the upcoming tile
    if (x <  n + m - 1) {
        A_low[threadIdx.x] = T[x];
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ < n + m - 1) {
        A_low[threadIdx.x + BLOCKSZ] = T[x + BLOCKSZ];
    }
    
    if (x + m < n + m - 1) {
        A_high[threadIdx.x] = T[x + m];
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ + m < n + m - 1) {
        A_high[threadIdx.x + BLOCKSZ] = T[x + BLOCKSZ + m];
    }
    if (threadIdx.x < tile_height && y + threadIdx.x < n + m - 1) {
        B_low[threadIdx.x] = T[y + threadIdx.x];
    }
    if (threadIdx.x < tile_height && y + threadIdx.x + m < n + m - 1) {
        B_high[threadIdx.x] = T[y + threadIdx.x + m];
    }
    if (x < n) {
        inv_std_x[threadIdx.x] = inv_stds[x];
        // We precompute part of the distance calculation in the mean_x variable
        // This saves us a multiply in the main loop
        mean_x[threadIdx.x] = means[x] * m;
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ < n) {
        inv_std_x[threadIdx.x + BLOCKSZ] = inv_stds[x + BLOCKSZ];
        // We precompute part of the distance calculation in the mean_x variable
        // This saves us a multiply in the main loop
        mean_x[threadIdx.x + BLOCKSZ] = means[x + BLOCKSZ] * m;
    }
    if (threadIdx.x < tile_height && y + threadIdx.x < n) {
        inv_std_y[threadIdx.x] = inv_stds[y + threadIdx.x];
        mean_y[threadIdx.x] = means[y + threadIdx.x];
    }
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
template<typename DTYPE, unsigned int BLOCKSZ, unsigned int UNROLL_COUNT>
__global__ void WavefrontUpdateSelfJoin(const double* __restrict__ QT, //[d]
										  const double* __restrict__ T, //[d]
										  const double* __restrict__ inv_stds, //[d]
										  const double* __restrict__ means, //[d]
										  unsigned long long int* __restrict__ profile,  //[kmax]   //kmax<=d
										  const unsigned int m, 
										  const unsigned int n, 
										  const int d,
										  const int kmax, //top k dimensions
										  int startPos, 
										  int numDevices)
{
    // Factor and threads per block must both be powers of two where: factor <= threads per block
    // UNROLL_COUNT * factor must also evenly divide WORK_SIZE
    // 'factor' is a scaling factor for the tile size, due to shared memory considerations
    // we cannot do a full tile at once, we must chop it into pieces
    // The values that are set here should give good performance already
    // but may be fine tuned for your specific Nvidia architecture
    const int tile_height = BLOCKSZ / TILE_HEIGHT_ADJUSTMENT;
    const int tile_width = tile_height + BLOCKSZ;
	
    __shared__ DTYPE A_low[tile_width];
    __shared__ DTYPE A_high[tile_width];
    __shared__ DTYPE inv_std_x[tile_width];
    __shared__ DTYPE inv_std_y[tile_height];
    __shared__ DTYPE mean_x[tile_width];
    __shared__ DTYPE mean_y[tile_height];
    __shared__ DTYPE B_high[tile_height];
    __shared__ DTYPE B_low[tile_height];

	const int dstride = ((d+15)/16)*16;
	DTYPE dist[tile_height*MAX_DIM];
	double qt_curr[MAX_DIM];

    // This is the index of the meta-diagonal that this thread block will work on
    int meta_diagonal_idx = blockIdx.x * numDevices + startPos;

    // The first threads are acutally computing the trivial match between the same subsequence
    // we exclude these from the calculation
    const int exclusion = (m / 2);
    int tile_start_x = meta_diagonal_idx * BLOCKSZ + exclusion;
    int tile_start_y = 0;
    
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // localX, localY are the local coordinates of the thread position in the tile it is working on
    int x = tile_start_x + threadIdx.x;
    int y = 0;
    int localX, localY;
	int dim = 0;

    // Load the first dot product value
	for(int i=0; i<d; ++i)
		qt_curr[i] = QT[n*i+x]; //load initial qt_curr for each dimension!

    /////////////////////////////////////    
    // Main loop
    /////////////////////////////////////
    // Each threadblock finds all the distances on a 'metadiagonal'
    // We use a tiled approach for each thread block
    // The tiles are horizontal slices of the diagonal, think of a parallelogram cut
    // from a diagonal slice of the distance matrix 
    // Each thread starts on the first row and works its way down-right towards right
    // side of the distance matrix
    while (tile_start_x < n)
    {
		const int offset = n*dim;
		
		// Necessary to sync results
		__syncthreads(); 
        
        // Initialize the next tile's shared memory
        initialize_tile_memory<DTYPE, BLOCKSZ, tile_height>(T+(n+m-1)*dim, means+offset, inv_stds+offset, 
                                                A_low, A_high, B_low, B_high, mean_x, mean_y, inv_std_x,
                                                inv_std_y, n, m, x, y);

        // Reset the tile local positions
        localY = 0;
        localX = threadIdx.x;

        // Start of new tile, sync shared mem
        __syncthreads();

        // Process the tile:
        // Each iteration generates the next UNROLL_COUNT distances
        // This loop is partially unrolled to improve instruction level parallelism
        // In all but the last tile in each metadiagonal, this first loop will compute
        // the entire tile, at the end we will have some leftover (UNROLL_COUNT may
        // not cleanly divide x) which is handled by the second loop
        
        //  { UNrolled loop removed for simplicity }

        
        // Finish the remaining iterations of the final tile if there were leftover
        // NOTE: this loop should only execute once for each thread beacuse we restrict
        // UNROLL_COUNT to be a factor of tile_height
        while (x < n && localY < tile_height) {
			const DTYPE dst = (static_cast<DTYPE>(qt_curr[dim]) - (mean_x[localX] * mean_y[localY])) * inv_std_x[localX] * inv_std_y[localY];
			//online in-place insertion sort (see test/insrtest.cc for tests)
			int j = localY*dstride+min(dim,kmax)-1;
 			if (dim<kmax || dst>dist[j]) {
				const int jmin = localY*dstride;
				for(; j>=jmin && dst>dist[j]; --j)
					dist[j+1] = dist[j];
    			dist[j+1] = dst; 
			}
			//TODO improve sort? probably not worth it
	        qt_curr[dim] = qt_curr[dim] + A_high[localX] * B_high[localY] - A_low[localX] * B_low[localY];
		
            x++;
            y++;
            localX++;
            localY++;
        }

		/////////// 

		// If dim<d, roll back x and y to do the same tile on the next dimension
		if (++dim<d) {
			x -= localY;
			y -= localY;
	        // Make sure our updates were committed before we pull in the next tile
	        //we now do syncthreads at the beginning of the loop anyway--- __threadfence_block(); 
			continue;
		}
		dim = 0;


		// do this when we have dists for all sequences and switch to the next tile 
		int x2=x, y2=y;
		while(--localY>=0) {
			--localX;
			--y2;
			--x2;
			DTYPE dsum=0.;
			for(int k=0; k<kmax && k<d; ++k) {
				// avg dist[localY][k]
 				dsum += dist[localY*dstride+k];
				const DTYPE davg = dsum/(k+1);
				// store in profile[k][xyetc] ---v
				MPatomicMax(profile + n*k + tile_start_x + localX, davg, y2);
				MPatomicMax(profile + n*k + tile_start_y + localY, davg, x2);
        }
		} /**/

		//Old, 1 dimension. Useful for testing (each profile is not top-k but simply one corresponding to each dim) -remember to disable sorting above for this^. 
		/*int x2=x, y2=y;
		while(--localY>=0) {
			--localX;
			--y2;
			--x2;
			for(int k=0; k<kmax; ++k) {
				MPatomicMax(profile + n*k + tile_start_x + localX, dist[localY*dstride+k], y2);
				MPatomicMax(profile + n*k + tile_start_y + localY, dist[localY*dstride+k], x2);
			}
		} /**/

        // Update the tile position
        tile_start_x += tile_height;
        tile_start_y += tile_height;

        // Make sure our updates were committed before we pull in the next tile
        //we now do syncthreads at the beginning of the loop anyway--- __threadfence_block(); 
    }
    
}

__global__ void cross_correlation_to_ed(float *profile, unsigned int n, unsigned int m) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        profile[tid] = (float)sqrt(max(2.*((double)m - (double)profile[tid]), 0.0));
    }
}

// The mSTOMP algorithm
template<class DTYPE, class CUFFT_DTYPE>
void do_mSTOMP(const vector<DTYPE> &T_h, vector<float> &profile_h, vector<unsigned int> &profile_idx_h, const unsigned int m, const vector<int> &devices, const int d, const int kmax) 
{
    if(devices.empty()) {
        printf("Error: no gpu provided\n");
        exit(0);
    }
	assert(kmax<=d);
    
    const size_t n = T_h.size()/d - m + 1;
    
	// global arrays allocated at each device x[ndevice]=ptr
    unordered_map<int, DTYPE*> T_dev, QT_dev, means, stds;
    unordered_map<int, unsigned long long int*> profile_merged;
    unordered_map<int, float*> profile_dev;
    unordered_map<int, unsigned int*> profile_idx_dev;
    unordered_map<int, cudaEvent_t> clocks_start, clocks_end;
    unordered_map<int, cudaStream_t> streams;

    // Allocate and initialize memory
    for (auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        T_dev.insert(make_pair(device, (DTYPE*) 0));
        QT_dev.insert(make_pair(device, (DTYPE*) 0));
        means.insert(make_pair(device, (DTYPE*) 0));
        stds.insert(make_pair(device, (DTYPE*) 0));
        profile_dev.insert(make_pair(device,(float*) NULL));
        profile_merged.insert(make_pair(device,(unsigned long long int*) NULL));
        profile_idx_dev.insert(make_pair(device,(unsigned int *) NULL));

        cudaMalloc(&T_dev.at(device),			T_h.size() * sizeof(DTYPE)); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_dev.at(device),		profile_h.size() * sizeof(float)); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_idx_dev.at(device), profile_idx_h.size() * sizeof(unsigned int)); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&QT_dev.at(device),			n * d * sizeof(DTYPE)); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&means.at(device),			n * d * sizeof(DTYPE)); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&stds.at(device),			n * d * sizeof(DTYPE)); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_merged.at(device),  n * kmax * sizeof(unsigned long long int)); gpuErrchk(cudaPeekAtLastError());
        cudaEvent_t st, ed;
        cudaEventCreate(&ed); gpuErrchk(cudaPeekAtLastError());
        cudaEventCreate(&st); gpuErrchk(cudaPeekAtLastError());
        clocks_start.emplace(device, st);
        clocks_end.emplace(device, ed);
        cudaStream_t s;
        cudaStreamCreate(&s);
        gpuErrchk(cudaPeekAtLastError());
        streams.emplace(device, s);
    }

    MPIDXCombine combiner;
	const int exclusion = m/2;
	int num_workers = ceil((n - exclusion) / (double) devices.size()); //was m/4
    
    // Asynchronously copy relevant data, precompute statistics, generate partial matrix profile
    int count = 0;
    for (auto &device : devices) {
        cudaSetDevice(device);
        cudaMemcpyAsync(T_dev[device], T_h.data(), sizeof(DTYPE) * T_h.size(), cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_dev[device], profile_h.data(), sizeof(float) * profile_h.size(), cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_idx_dev[device], profile_idx_h.data(), sizeof(unsigned int) * profile_idx_h.size(), cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());

		for(int idx=0; idx<d; ++idx) {
			const int nt = T_h.size()/d;
        // Computing the statistics for each device is overkill, but it avoids needing to do some staging on the host if P2P transfer doesn't work
			compute_statistics<DTYPE>(T_dev[device]+idx*nt, means[device]+idx*n, stds[device]+idx*n, n, m, streams.at(device));
			sliding_dot_products_and_distance_profile<DTYPE, CUFFT_DTYPE>(T_dev[device]+idx*nt, T_dev[device]+idx*nt, QT_dev[device]+idx*n, n, m, streams.at(device));
		}
        
        thrust::device_ptr<unsigned long long int> ptr = thrust::device_pointer_cast(profile_merged[device]);
        thrust::transform(
				thrust::cuda::par.on(streams.at(device)), 
				profile_dev[device], 
				profile_dev[device] + profile_h.size(), //was +n
				profile_idx_dev[device], 
				profile_merged[device], 
				combiner);
        printf("Start main kernel on GPU %d\n", device);
        cudaEventRecord(clocks_start[device], streams.at(device));

		//WORK_SIZE is defined as 512 (threads per block)
		WavefrontUpdateSelfJoin<MSTOMP_PRECISION, WORK_SIZE, AMT_UNROLL>
			<<< dim3(ceil(num_workers / (double) WORK_SIZE), 1, 1),
				dim3(WORK_SIZE, 1,1), 
				0, 
				streams.at(device)>>>
			(QT_dev[device], T_dev[device], stds[device], means[device], profile_merged[device], 
			 m, n, d, kmax, count, devices.size());
			 
        cudaEventRecord(clocks_end[device], streams.at(device));
        ++count;
    }
   
    float time;
    for(auto &device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamSynchronize(streams.at(device));
        cudaEventElapsedTime(&time, clocks_start[device], clocks_end[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventDestroy(clocks_start.at(device));
        cudaEventDestroy(clocks_end.at(device));
        printf("Device %d took %f seconds\n", device, time / 1000);
    }

    printf("Finished mSTOMP to generate partial matrix profile of size %lu on %lu devices:\n", n, devices.size());

    // Free unneeded resources
    for (auto &device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(T_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        // Keep the profile for the first device as a staging area for the final result
        if (device != devices.at(0)) { 
            cudaFree(profile_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_idx_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaFree(QT_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(means[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(stds[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamDestroy(streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
    }
   

    // Consolidate the partial matrix profiles to a single vector using the first gpu 
    printf("Merging partial matrix profiles into final result\n");
    vector<unsigned long long int> partial_profile_host(n*kmax);
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
    auto ptr_profile = thrust::device_ptr<float>(profile_dev[devices.at(0)]);
    auto ptr_index = thrust::device_ptr<unsigned int>(profile_idx_dev[devices.at(0)]);
    auto ptr_merged = thrust::device_ptr<unsigned long long int>(profile_merged[devices.at(0)]);
    auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile, ptr_index, ptr_merged));
    auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile + n*kmax, ptr_index + n*kmax, ptr_merged + n*kmax));
    for(int i = 0; i < devices.size(); ++i) {
        cudaSetDevice(devices.at(i));
        gpuErrchk(cudaPeekAtLastError());
        if (i != 0) {
			//TODO check this works ok with multiple gpus
            cudaMemcpy(partial_profile_host.data(), profile_merged[devices.at(i)], n*kmax * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_merged[devices.at(i)]);
            gpuErrchk(cudaPeekAtLastError());
            cudaSetDevice(devices.at(0));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpy(profile_merged[0], partial_profile_host.data(), n*kmax * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
            gpuErrchk(cudaPeekAtLastError());
        }
        thrust::for_each(iter_begin, iter_end, max_with_index());
        gpuErrchk(cudaPeekAtLastError());
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
         
    // Compute the final distance calculation to convert cross correlation computed earlier into euclidean distance
	for(int i=0; i<kmax; ++i)
		cross_correlation_to_ed<<<dim3(ceil(n / (double) WORK_SIZE), 1, 1), dim3(WORK_SIZE, 1, 1)>>>(profile_dev[devices.at(0)]+i*n, n, m); 

    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(profile_idx_h.data(), profile_idx_dev[devices.at(0)], sizeof(unsigned int) * n*kmax, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(profile_h.data(), profile_dev[devices.at(0)], sizeof(float) * n*kmax, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_idx_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_merged[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());

}

//Reads input time series from file
template<class DTYPE>
void readFile(const char* filename, vector<DTYPE>& v, const char *format_str) 
{
    FILE* f = fopen( filename, "r");
    if (f==NULL) {
		printf("Unable to open %s for reading, please make sure it exists\n", filename);
		exit(0);
    }
    DTYPE num;
    while(!feof(f)) {
		if (fscanf(f, format_str, &num)==1)
			v.push_back(num);
	}
    //v.pop_back();
    fclose(f);
}


// Main program for reading from flat text file
int main(int argc, char** argv) 
{
    if (argc<7) {
		printf("Usage: mSTOMP <dimensions> <window_len> <flat input file> <top k dim> <profile output file> <index output file> [Optional: list of GPU device numbers to run on]\n");
        exit(0);
    }

    const int window_size = atoi(argv[2]);
	const int d = atoi(argv[1]);
	const int kmax = atoi(argv[4]);

	vector<double> T_h;
    readFile<double>(argv[3], T_h, "%lf");

    const int n = T_h.size()/d - window_size + 1;
    vector<float> profile(n*kmax, CC_MIN);
    vector<unsigned int> profile_idx(n*kmax, 0);

	if (d>MAX_DIM) {
		printf("d must be <= MAX_DIM (%d)\n", MAX_DIM);
		exit(0);
	}
	if (kmax>d) {
		printf("kmax must be <=d (%d)\n", d);
		exit(0);
	}

    cudaFree(0);
    
    vector<int> devices;
    if (argc==7) {
        // Use all available devices 
        int num_dev;
        cudaGetDeviceCount(&num_dev);
        for(int i = 0; i < num_dev; ++i){ 
            devices.push_back(i);
        }
    } else {
        // Use the devices specified
        int x = 7;
        while (x < argc) {
            devices.push_back(atoi(argv[x]));
            ++x;
        }
    }
    
    printf("Starting mSTOMP\n");
     
    do_mSTOMP<double,__CUFFT_TYPE__>(T_h, profile, profile_idx, window_size, devices, d, kmax);

    printf("Now writing result to files: ");
    FILE* f1 = fopen(argv[5], "w");
    FILE* f2 = fopen(argv[6], "w");
	const int psz = profile.size()/kmax;
	printf("%d top k profiles (%d values each)\n", kmax, psz);
	if (f1==NULL || f2==NULL) {
		printf("ERROR: couldn't open output files\n");
		exit(0);
	}
    for(int i=0; i<psz; ++i) {
		int j;
		for (j=0; j<kmax-1; ++j) {
			fprintf(f1, "%f,", profile[j*psz + i]);
			fprintf(f2, "%u,", profile_idx[j*psz + i]+1);
		}
		fprintf(f1, "%f\n", profile[j*psz + i]);
		fprintf(f2, "%u\n", profile_idx[j*psz + i]+1);
    }
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());
    fclose(f1);
    fclose(f2);

    printf("Done\n");
    return 0;
}
