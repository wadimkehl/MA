

#include <vector>

#define NR_SCRIBBLES 0
#define NR_LABELS 1
#define NX 2
#define NY 3
#define WAVELET_STEPS 5
#define WAVELET_WIN_SIZE 6
#define NR_SEG_IT 7
#define TEX_DIM 8
#define INT_PARAMS 9

#define ALPHA 0
#define BETA 1
#define SIGMA 2
#define SEG_TAU 3
#define SEG_ETA 4
#define DELTA 5
#define FLOAT_PARAMS 6

#define KERNEL(k,x,y) k<<<x,y>>>
#define CUDA(x,y) {cudaError_t e = x; if(cudaSuccess != e) {printf("CUDAError: %s, Code: %i, String: %s\n",y,e,cudaGetErrorString(e));return false;}}
#define CUDA_2(x,y) {cudaError_t e = x; if(cudaSuccess != e) {printf("CUDAError: %s, Code: %i, String: %s\n",y,e,cudaGetErrorString(e));}}


using namespace std;

struct GPU_DATA
{
    int* scribbles; // x,y,label

    float *colors; // x*y*3

    float *textures; // x*y*tex_dimensions

    int *label_count; // label
    float *likely; // x*y*nr_labels


    float *primal, *temp;   //x*y*nr_labels
    float *g;               //x*y
    float *dual;            //x*y*nr_labels*2
    bool stepwise, ani;

    float *hh; // x*y*wavelet_steps
    float *lh; // x*y*wavelet_steps
    float *hl; // x*y*wavelet_steps

    float *hh_avg; // x*y*wavelet_steps
    float *hl_avg; // x*y*wavelet_steps
    float *lh_avg; // x*y*wavelet_steps

    float *hh_stddev; // x*y*wavelet_steps
    float *hl_stddev; // x*y*wavelet_steps
    float *lh_stddev; // x*y*wavelet_steps

    int *int_params;
    float *float_params;

    size_t lt_p,pd_p,g_p,col_p, tex_p, wave_p;

};


bool gpu_density(GPU_DATA data);
bool gpu_density(GPU_DATA data, GPU_DATA const_gpu);


bool gpu_wavelet(GPU_DATA data);
bool gpu_wavelet(GPU_DATA data, GPU_DATA const_gpu);

bool gpu_segmentation(GPU_DATA data, int &currIt);
bool gpu_segmentation(GPU_DATA data, GPU_DATA const_gpu, int &currIt);



void resetCuda();
