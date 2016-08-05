

#include "gpu.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "sys/time.h"

using namespace std;

#define WIN_SIZE 0


#define RED(x)   ((x >> 16) & 0xff)
#define GREEN(x) ((x >> 8) & 0xff)
#define BLUE(x)  (x & 0xff)

#define SQRT_2PI 2.50662827f


__global__ void kernel_segmentation(float *likely,float *primal, float *dual, float *tmp, float *g, float tau, int nx, int ny, int nr_labels,
                                    size_t lt_p,size_t pd_p, size_t g_p, int method)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = y*pd_p + x, c = pd_p*ny, page=2*c, book=pd_p*ny*nr_labels;

    bool work = x < nx && y < ny;
    work = (x >= WIN_SIZE && x < nx-WIN_SIZE && y >= WIN_SIZE  && y < ny-WIN_SIZE);

    //primal holds dashed V_n+1
    //primal[book] holds normal primal variable V_n+1

    // 1st step
    if (work)
    {

        // Add tau*grad to dual and consider boundaries
        if(x<nx-1-WIN_SIZE)
            for(int l=0; l < nr_labels;l++)
                dual[l*page     +pos] += tau*(primal[l*c +pos  +1]-primal[l*c + pos]);

        if(y<ny-1-WIN_SIZE)
            for(int l=0; l < nr_labels;l++)
                dual[l*page + c +pos] += tau*(primal[l*c +pos +pd_p]-primal[l*c + pos]);

        if (method==0)
        {
            //Lellmann dual space
            float v=0;
            for(unsigned int i = 0; i < nr_labels; i++)
                v += dual[pos     + page*i]*dual[pos     + page*i]+dual[pos + c + page*i]*dual[pos + c + page*i];
            if (v>1) //if (v>g[y*g_p + x])
            {
                v = sqrt(v);
                for(unsigned int i = 0; i < nr_labels; i++)
                {
                    dual[pos     + page*i] /= v;
                    dual[pos + c + page*i] /= v;
                }
            }
        }
        else if (method==1)
        {
            // Zach dual space
            float v=0;
            for(unsigned int i = 0; i < nr_labels; i++)
                v += abs(dual[pos     + page*i])+abs(dual[pos + c + page*i]);
            if (v>1.0f)
                for(unsigned int i = 0; i < nr_labels; i++)
                {
                    dual[pos     + page*i] /= v;
                    dual[pos + c + page*i] /= v;
                }
        }

        else
        {

            //      Chambolle dual space
            // Do Dykstra projection
            float diffus=g[y*g_p + x];
            float diffus_sq = diffus*diffus;
            while(true)
            {
                float change = 0;
                for(unsigned int i1 = 0; i1 < nr_labels; i1++)
                    for(unsigned int i2 = i1; i2 < nr_labels; i2++)
                    {
                        //compute sum over all input images between i1 and i2 for each component x,y
                        float v1 = 0;
                        float v2 = 0;
                        for(unsigned int j = i1; j <= i2; j++)
                        {
                            v1 += dual[pos     + page*j];
                            v2 += dual[pos + c + page*j];
                        }
                        float mm = v1*v1+v2*v2;  //compute length of the summarized vector
                        if(mm > diffus_sq)
                        {
                            mm = sqrt(mm);
                            //remove the length above 1 (-> m - 1) in direction of normalized sum vector avg
                            float rest = (mm - diffus)/(i2 - i1 + 1);
                            float mod1 = rest * (v1 / mm);
                            float mod2 = rest * (v2 / mm);
                            for(unsigned int j = i1; j <= i2; j++)
                            {
                                dual[pos     + page*j] -= mod1;
                                dual[pos + c + page*j] -= mod2;
                                change += mod1*mod1+mod2*mod2;
                            }
                        }
                    }
                if (change < 0.01f) break;
            }

        }

    }

    __syncthreads();

    // 2nd and 3rd step
    if (work)
    {


        // Add tau*(div-likely) to primal and check boundaries
        for(int l=0; l < nr_labels;l++)
        {
            int off =l*c + pos;
            int tl_off  = l*lt_p*ny + y*lt_p+x;
            float div = 0;

            if (x>WIN_SIZE)
                div -= dual[l*page     + pos-1] ;
            if (x<nx-1-WIN_SIZE)
                div += dual[l*page     +pos];
            if (y>WIN_SIZE)
                div -= dual[l*page + c + pos-pd_p];
            if (y<ny-1-WIN_SIZE)
                div += dual[l*page + c +pos];

            tmp[tl_off] = primal[off];
            primal[off] += tau*(div-likely[tl_off]);

        }

        // Simplex projection
        bool finished=false;
        while(!finished)
        {
            finished=true;
            // Determine n (nonzero-dimension of vector) and the sum of the entries
            int n = 0;
            float sum=0.0f;
            for(int l=0; l < nr_labels;l++)
            {
                float value = primal[c*l + pos];
                if(value!=0) n++;
                sum += value;
            }


            // Do projection
            for(int l=0; l < nr_labels;l++)
            {
                float value = primal[c*l + pos];
                if(value!=0) value -= (sum-1.0f)/(float)n;
                if(value < 0.0f)
                {
                    value = 0.0f;
                    finished = false;
                }
                primal[c*l + pos] = value;
                primal[book + c*l + pos] = value;
            }
        }



        // Acceleration step
        for(int l=0; l < nr_labels;l++)
        {
            int off =l*c + pos;
            int tl_off  = l*lt_p*ny + y*lt_p+x;
            primal[off] =  2*primal[off] - tmp[tl_off];
        }


    }

    __syncthreads();


}

bool gpu_segmentation(GPU_DATA data, GPU_DATA const_gpu,int &currIt)
{

    int nx = data.int_params[NX];
    int ny = data.int_params[NY];
    int nr_labels = data.int_params[NR_LABELS];
    int nr_seg_it = data.int_params[NR_SEG_IT];
    float tau = data.float_params[SEG_TAU];

    float size = 16;
    dim3 blockSize(size, size);
    dim3 gridSize( (int)ceil(nx/size), (int)ceil(ny/size) );


    CUDA(cudaMemcpy2D((void*) const_gpu.likely,   const_gpu.lt_p,data.likely,nx*sizeof(float) , nx*sizeof(float), ny*nr_labels,cudaMemcpyHostToDevice),"MemCpyLikely");
    CUDA(cudaMemcpy2D((void*) const_gpu.g,        const_gpu.g_p,data.g,     nx*sizeof(float) , nx*sizeof(float), ny,cudaMemcpyHostToDevice),"MemCpyG");

    if(currIt==0)
    {
        CUDA(cudaMemset2D((void*)const_gpu.primal, const_gpu.pd_p,0, nx*sizeof(float), ny*nr_labels*2), "MemsetPrimal") ;
        CUDA(cudaMemset2D((void*)const_gpu.dual,   const_gpu.pd_p,0, nx*sizeof(float), ny*nr_labels*2), "MemsetDual") ;
    }
    else
    {
        CUDA(cudaMemcpy2D((void*) const_gpu.primal, const_gpu.pd_p,data.primal,nx*sizeof(float) , nx*sizeof(float), 2*ny*nr_labels,cudaMemcpyHostToDevice),"MemCpyPrimal");
        CUDA(cudaMemcpy2D((void*) const_gpu.dual,   const_gpu.pd_p,data.dual  ,nx*sizeof(float) , nx*sizeof(float), 2*ny*nr_labels,cudaMemcpyHostToDevice),"MemCpyDual");
    }


    int dual_space = 0;
    if(data.stepwise)
    {

        for(int i=0; i < nr_seg_it;i++,currIt++)
        {
            KERNEL (kernel_segmentation,gridSize, blockSize)
                    (const_gpu.likely, const_gpu.primal,const_gpu.dual,const_gpu.temp,const_gpu.g,tau,nx,ny,nr_labels,
                     const_gpu.lt_p/sizeof(float),const_gpu.pd_p/sizeof(float),const_gpu.g_p/sizeof(float),dual_space);
        }

    }
    else
    {

        int start_it = currIt;
        while(currIt++-start_it<1500 )
        {

            KERNEL (kernel_segmentation,gridSize, blockSize)
                    (const_gpu.likely, const_gpu.primal,const_gpu.dual,const_gpu.temp,const_gpu.g,tau,nx,ny,nr_labels,
                     const_gpu.lt_p/sizeof(float),const_gpu.pd_p/sizeof(float),const_gpu.g_p/sizeof(float),dual_space);
        }


    }

    CUDA(cudaThreadSynchronize(),"Syncsegmentation");
    CUDA(cudaMemcpy2D((void*) data.primal,  nx*sizeof(float),const_gpu.primal, const_gpu.pd_p, nx*sizeof(float), ny*nr_labels,cudaMemcpyDeviceToHost),"MemCpyPrimal");
    CUDA(cudaMemcpy2D((void*) data.dual  ,  nx*sizeof(float),const_gpu.dual,   const_gpu.pd_p, nx*sizeof(float), 2*ny*nr_labels,cudaMemcpyDeviceToHost),"MemCpyDual");


    return true;


}


__device__ float kernel_gauss(float x,float var)
{
    return expf(-0.5f*x*x/(var*var))/(var*SQRT_2PI);
}

__device__ float p2NormSq(int x1, int y1, int x2, int y2)
{
    int a = (x1-x2);
    int b = (y1-y2);
    return a*a+b*b;
}


__global__ void kernel_density(int *scribbles, float *colors, float *textures, float *params, float *likely,float *temp,
                               int nx, int ny, int nr_labels, int nr_scribbles,int tex_dim, int *label_count,
                               size_t like_p,size_t col_p,size_t tex_p)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int border = 0;
    if (x >= border && x < nx-border && y >= border  && y < ny-border) {

        float alpha = params[0];
        //float delta = params[1];
        float scale = 1.0f/max(nx,ny);
        int tex_page = tex_p*ny;
        int col_page = col_p*ny;

        // Find NN-scribble
        if(alpha>0)
        {
            float nnScribble_dist= nx*nx+ny*ny;
            for(int i=0; i < nr_scribbles; i++)
            {
                int sx = scribbles[i*3+0];
                int sy = scribbles[i*3+1];
                float dist = p2NormSq(x,y,sx,sy);
                if(dist < nnScribble_dist)
                    nnScribble_dist=dist;
            }
            if(nnScribble_dist < 1.f) nnScribble_dist=1.f;
            alpha*= sqrt(nnScribble_dist)*scale;
        }

        // Run through scribbles and estimate
        for(int i=0; i < nr_scribbles; i++)
        {
            float space=1.0f,color=1.0f,texture=1.0f;
            int sx = scribbles[i*3+0];
            int sy = scribbles[i*3+1];
            int slabel = scribbles[i*3+2];
            float sigma = params[2+slabel];
            float beta =  params[2+nr_labels+slabel];

            if(alpha>0)
            {
                float v1 = x-sx;
                float v2 = y-sy;
                float distance = v1*(v1*temp[i*3+0] + v2*temp[i*3+2]) +
                        v2*(v1*temp[i*3+2] + v2*temp[i*3+1]);
                space = kernel_gauss(sqrt(distance)*scale,alpha);
            }

            if(sigma>0)
            {
                float r = colors[y*col_p+x]                - colors[sy*col_p+sx];
                float g = colors[y*col_p+x + col_page]     - colors[sy*col_p+sx + col_page];
                float b = colors[y*col_p+x + 2*col_page]   - colors[sy*col_p+sx + 2*col_page];

                color = kernel_gauss(r,sigma)*kernel_gauss(g,sigma)*kernel_gauss(b,sigma);
            }

            if(beta>0)
            {
                /*
               float diff=0;
               for(int t = 0; t < tex_dim; t++)
               {
                   int bits = ((int)textures[t*tex_page+y*tex_p+x]) ^ ((int) textures[t*tex_page+sy*tex_p+sx]);
                   while (bits)
                     {
                         diff++;
                         bits &= bits - 1;
                    }
               }
                texture = kernel_gauss(diff,beta);
                */

                for(int t = 0; t < tex_dim; t++)
                {
                    float diff = textures[t*tex_page+y*tex_p+x]-textures[t*tex_page+sy*tex_p+sx];
                    texture *= kernel_gauss(diff,beta);
                }

            }

            likely[ny*like_p*slabel + y*like_p + x] += space*color*texture;

        }

        // Divide by scribble number
        for(int i=0; i < nr_labels;i++)
            likely[ny*like_p*i + y*like_p + x] /= ((float)label_count[i]);

    }

}


bool gpu_density(GPU_DATA data, GPU_DATA const_gpu)
{



    int nx = data.int_params[NX];
    int ny = data.int_params[NY];
    int nr_labels = data.int_params[NR_LABELS];
    int nr_scribbles = data.int_params[NR_SCRIBBLES];
    int tex_dim = data.int_params[TEX_DIM];
    float size = 16;
    dim3 blockSize(size, size);
    dim3 gridSize( (int)ceil(nx/size), (int)ceil(ny/size) );


    CUDA(cudaMemset2D((void*)const_gpu.likely, const_gpu.lt_p,0, nx*sizeof(float), ny*nr_labels), "MemsetLikely") ;

    CUDA(cudaMemcpy(const_gpu.scribbles, data.scribbles, nr_scribbles*sizeof(int)*3,cudaMemcpyHostToDevice),"MemCopyScribbles");
    CUDA(cudaMemcpy2D((void*) const_gpu.colors, const_gpu.col_p,data.colors,nx*sizeof(float) , nx*sizeof(float), ny*3,cudaMemcpyHostToDevice),"MemCpyColors");
    CUDA(cudaMemcpy2D((void*) const_gpu.textures, const_gpu.tex_p,data.textures,nx*sizeof(float) , nx*sizeof(float), ny*tex_dim,cudaMemcpyHostToDevice),"MemCpyTex");
    CUDA(cudaMemcpy(const_gpu.label_count, data.label_count, nr_labels*sizeof(int),cudaMemcpyHostToDevice),"MemCopyLabelCount");
    CUDA(cudaMemcpy(const_gpu.float_params, data.float_params,   (2+2*nr_labels)*sizeof(float),cudaMemcpyHostToDevice),"MemCopyFLOATPARAMS");
    CUDA(cudaMemcpy(const_gpu.temp, data.temp,   nr_scribbles*3*sizeof(float),cudaMemcpyHostToDevice),"MemCopyTemp");

    KERNEL(kernel_density,gridSize,blockSize)(const_gpu.scribbles, const_gpu.colors, const_gpu.textures,const_gpu.float_params,
                                              const_gpu.likely, const_gpu.temp,
                                              nx,ny,nr_labels,nr_scribbles,tex_dim, const_gpu.label_count,
                                              const_gpu.lt_p/sizeof(float),const_gpu.col_p/sizeof(float),const_gpu.tex_p/sizeof(float));


    CUDA(cudaMemcpy2D((void*) data.likely, nx*sizeof(float),const_gpu.likely, const_gpu.lt_p, nx*sizeof(float), ny*nr_labels,cudaMemcpyDeviceToHost),"MemCpyLikely");

    return true;


}


__global__ void kernel_wavelet(float *hh, float *hl, float *lh,
                               float *hh_avg, float *hl_avg, float *lh_avg,
                               float *hh_stddev, float *hl_stddev, float *lh_stddev,
                               int nx, int ny, int win_size, int nr_wav_steps, size_t pitch )
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int startx = max(x-win_size,0), endx = min(x+win_size,nx);
    int starty = max(y-win_size,0), endy = min(y+win_size,ny);
    if (x > win_size && x < nx-win_size-1 && y > win_size  && y < ny-win_size-1) {


        for(int n=0; n < nr_wav_steps;n++)
        {
            int off = n*pitch*ny;
            int nrVals=0;
            float mean_hh=0.0f, mean_hl=0.0f, mean_lh=0.0f;
            for(int i= startx; i < endx; i++)
                for(int j= starty; j < endy; j++)
                {
                    int pos = off+j*pitch+i;
                    mean_hh += (hh[pos]);
                    mean_hl += (hl[pos]);
                    mean_lh += (lh[pos]);
                    nrVals++;
                }

            hh_avg[off+y*pitch+x] =  mean_hh/nrVals;
            hl_avg[off+y*pitch+x] =  mean_hl/nrVals;
            lh_avg[off+y*pitch+x] =  mean_lh/nrVals;
        }
    }

    __syncthreads();

    if (x >= win_size && x < nx-win_size && y >= win_size  && y < ny-win_size) {
        for(int n=0; n < nr_wav_steps;n++)
        {
            int off = n*pitch*ny;
            int nrVals=0;
            float mean_hh=0.0f, mean_hl=0.0f, mean_lh=0.0f;
            for(int i= startx; i < endx; i++)
                for(int j= starty; j < endy; j++)
                {
                    int pos = off+j*pitch+i;
                    float res = (hh[pos]) - hh_avg[pos];
                    mean_hh += res*res;
                    res = (hl[pos]) - hl_avg[pos];
                    mean_hl += res*res;
                    res = (lh[pos]) - lh_avg[pos];
                    mean_lh += res*res;
                    nrVals++;
                }

            hh_stddev[off+y*pitch+x] =  sqrtf(mean_hh/(nrVals));
            hl_stddev[off+y*pitch+x] =  sqrtf(mean_hl/(nrVals));
            lh_stddev[off+y*pitch+x] =  sqrtf(mean_lh/(nrVals));
        }
    }

}



bool gpu_wavelet(GPU_DATA data)
{

    int nx = data.int_params[NX];
    int ny = data.int_params[NY];
    int win_size = data.int_params[WAVELET_WIN_SIZE];
    int nr_wav_steps = data.int_params[WAVELET_STEPS];

    float size = 8;
    dim3 blockSize(size, size);
    dim3 gridSize( (int)ceil(nx/size), (int)ceil(ny/size) );

    float *hh,*hl,*lh,*hh_avg,*hl_avg,*lh_avg,*hh_stddev,*hl_stddev,*lh_stddev;
    size_t pitch;


    CUDA(cudaMallocPitch((void**) &hh,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocHH") ;
    CUDA(cudaMallocPitch((void**) &hl,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocHL") ;
    CUDA(cudaMallocPitch((void**) &lh,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocLH") ;

    CUDA(cudaMemcpy2D((void*) hh, pitch,data.hh,nx*sizeof(float) , nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyHostToDevice),"MemCpyHH");
    CUDA(cudaMemcpy2D((void*) hl, pitch,data.hl,nx*sizeof(float) , nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyHostToDevice),"MemCpyHL");
    CUDA(cudaMemcpy2D((void*) lh, pitch,data.lh,nx*sizeof(float) , nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyHostToDevice),"MemCpyLH");


    CUDA(cudaMallocPitch((void**) &hh_avg,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocHHAVG") ;
    CUDA(cudaMallocPitch((void**) &hl_avg,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocHLAVG") ;
    CUDA(cudaMallocPitch((void**) &lh_avg,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocLHAVG") ;
    CUDA(cudaMallocPitch((void**) &hh_stddev,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocHHSTDDEV") ;
    CUDA(cudaMallocPitch((void**) &hl_stddev,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocHLSTDDEV") ;
    CUDA(cudaMallocPitch((void**) &lh_stddev,&pitch,  nx*sizeof(float), ny*nr_wav_steps), "MallocLHSTDDEV") ;

    CUDA(cudaMemset2D((void*)hh_avg, pitch,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHHAVG") ;
    CUDA(cudaMemset2D((void*)hl_avg, pitch,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHLAVG") ;
    CUDA(cudaMemset2D((void*)lh_avg, pitch,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetLHAVG") ;
    CUDA(cudaMemset2D((void*)hh_stddev, pitch,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHHSTDDEV") ;
    CUDA(cudaMemset2D((void*)hl_stddev, pitch,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHLSTDDEV") ;
    CUDA(cudaMemset2D((void*)lh_stddev, pitch,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetLHSTDDEV") ;


    KERNEL(kernel_wavelet,gridSize,blockSize)(hh,hl,lh,hh_avg,hl_avg,lh_avg,hh_stddev,hl_stddev,lh_stddev,
                                              nx,ny,win_size,nr_wav_steps,pitch/sizeof(float));


    CUDA(cudaMemcpy2D((void*) data.hh_avg, nx*sizeof(float),hh_avg, pitch, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHH_AVG");
    CUDA(cudaMemcpy2D((void*) data.hl_avg, nx*sizeof(float),hl_avg, pitch, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHL_AVG");
    CUDA(cudaMemcpy2D((void*) data.lh_avg, nx*sizeof(float),lh_avg, pitch, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyLH_AVG");
    CUDA(cudaMemcpy2D((void*) data.hh_stddev, nx*sizeof(float),hh_stddev, pitch, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHH_STDDEV");
    CUDA(cudaMemcpy2D((void*) data.hl_stddev, nx*sizeof(float),hl_stddev, pitch, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHL_STDDEV");
    CUDA(cudaMemcpy2D((void*) data.lh_stddev, nx*sizeof(float),lh_stddev, pitch, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyLH_STDDEV");


    cudaFree(lh);
    cudaFree(hh);
    cudaFree(hl);
    cudaFree(hh_avg);
    cudaFree(hl_avg);
    cudaFree(lh_avg);
    cudaFree(hh_stddev);
    cudaFree(hl_stddev);
    cudaFree(lh_stddev);

    return true;


}

bool gpu_wavelet(GPU_DATA data, GPU_DATA const_gpu)
{

    int nx = data.int_params[NX];
    int ny = data.int_params[NY];
    int win_size = data.int_params[WAVELET_WIN_SIZE];
    int nr_wav_steps = data.int_params[WAVELET_STEPS];

    float size = 8;
    dim3 blockSize(size, size);
    dim3 gridSize( (int)ceil(nx/size), (int)ceil(ny/size) );





    CUDA(cudaMemcpy2D((void*) const_gpu.hh, const_gpu.wave_p,data.hh,nx*sizeof(float) , nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyHostToDevice),"MemCpyHH");
    CUDA(cudaMemcpy2D((void*) const_gpu.hl, const_gpu.wave_p,data.hl,nx*sizeof(float) , nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyHostToDevice),"MemCpyHL");
    CUDA(cudaMemcpy2D((void*) const_gpu.lh, const_gpu.wave_p,data.lh,nx*sizeof(float) , nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyHostToDevice),"MemCpyLH");


    CUDA(cudaMemset2D((void*)const_gpu.hh_avg, const_gpu.wave_p,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHHAVG") ;
    CUDA(cudaMemset2D((void*)const_gpu.hl_avg, const_gpu.wave_p,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHLAVG") ;
    CUDA(cudaMemset2D((void*)const_gpu.lh_avg, const_gpu.wave_p,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetLHAVG") ;
    CUDA(cudaMemset2D((void*)const_gpu.hh_stddev, const_gpu.wave_p,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHHSTDDEV") ;
    CUDA(cudaMemset2D((void*)const_gpu.hl_stddev, const_gpu.wave_p,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetHLSTDDEV") ;
    CUDA(cudaMemset2D((void*)const_gpu.lh_stddev, const_gpu.wave_p,0, nx*sizeof(float), ny*nr_wav_steps), "MemsetLHSTDDEV") ;


    KERNEL(kernel_wavelet,gridSize,blockSize)(const_gpu.hh,const_gpu.hl,const_gpu.lh,const_gpu.hh_avg,const_gpu.hl_avg,const_gpu.lh_avg,
                                              const_gpu.hh_stddev,const_gpu.hl_stddev,const_gpu.lh_stddev,
                                              nx,ny,win_size,nr_wav_steps,const_gpu.wave_p/sizeof(float));


    CUDA(cudaMemcpy2D((void*) data.hh_avg, nx*sizeof(float),const_gpu.hh_avg, const_gpu.wave_p, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHH_AVG");
    CUDA(cudaMemcpy2D((void*) data.hl_avg, nx*sizeof(float),const_gpu.hl_avg, const_gpu.wave_p, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHL_AVG");
    CUDA(cudaMemcpy2D((void*) data.lh_avg, nx*sizeof(float),const_gpu.lh_avg, const_gpu.wave_p, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyLH_AVG");
    CUDA(cudaMemcpy2D((void*) data.hh_stddev, nx*sizeof(float),const_gpu.hh_stddev, const_gpu.wave_p, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHH_STDDEV");
    CUDA(cudaMemcpy2D((void*) data.hl_stddev, nx*sizeof(float),const_gpu.hl_stddev, const_gpu.wave_p, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyHL_STDDEV");
    CUDA(cudaMemcpy2D((void*) data.lh_stddev, nx*sizeof(float),const_gpu.lh_stddev, const_gpu.wave_p, nx*sizeof(float), ny*nr_wav_steps,cudaMemcpyDeviceToHost),"MemCpyLH_STDDEV");



    return true;


}


