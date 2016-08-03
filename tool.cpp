
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>
#include <QFileDialog>
#include <iomanip>
#include <stdio.h>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>
#include <sys/types.h>
#include <sys/dir.h>
#include <dirent.h>
#include <ctime>

#include <opencv2/imgproc.hpp>

#include <cuda_runtime.h>

#include "tool.h"



#ifndef __APPLE__
#define CONST_GPU
#endif

using namespace std;


QImage fromCVMat(Mat &m)
{
    QImage image(m.cols,m.rows,QImage::Format_RGB888);
    switch(m.type())
    {
    case CV_32FC3:
    {
        for(int x=0; x < m.cols; x++)
            for(int y=0; y < m.rows; y++){
                Vec3f val= m.at<Vec3f>(y,x)*255;
                image.setPixel(x,y,qRgb(val[0],val[1],val[2]));   // RGB
            }
    }
        break;

    case CV_32FC1:
    {
        for(int x=0; x < m.cols; x++)
            for(int y=0; y < m.rows; y++){
                float f = m.at<float>(y,x)*255;
                image.setPixel(x,y,qRgb(f,f,f));
            }
    }
        break;


    default:
        cerr << "Unknown cv::Mat type. CHECK AND ADD TO THIS FUNCTION! " << m.type() << endl;
        return QImage();
    }

    return image;
}

Mat fromQimage(QImage &image)
{
    Mat m(image.height(),image.width(),CV_32FC3);
    for(int x=0; x < m.cols; x++)
        for(int y=0; y < m.rows; y++){
            QRgb rgb = image.pixel(x,y);
            m.at<Vec3f>(y,x)[2] = qRed(rgb)/255.f;
            m.at<Vec3f>(y,x)[1] = qGreen(rgb)/255.f;
            m.at<Vec3f>(y,x)[0] = qBlue(rgb)/255.f;
        }
    return m;
}



Tool::Tool() : QObject()
{
    kernel_alpha = 2;
    kernel_sigma=0.2f;
    kernel_beta=0.2f;
    kernel_delta=1.0f;
    nr_labels=0;
    eig=0;
    ColorMode=0;
    GMode=0;
    LDATexSpace = false;
    seg_it=50;
    seg_lambda=10;
    Anisotropy=false;
    WaveletFilter = "Haar";
    Color=Space=Texture=true;
    TexWinSize=10;
    TexDim = 5;
    NrWaveletSteps=1;
    BrushSize=1;
    EstimateGPU = true;
    seg_tau = 0.5;
    seg_eta = 0.1;
    ColorDone=TextureDone = false;
    AutoParam = false;


#ifdef CONST_GPU


    gpu.likely = new float[1000*1000*15];
    gpu.labeling = new int[1000*1000];
    gpu.scribbles = new int[1000*1000*3];
    gpu.label_count = new int[15];
    gpu.colors = new float[1000*1000*3];
    gpu.int_params = new int[INT_PARAMS];
    gpu.float_params = new float[2 + 2*15];
    gpu.textures = new float[1000*1000*10];
    gpu.hh = new float[1000*1000*3];
    gpu.hl = new float[1000*1000*3];
    gpu.lh = new float[1000*1000*3];
    gpu.hh_avg = new float[1000*1000*3];
    gpu.hl_avg = new float[1000*1000*3];
    gpu.lh_avg = new float[1000*1000*3];
    gpu.hh_stddev = new float[1000*1000*3];
    gpu.hl_stddev = new float[1000*1000*3];
    gpu.lh_stddev = new float[1000*1000*3];

    gpu.primal = new float[2*1000*1000*15];
    gpu.dual = new float[2*1000*1000*15];
    gpu.g = new float[1000*1000];
    gpu.temp = new float[1000*1000*15];



    // Takes about 500MB of vmem
    cudaDeviceReset();
    //   cerr << "USING CONSTANT GPU MEMORY" << endl;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.hh,&const_gpu.wave_p,        1000*sizeof(float), 1000*3), "MallocHH") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.hl,&const_gpu.wave_p,        1000*sizeof(float), 1000*3), "MallocHL") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.lh,&const_gpu.wave_p,        1000*sizeof(float), 1000*3), "MallocLH") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.hh_avg,&const_gpu.wave_p,    1000*sizeof(float), 1000*3), "MallocHHAVG") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.hl_avg,&const_gpu.wave_p,    1000*sizeof(float), 1000*3), "MallocHLAVG") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.lh_avg,&const_gpu.wave_p,    1000*sizeof(float), 1000*3), "MallocLHAVG") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.hh_stddev,&const_gpu.wave_p, 1000*sizeof(float), 1000*3), "MallocHHSTDDEV") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.hl_stddev,&const_gpu.wave_p, 1000*sizeof(float), 1000*3), "MallocHLSTDDEV") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.lh_stddev,&const_gpu.wave_p, 1000*sizeof(float), 1000*3), "MallocLHSTDDEV") ;

    CUDA_2(cudaMallocPitch((void**) &const_gpu.likely,  &const_gpu.lt_p,    1000*sizeof(float), 1000*15), "MallocLikely") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.primal,  &const_gpu.pd_p,    1000*sizeof(float), 1000*15*2), "MallocPrimal") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.dual,    &const_gpu.pd_p,    1000*sizeof(float), 1000*15*2), "MallocDual") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.temp,    &const_gpu.lt_p,    1000*sizeof(float), 1000*15), "MallocTemp") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.g,       &const_gpu.g_p,     1000*sizeof(float), 1000), "MallocG") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.colors,  &const_gpu.col_p,   1000*sizeof(float), 1000*3), "MallocColors") ;
    CUDA_2(cudaMallocPitch((void**) &const_gpu.textures,&const_gpu.tex_p,   1000*sizeof(float), 1000*10), "MallocTex") ;

    CUDA_2(cudaMalloc(&(const_gpu.labeling),   1000*1000*sizeof(int)),"MallocLabeling") ;
    CUDA_2(cudaMalloc(&(const_gpu.scribbles),  1000*1000*sizeof(int)*3),"MallocScribbles") ;
    CUDA_2(cudaMalloc(&(const_gpu.label_count),       15*sizeof(int)),"MallocLabelCount") ;
    CUDA_2(cudaMalloc(&(const_gpu.float_params),(2+2*15)*sizeof(float)),"MallocFLOATPARAMS") ;

#endif



}

Tool::~Tool()
{

}


void Tool::readScribbles()
{
    scribbles.clear();
    scribbles_orig.clear();
    nr_labels = 0;

    label_count.assign(15,0);
    label_count_orig.assign(15,0);

    Mat im = fromQimage(in->back);

    for (int i=0; i < 1000; i++)
        for(int j=0; j < 1000; j++)
        {
            if(in->scribbles[i][j] > -1)
            {
                Scribble s;
                s.x = i;
                s.y = j;
                s.label = in->scribbles[i][j];
                s.rgb = im.at<Vec3f>(i,j);
                label_count[s.label]++;
                nr_labels = max((int)nr_labels,s.label+1);
                scribbles.push_back(s);
            }

            if(in->scribbles_orig[i][j] > -1)
            {
                Scribble s;
                s.x = i;
                s.y = j;
                s.label = in->scribbles_orig[i][j];
                s.rgb = im.at<Vec3f>(i,j);
                label_count_orig[s.label]++;
                scribbles_orig.push_back(s);
            }

        }


    ColorDone=false;
    TextureDone=false;
}


void Tool::setColorMode(int i)
{
    ColorMode=i;
}

void Tool::setDelta(double d)
{
    kernel_delta = d;
}

void Tool::setTextureMode(int i)
{
    LDATexSpace= (i == 0) ? false : true;
    TextureDone=false;
}

void Tool::setEstimateGPU(int i)
{
    EstimateGPU = (i == 0) ? false : true;
}

void Tool::setWaveletSteps(int i)
{
    NrWaveletSteps = i;
    TextureDone=false;
}

void Tool::setTexDim(int i)
{
    TexDim = i;
}

void Tool::setTexWin(int i)
{
    TexWinSize = i;
}

void Tool::setAnisotropy(int i)
{
    Anisotropy = i;
}

void Tool::setGMode(int i)
{
    GMode = i;
}

void Tool::setBrushSize(int i)
{
    BrushSize=i;
    in->brushSize = i;
}



void Tool::setSigma(double s)
{
    kernel_sigma=s;
}

void Tool::setLambda(double l)
{
    seg_lambda=l;
}

void Tool::setTau(double t)
{
    seg_tau =t;
}

void Tool::setAlpha(double a)
{
    kernel_alpha=a;
}

void Tool::setBeta(double b)
{
    kernel_beta=b;
}

void Tool::setEta(double e)
{
    seg_eta=e;
}

void Tool::setColor(int i)
{
    Color = (i == 0) ? false : true;
}

void Tool::setSpace(int i)
{
    Space= (i == 0) ? false : true;
}

void Tool::setTexture(int i)
{
    Texture= (i == 0) ? false : true;
}

void Tool::setAutoParam(int i)
{
    AutoParam= (i == 0) ? false : true;
}

void Tool::setWaveletFilter(QString str)
{
    WaveletFilter = str;
    TextureDone=false;
}

void Tool::setIterations(int it)
{
    seg_it=it;
}


bool Tool::ColorLDA(bool show,bool ortho)
{

    if(scribbles_orig.size() == 0 || nr_labels < 2)
    {
        cout << "No scribbles read!" << endl;
        return false;
    }

    Mat Call;
    vector<Mat> mats(nr_labels);

    // Fill the matrices with rgb values for each label
    for(Scribble s : scribbles_orig)
    {
        Mat t = (cv::Mat_<float>(1,3) << s.rgb[0], s.rgb[1], s.rgb[2]);
        mats[s.label].push_back(t);
        Mat t2 = (cv::Mat_<float>(1,3) << s.rgb[0], s.rgb[1], s.rgb[2]);
        Call.push_back(t2);
    }

    Mat LDAColorMatrix(3,3,CV_32F);

    bool result;
    if(ortho)  result = OLDA(LDAColorMatrix,Call,mats, scribbles_orig,label_count_orig);
    else result = LDA(LDAColorMatrix,Call,mats);
    if(result == false) return false;

    int w = in->back.width();
    int h = in->back.height();
    Mat temp = fromQimage(in->back);
    cv::split(temp,Colors);

    Vec3f min,max;
    for(int x=0; x < w; x++)
        for(int y=0; y < h; y++)
        {
            Vec3f rgb = temp.at<Vec3f>(x,y);
            rgb = LDAColorMatrix.dot(rgb);
            for(int j=0; j < 3; j++)
            {
                Colors.at(j).at<float>(x,y) = rgb[j];
                min(j) = std::min(min(j),Colors.at(j).at<float>(x,y));
                max(j) = std::max(max(j),Colors.at(j).at<float>(x,y));
            }


        }

    ColorDone=true;
    LDAimage = Mat(w,h,CV_32F);
    for(int x=0; x < temp.rows; x++)
        for(int y=0; y < temp.rows; y++)
        {
            float r = (Colors.at(0).at<float>(x,y)-min(0))/(max(0)-min(0));
            float g = (Colors.at(1).at<float>(x,y)-min(1))/(max(1)-min(1));
            float b = (Colors.at(2).at<float>(x,y)-min(2))/(max(2)-min(2));
            float c = (r+g+b)/3.0f;
            LDAimage.at<float>(x,y)=c;
            Vec3f pix;
            if(eig==0) pix = Vec3f(r,g,b);
            else if (eig==1) pix = Vec3f(r,r,r);
            else if (eig==2) pix = Vec3f(g,g,g);
            else if (eig==3) pix = Vec3f(b,b,b);
            else if (eig==4) pix = Vec3f(c,c,c);
            temp.at<Vec3f>(x,y) = pix;
        }

    if(!show) return true;
    QImage im = fromCVMat(temp);
    out->setImage(im);
    return true;

}




void Tool::doColorLDA()
{
    readScribbles();
    if(!ColorLDA(true,false))
        cerr << "ColoLDA failed" << endl;
}

void Tool::doColorOLDA()
{
    readScribbles();
    if(!ColorLDA(true,true))
        cerr << "ColoLDA failed" << endl;}


void Tool::setEigenVector(int i)
{
    eig = i;
}

void Tool::doGTerm()
{
    GTerm(true);
}

bool Tool::GTerm(bool show)
{

    readScribbles();
    if(scribbles.size() == 0)
        return false;

    QImage im(in->back);
    int w = im.width();
    int h = im.height();
    Mat values(w,h,CV_32F,Scalar(0));

    // G is color gradient
    if(GMode==1)
    {
        if(ColorMode>0 && !ColorLDA(false,ColorMode==2)) return false;

        // standard grayscale
        if(ColorMode==0)
        {
            Colors.clear();
            for(int i=0; i < 3;i++)
                Colors.push_back(Mat(w,h,CV_32F));

            for(int x=0; x < w; x++)
                for(int y=0; y < h; y++)
                {
                    int col = im.pixel(x,y);
                    Colors.at(0).at<float>(x,y) = qRed(col);
                    Colors.at(1).at<float>(x,y) = qGreen(col);
                    Colors.at(2).at<float>(x,y) = qBlue(col);
                    values.at<float>(x,y) = qGray(col);
                }
        }
        // else in OLDA space
        else Colors[0].copyTo(values);

    }

    // G is texture gradient
    if(GMode==2)
    {

        if (!Wavelet(false)) return false;
        for(Mat &t : Textures) values += t;
        values /= Textures.size();

    }

    G = Mat(w,h,CV_32F,Scalar(1));
    if(GMode>0)
    {
        for(int x=1; x < w-1; x++)
            for(int y=1; y < h-1; y++)
            {
                float dx=0,dy=0;
                dx = values.at<float>(x+1,y)-values.at<float>(x-1,y);
                dy = values.at<float>(x,y+1)-values.at<float>(x,y-1);
                G.at<float>(x,y)= exp(-seg_eta*(abs(0.5*dx)+abs(0.5*dy)));
            }
    }

    if(!show) return true;

    double Gmin,Gmax;
    cv::minMaxIdx(G,&Gmin,&Gmax);
    float scale = 255.0f/Gmax;

    for(int x=0; x < w; x++)
        for(int y=0; y < h; y++)
        {
            float val = G.at<float>(x,y)*scale;
            im.setPixel(x,y,qRgb(val,val,val));
        }
    out->setImage(im);
    return true;
}


bool Tool::doKernelEstimation()
{
    return doKernelEstimation(true);
}

bool Tool::doKernelEstimation(bool doAll)
{
    readScribbles();
    if(scribbles.empty())
        return false;

    vector<Mat> mats(nr_labels);
    for(Scribble &s : scribbles)
    {
        Mat t = (cv::Mat_<float>(1,2) << s.x, s.y);
        mats[s.label].push_back(t);
    }

    /*  TODO
    for(unsigned i=0; i < nr_labels; i++)
    {
        Mat a = cv::cov(mats.at(i));
        double s = sum(sum(abs(a)));
        a = a/s;
        //cout << a << endl;
    }
    */
    if(doAll)
    {
        if(Color && ColorMode>0)
        {
            if(!ColorLDA(false,ColorMode==2))
            {
                cerr << "ColorLDA failure"<<endl;
                return false;
            }
        }
        if((Texture))
        {
            if (!Wavelet(false))
            {
                cerr << "Wavelet failure"<<endl;
                return false;
            }
        }
    }


    Mat im = fromQimage(in->back);
    int unstable=0;
    unsigned w = im.cols;
    unsigned h = im.rows;

    currentIt = 0;


#ifndef CONST_GPU
    gpu.likely = new float[w*h*nr_labels];
    gpu.labeling = new int[w*h];
    gpu.scribbles = new int[scribbles.size()*3];
    gpu.label_count = new int[nr_labels];
    gpu.colors = new float[w*h*3];
    gpu.int_params = new int[INT_PARAMS];
    gpu.float_params = new float[2 + 2*nr_labels];
    gpu.textures = new float[w*h*Textures.size()];
    gpu.temp = new float[scribbles.size()*3];
#endif

    gpu.int_params[NX] = w;
    gpu.int_params[NY] = h;
    gpu.int_params[NR_SCRIBBLES] = scribbles.size();
    gpu.int_params[NR_LABELS] = nr_labels;
    gpu.int_params[TEX_DIM] = Textures.size();

    gpu.ani=Anisotropy;


    vector<Mat> maha(nr_labels);

    if(gpu.ani)
    {
        for(Scribble &s : scribbles_orig)
        {
            Mat t = (cv::Mat_<float>(1,2) << s.x, s.y);
            maha[s.label].push_back(t);
        }

        /* TODO
        for(unsigned i=0; i < nr_labels;i++)
        {
            // If non-invertible, just use isotropic model
            if(inv(maha.at(i),cov(maha.at(i)))==true)
                maha.at(i) /= max(max(abs(maha.at(i))));
            else
                maha.at(i) = eye(2,2);
            //cout << maha.at(i) << endl;
        }
        */
    }

    for(size_t i=0; i < scribbles.size();i++)
    {
        Scribble s = scribbles.at(i);
        gpu.scribbles[i*3 + 0] = s.x;
        gpu.scribbles[i*3 + 1] = s.y;
        gpu.scribbles[i*3 + 2] = s.label;
        if(gpu.ani)
        {
            gpu.temp[i*3 + 0] = maha.at(s.label).at<float>(0,0);
            gpu.temp[i*3 + 1] = maha.at(s.label).at<float>(1,1);
            gpu.temp[i*3 + 2] = Anisotropy==1 ? maha.at(s.label).at<float>(1,0) : 0.0f;
        }
        else
        {
            gpu.temp[i*3 + 0] = gpu.temp[i*3 + 1] = 1.0f;
            gpu.temp[i*3 + 2] = 0.0f;
        }

    }


    gpu.float_params[0] = Space? kernel_alpha : 0.0f;
    gpu.float_params[1] = kernel_delta;
    for(unsigned i=0; i < nr_labels; i++)
    {
        gpu.label_count[i] = label_count[i];
        gpu.float_params[2 + i] = Color ? kernel_sigma : 0.0f;
        gpu.float_params[2 + nr_labels + i] = Texture? kernel_beta : 0.0f;
    }


    if(Color)
    {

        if (ColorMode==0)
            cv::split(im,Colors);

        // Copy color data into gpu array
        for(unsigned i=0; i < Colors.size();i++ )
            memcpy(&(gpu.colors[w*h*i]), Colors[i].data,sizeof(float)*w*h);


        if(AutoParam)
        {
            vector<Mat> mats(nr_labels);
            for(Scribble &s : scribbles)
            {
                Mat t(1,3,CV_32F);
                for(int k=0; k < 3; k++)
                    t.at<float>(0,k) += gpu.colors[w*h*k + (s.y*w+s.x)];
                mats[s.label].push_back(t);
            }

            for(unsigned i=0; i < nr_labels;i++)
            {
                double mmin,mmax, value=1;
                if (!mats[i].empty())
                {
                    Mat cov,mu;
                    cv::calcCovarMatrix(mats[i],cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
                    cov = cov / (mats[i].rows - 1);
                    cv::minMaxIdx(cov,&mmin,&mmax);
                    value = sqrt(mmax);
                }
                cout << value << " ";
                gpu.float_params[2 +i] = max(min(value,1.0),0.03);
            }
            cout << endl;

        }


    }

    /*
    if(Texture)
    {

        // Scale and copy color data into gpu array
        for(unsigned i=0; i < Textures.size();i++ )
        {
            float c = as_scalar(max(max(abs(Textures.at(i)))));
            for(unsigned x=0; x < w; x++)
                for(unsigned y=0; y < h; y++)
                    gpu.textures[w*h*i+y*w+x] = Textures.at(i).at(x,y)/c;
        }

        if(AutoParam)
        {

            vector<mat> mats;
            for(unsigned i=0; i < nr_labels;i++)
                mats.push_back(zeros(label_count[i],Textures.size()));
            vec off = zeros(nr_labels);
            for(unsigned i=0; i < scribbles.size(); i++)
            {
                Scribble s = scribbles.at(i);
                for(unsigned k=0; k < Textures.size(); k++)
                    mats.at(s.label).at(off.at(s.label),k) += gpu.textures[w*h*k + (s.y*w+s.x)];
                off.at(s.label)++;
            }

            for(unsigned i=0; i < nr_labels;i++)
            {
                double value = sqrt(as_scalar(max(var(mats.at(i),1),1)));
                gpu.float_params[2 + nr_labels +i] = max(min(value,1.0),0.03);
                cout << value << " " ;

            }
            //cout << endl;

        }
    }
*/


#ifdef CONST_GPU
    if(!gpu_density(gpu,const_gpu))
#else
    if(!gpu_density(gpu))
#endif
    {
        cerr << "Kernel estimation failed!" << endl;
        return false;
    }


    // Copy likelihoods and clamp to good numeric bounds
    likely.clear();
    float clamp = 1000.f;
    for(unsigned l=0; l < nr_labels; l++)
    {
        likely.push_back(Mat(w,h,CV_32F));
        for(unsigned x=0; x < w; x++)
            for(unsigned y=0; y < h; y++)
            {
                float value = log(gpu.likely[w*h*l + y*w + x]);
                likely.at(l).at<float>(x,y)= max(-clamp,min(value,clamp));
            }
    }


    for(unsigned x=0; x < w; x++)
        for(unsigned y=0; y < h; y++)
        {
            // numerical stability check (likelihoods should not coincide)
            bool stable = true;
            for(unsigned l=0; l < nr_labels-1;l++)
                stable &= (likely.at(l).at<float>(x,y) != likely.at(l+1).at<float>(x,y));

            Vec3f lab = getRgbOfLabel(gpu.labeling[y*w+x]);
            if(!stable) lab = getRgbOfLabel(-1);
            im.at<Vec3f>(y,x) = im.at<Vec3f>(y,x)*0.5 + lab*0.5;
        }

#ifndef CONST_GPU
    delete gpu.labeling;
    delete gpu.likely;
    delete gpu.scribbles;
    delete gpu.label_count;
    delete gpu.colors;
    delete gpu.textures;
    delete gpu.int_params;
    delete gpu.float_params;
#endif

    QImage QIm = fromCVMat(im);
    out->setImage(QIm);
    return true;


}

void Tool::doLBP()
{
    ColorLBP(5,31);
}

bool Tool::LBP(int radius, int samples)
{

    int w = in->back.width();
    int h = in->back.height();
    Mat src = Mat::zeros(w,h,CV_32F);
    Textures.clear();
    Textures.push_back(Mat::zeros(w,h,CV_32F));

    /*
    for(int i=0; i < w;i++)
        for (int j=0; j < h;j++)
            src.at(i,j) = qGray(in->back.pixel(i,j));


    mat dst= zeros(w,h);
    for(int n=0; n<samples; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0f*M_PI*n/static_cast<float>(samples));
        float y = static_cast<float>(radius) * -sin(2.0f*M_PI*n/static_cast<float>(samples));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < w-radius;i++) {
            for(int j=radius;j < h-radius;j++) {
                float t = w1*src.at(i+fy,j+fx) + w2*src.at(i+fy,j+cx) + w3*src.at(i+cy,j+fx) + w4*src.at(i+cy,j+cx);
                float res = ((t > src.at(i,j)) && (abs(t-src.at(i,j)) > std::numeric_limits<float>::epsilon())) << n;
                dst.at(i,j) += res;
            }
        }
    }

    mat dist = zeros(w,h);
    for (unsigned x=0; x < dst.n_rows;x++)
        for (unsigned y=0; y < dst.n_cols;y++)
        {

            int diff=0;
            for(unsigned s=0; s < scribbles.size();s++)
            {
                Scribble sc = scribbles.at(s);
                if(sc.label!=0) continue;
                int bits = ((int)dst.at(x,y)) ^ ((int) dst.at(sc.x,sc.y));
                while (bits)
                {
                    diff++;
                    bits &= bits - 1;
                }
            }
            dist.at(x,y) += diff;

        }

    float max_dist = as_scalar(max(max(dist)));
    QImage image(dst.n_rows,dst.n_cols,QImage::Format_RGB32);
    for (unsigned x=0; x < dst.n_rows;x++)
        for (unsigned y=0; y < dst.n_cols;y++)
        {
            Textures.at(0).at(x,y) = dst.at(x,y);
            int value = (dist.at(x,y)/max_dist)*255;

            image.setPixel(x,y,qRgb(value,value,value));
        }
    out->setImage(image);


*/
    return true;
}




bool Tool::ColorLBP(int radius, int samples)
{
    /*
    int w = in->back.width();
    int h = in->back.height();

    Mat src_r = zeros(w,h),src_g = zeros(w,h),src_b = zeros(w,h);
    for(int i=0; i < w;i++)
        for (int j=0; j < h;j++)
        {
            src_r.at(i,j) = qRed(in->back.pixel(i,j));
            src_g.at(i,j) = qGreen(in->back.pixel(i,j));
            src_b.at(i,j) = qBlue(in->back.pixel(i,j));
        }

    mat dst_r= zeros(w,h),dst_g= zeros(w,h),dst_b = zeros(w,h);
    for(int n=0; n<samples; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0f*M_PI*n/static_cast<float>(samples));
        float y = static_cast<float>(radius) * -sin(2.0f*M_PI*n/static_cast<float>(samples));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < w-radius;i++) {
            for(int j=radius;j < h-radius;j++) {
                float r = w1*src_r.at(i+fy,j+fx) + w2*src_r.at(i+fy,j+cx) + w3*src_r.at(i+cy,j+fx) + w4*src_r.at(i+cy,j+cx);
                float g = w1*src_g.at(i+fy,j+fx) + w2*src_g.at(i+fy,j+cx) + w3*src_g.at(i+cy,j+fx) + w4*src_g.at(i+cy,j+cx);
                float b = w1*src_b.at(i+fy,j+fx) + w2*src_b.at(i+fy,j+cx) + w3*src_b.at(i+cy,j+fx) + w4*src_b.at(i+cy,j+cx);

                dst_r.at(i,j) += ((r > src_r.at(i,j)) && (abs(r-src_r.at(i,j)) > std::numeric_limits<float>::epsilon())) << n;
                dst_g.at(i,j) += ((g > src_g.at(i,j)) && (abs(g-src_g.at(i,j)) > std::numeric_limits<float>::epsilon())) << n;
                dst_b.at(i,j) += ((b > src_b.at(i,j)) && (abs(b-src_b.at(i,j)) > std::numeric_limits<float>::epsilon())) << n;

            }
        }
    }

    Textures.clear();
    for(unsigned i=0; i < 3; i++)
        Textures.push_back(mat(w,h));
    mat dist = zeros(w,h);
    for (int x=0; x < w;x++)
        for (int y=0; y < h;y++)
        {

            int diff=0;
            for(unsigned s=0; s < scribbles.size();s++)
            {
                Scribble sc = scribbles.at(s);
                if(sc.label!=0) continue;
                int bits = ((int)dst_r.at(x,y)) ^ ((int) dst_r.at(sc.x,sc.y));
                while (bits)
                {
                    diff++;
                    bits &= bits - 1;
                }
                bits = ((int)dst_g.at(x,y)) ^ ((int) dst_g.at(sc.x,sc.y));
                while (bits)
                {
                    diff++;
                    bits &= bits - 1;
                }
                bits = ((int)dst_b.at(x,y)) ^ ((int) dst_b.at(sc.x,sc.y));
                while (bits)
                {
                    diff++;
                    bits &= bits - 1;
                }
            }
            dist.at(x,y) += diff;

        }

    float max_dist = as_scalar(max(max(dist)));
    QImage image(w,h,QImage::Format_RGB32);
    for (int x=0; x < w;x++)
        for (int y=0; y < h;y++)
        {
            //image.setPixel(x,y,qRgb(dst_r.at(x,y),dst_g.at(x,y),dst_b.at(x,y)));
            int value = (dist.at(x,y)/max_dist)*255;
            image.setPixel(x,y,qRgb(value,value,value));
            Textures.at(0).at(x,y) = dst_r.at(x,y);
            Textures.at(1).at(x,y) = dst_g.at(x,y);
            Textures.at(2).at(x,y) = dst_b.at(x,y);
        }

    out->setImage(image);


*/
    return true;
}



void Tool::doDumpEstimation()
{
    unsigned w = likely.at(0).rows;
    unsigned h = likely.at(0).cols;
    double maxi=-10000000,mini=1000000000;
    for (unsigned i=0; i< likely.size();i++)
    {
        double tmin,tmax;
        cv::minMaxIdx(likely[i],&tmin,&tmax);
        maxi = std::max(tmax,maxi);
        mini = std::min(tmin,mini);
    }
    cerr << mini << " " << maxi << endl;
    float lmaxi = log(maxi);
    float lmini = -log(-mini);
    cerr << lmini << " " << lmaxi << endl;

    for (unsigned i=0; i< likely.size();i++)
    {
        QImage image(w,h,QImage::Format_RGB32);
        for (unsigned x=0; x < w;x++)
            for (unsigned y=0; y < h;y++)
            {
                float v = likely.at(i).at<float>(x,y)-mini+0.00000001f;
                int value = (int)((v/(maxi-mini))*255);
                //if(value < 0 || value > 255) cerr << value<<endl;
                image.setPixel(x,y,qRgb(value,value,value));
            }
        stringstream ss;

#ifdef __APPLE__
        ss << "../../../";
#endif
        ss << i << "_estimate.png";
        image.save(ss.str().c_str());

    }
}

void Tool::printPixelInformation(unsigned x, unsigned y)
{
    if(likely.size() == 0) return;

    unsigned w = likely.at(0).rows;
    unsigned h = likely.at(0).cols;

    if(x>=w || y >= h) return;

    cout << "("<< setw(3)<<x<<","<< setw(3) <<y<<")"  << endl << "   LIKELY: ";
    for(unsigned l=0; l < nr_labels; l++)
        cout << setw(12)<< likely.at(l).at<float>(x,y) << "  " << flush;

    if (!labeling.empty())
        cout << " Label: " << labeling.at<float>(x,y);

    if (!groundtruth.empty())
        cout << " GT: " << groundtruth.at<float>(x,y);

    if (!G.empty())
        cout << " G: " << G.at<float>(x,y);

    cout << endl;

    if (!primal.empty()>0)
    {
        float sum = 0;
        cout << "   PRIMAL:  ";
        for(unsigned l=0; l < nr_labels; l++)
        {
            cout << setw(12)<< primal.at(l).at<float>(x,y) << "  ";
            sum += primal.at(l).at<float>(x,y);
        }
        cout << " | " << sum << endl;
    }

    if (dual.size()>0)
    {
        float sum_x = 0;
        cout << "   DUAL_X: ";
        for(unsigned l=0; l < nr_labels; l++)
        {
            cout << setw(12)<< dual.at(l).at<float>(x,y) << "  ";
            sum_x += dual.at(l).at<float>(x,y);
        }
        cout << endl;

        float sum_y = 0;
        cout << "   DUAL_Y: ";
        for(unsigned l=0; l < nr_labels; l++)
        {
            cout << setw(12) << dual.at(l).at<float>(x+w,y) << "  ";
            sum_y += dual.at(l).at<float>(x+w,y);

        }
        cout << endl;

        cout << "   NORM:  ";
        for(unsigned l=0; l < nr_labels; l++)
        {
            float norm = dual.at(l).at<float>(x,y)*dual.at(l).at<float>(x,y)+
                    dual.at(l).at<float>(x+w,y)*dual.at(l).at<float>(x+w,y);
            cout << setw(12) << sqrtf(norm) << "  ";
        }

        float k_norm = sqrtf(sum_x*sum_x+sum_y*sum_y);
        cout << k_norm << flush;
        if (k_norm > 1)
            cerr << "Exceeded from Dijkstra" << endl;

    }

    if (Textures.size() > 0)
    {
        cout << endl;

        //cout     << setw(8)<< " " << setw(8) << "HH" << "\t" << setw(8)<<"HL" << "\t" << setw(8)<< "LH"  << endl;

        //cout     << setw(8)<< "VALUE" << "\t" << setw(8)<< WaveletHH.at(0).at(x,y) << "\t" << setw(8)<<WaveletHL.at(0).at(x,y)<< "\t" << setw(8)<< WaveletLH.at(0).at(x,y) << endl;

        cout     << setw(8)<< "TEXSPACE" << "\t";
        for(unsigned i=0; i < Textures.size();i++)
        {
            cout << setw(8)<< Textures.at(i).at<float>(x,y) << "\t";
        }
        cout << endl << endl;
    }

}


void Tool::doGPUSegmentationStepwise()
{
    GPUSegmentation(true);
}

void Tool::doGPUSegmentationConvergence()
{
    GPUSegmentation(false);
}


bool Tool::GPUSegmentation(bool stepwise)
{


    unsigned w = likely.at(0).rows;
    unsigned h = likely.at(0).cols;

    QImage im(w,h,QImage::Format_RGB32);

#ifndef CONST_GPU

    gpu.likely = new float[w*h*nr_labels];
    gpu.primal = new float[2*w*h*nr_labels];
    gpu.dual = new float[2*w*h*nr_labels];
    gpu.g = new float[w*h];

    gpu.temp = new float[w*h*nr_labels];
    gpu.labeling = new int[w*h];

    gpu.int_params = new int[INT_PARAMS];
    gpu.float_params = new float[FLOAT_PARAMS];
#endif

    gpu.int_params[NX] = w;
    gpu.int_params[NY] = h;
    gpu.int_params[NR_LABELS] = nr_labels;
    gpu.int_params[NR_SEG_IT] = seg_it;
    gpu.float_params[SEG_TAU] = seg_tau;
    gpu.float_params[SEG_ETA] = seg_eta;
    gpu.stepwise = stepwise;

    GTerm(false);

    for(unsigned x=0; x < w; x++)
        for(unsigned y=0; y < h; y++)
        {
            gpu.g[y*w+x] = G.at<float>(x,y);
        }

    for(unsigned l=0; l < nr_labels; l++)
        for(unsigned x=0; x < w; x++)
            for(unsigned y=0; y < h; y++)
                gpu.likely[w*h*l + y*w + x] = -likely.at(l).at<float>(x,y)/seg_lambda;

    int currIt=currentIt;
    if(currentIt > 0)
    {
        int c = w*h;
        for(unsigned x=0; x < w; x++)
            for(unsigned y=0; y < h; y++)
                for(unsigned l = 0; l < nr_labels; l++)
                {
                    gpu.primal[c*l + y*w+x] = primal.at(l).at<float>(x,y);
                    gpu.dual[2*c*l + y*w+x] = dual.at(l).at<float>(x,y);
                    gpu.dual[2*c*l + c +y*w+x] = dual.at(l).at<float>(x+w,y);
                }
    }


#ifdef CONST_GPU
    if(!gpu_segmentation(gpu, const_gpu,currIt))
#else
    if(!gpu_segmentation(gpu, currIt))
#endif
    {
        cerr << "Error in segmentation!" << endl;
        return false;
    }
    currentIt=currIt;

    labeling = Mat(w,h,CV_32F);
    primal.clear();
    dual.clear();
    for(unsigned l = 0; l < nr_labels; l++)
    {
        primal.push_back(Mat(w,h,CV_32F));
        dual.push_back(Mat(2*w,h,CV_32F));
    }


    vector<Mat> real_primal;
    for (int l=0; l < nr_labels;l++)
        real_primal.push_back(Mat(w,h,CV_32F));

    int c = w*h;
    for(unsigned x=0; x < w; x++)
        for(unsigned y=0; y < h; y++)
        {
            for(unsigned l = 0; l < nr_labels; l++)
            {
                primal.at(l).at<float>(x,y) =  gpu.primal[c*l     + y*w+x];
                real_primal.at(l).at<float>(x,y) = gpu.primal[c*nr_labels + c*l     + y*w+x];
                dual.at(l).at<float>(x,y) =    gpu.dual[2*c*l     + y*w+x];
                dual.at(l).at<float>(x+w,y) =  gpu.dual[2*c*l + c + y*w+x];
            }

            labeling.at<float>(x,y)= gpu.labeling[y*w+x];
        }

    // compute duality gap
    float p=0,d=0;
    for(unsigned x=0; x < w; x++)
        for(unsigned y=0; y < h; y++)
        {
            // primal part
            for(unsigned l=0; l < nr_labels;l++)
            {
                if(x<w-1)
                    p -= abs(real_primal.at(l).at<float>(x+1,y) -real_primal.at(l).at<float>(x,y));
                if(y<h-1)
                    p -= abs(real_primal.at(l).at<float>(x,y+1) -real_primal.at(l).at<float>(x,y));

                p+= likely.at(l).at<float>(x,y)*real_primal.at(l).at<float>(x,y);
            }

            // dual part
            float min_val=1000;
            for(unsigned l=0; l < nr_labels;l++)
            {
                float div=0;
                if(x>0)
                    div -= dual.at(l).at<float>(x-1,y);
                if(x<w-1)
                    div += dual.at(l).at<float>(x,y);
                if(y>0)
                    div -= dual.at(l).at<float>(x+w,y-1);
                if(y>h-1)
                    div += dual.at(l).at<float>(x+w,y);

                float val = likely.at(l).at<float>(x,y) - div;
                if (val < min_val)
                    min_val = val;
            }
            d += min_val;
        }
    //cerr << currentIt << " " <<  p << " " << d << " " << p-d << endl;

    Mat beauty = makeBeautifulSegmentation();
    QImage QIm = fromCVMat(beauty);
    out->setImage(QIm);


#ifndef CONST_GPU

    delete gpu.likely;
    delete gpu.labeling;
    delete gpu.primal;
    delete gpu.dual;
    delete gpu.temp;
    delete gpu.int_params;
    delete gpu.float_params;

#endif
    return true;

}

Mat Tool::makeBeautifulSegmentation()
{
    unsigned w = likely.at(0).rows;
    unsigned h = likely.at(0).cols;
    Mat ret = fromQimage(in->back);
    Mat b = Mat::zeros(h,w,CV_8U);
    for(unsigned x=0; x < w; x++)
        for(unsigned y=0; y < h; y++)
        {
            int l = labeling.at<float>(x,y);
            bool interface=false;
            if (x>0) interface |= l != labeling.at<float>(x-1,y);
            if (y>0) interface |= l != labeling.at<float>(x,y-1);
            if (x>0 && y>0 ) interface |= l != labeling.at<float>(x-1,y-1);

            if (interface)    b.at<uchar>(y,x) = 255;
            ret.at<Vec3f>(y,x) = ret.at<Vec3f>(y,x)*0.5 + getRgbOfLabel(l)*0.5;
        }

    cv::dilate(b,b,Mat());
    ret.setTo(Scalar(0,0,0),b);
    return ret;
}

bool Tool::Wavelet(bool show)
{

    readScribbles();

    int w = in->back.width();
    int h = in->back.height();
    Image *img = new StillImage(h,w);
    for(int i=0; i < w;i++)
        for (int j=0; j < h;j++)
            img->to(j,i,qGray(in->back.pixel(i,j)));



    /*
    When using some of the filters (Haar, Daub, Villa) on images with side lengths
    which are not powers of two, the precision is far from satisfactory.
    */

    FilterSet *flt = &FilterSet::filterFromString ((char*)WaveletFilter.toStdString().c_str());
    WaveletTransform *transform = new PyramidTransform (*img, *flt);
    try
    {
        // Decompose
        transform->analysis(NrWaveletSteps+1);
        //transform->image().write("wavelet.pgm", true);

        WaveletHH.resize(NrWaveletSteps);
        WaveletHL.resize(NrWaveletSteps);
        WaveletLH.resize(NrWaveletSteps);
        WaveletLL.resize(NrWaveletSteps);

        Textures.clear();
        for(unsigned i=0; i < (LDATexSpace? TexDim : NrWaveletSteps*6); i++)
            Textures.push_back(Mat(w,h,CV_32F));


        // Copy all the coefficients of all subbands
        int scale=1;
        for(unsigned i=0; i < NrWaveletSteps;i++ )
        {
            WaveletHH.at(i) = Mat::zeros(w,h,CV_32F);
            WaveletHL.at(i) = Mat::zeros(w,h,CV_32F);
            WaveletLH.at(i) = Mat::zeros(w,h,CV_32F);
            WaveletLL.at(i) = Mat::zeros(w,h,CV_32F);

            scale *= 2;

            Image *hh = transform->subband(HH,i+1)->scale(scale);
            Image *hl = transform->subband(HL,i+1)->scale(scale);
            Image *lh = transform->subband(LH,i+1)->scale(scale);
            Image *ll = transform->subband(LL,i+1)->scale(scale);

            // Cols and Rows are always dividable by 2, w and h not always, so mirror!!!
            for(int x=0;  x < hh->cols() ;x++)
                for (int y=0;y < hh->rows();y++)
                {

                    WaveletHH.at(i).at<float>(x,y) =hh->at(y,x);
                    WaveletHL.at(i).at<float>(x,y) =hl->at(y,x);
                    WaveletLH.at(i).at<float>(x,y) =lh->at(y,x);
                    WaveletLL.at(i).at<float>(x,y) =ll->at(y,x);
                }



            if(show && i == NrWaveletSteps-1)
            {
                int y, x, a, b;
                transform->where (LL, NrWaveletSteps, y, x, a, b);
                Image *HH = hh->clone();
                Image *HL = hl->clone();
                Image *LH = lh->clone();
                Image *LL = ll->clone();
                HH->beautify (a, b, y, x);
                HL->beautify (a, b, y, x);
                LH->beautify (a, b, y, x);
                LL->beautify (a, b, y, x);

                QImage wavelet(hh->cols()*2,hh->rows()*2, QImage::Format_RGB32);
                int val;
                for(int i=0; i < hh->cols();i++)
                    for (int j=0;j <  hh->rows();j++)
                    {
                        // H = LH, V = HL, D = HH
                        val = LH->at(j,i);
                        wavelet.setPixel(i,j, QColor(val,val,val).rgb());
                        val = HL->at(j,i);
                        wavelet.setPixel(i+hh->cols(),j, QColor(val,val,val).rgb());
                        val = HH->at(j,i);
                        wavelet.setPixel(i,j+hh->rows(), QColor(val,val,val).rgb());
                        val = LL->at(j,i);
                        wavelet.setPixel(i+hh->cols(),j+hh->rows(), QColor(val,val,val).rgb());
                    }


                delete HH;
                delete HL;
                delete LH;
                delete LL;
                out->setImage(wavelet);

            }

            delete hh;
            delete hl;
            delete lh;
            delete ll;


        }


#ifndef CONST_GPU

        gpu.hh = new float[w*h*NrWaveletSteps];
        gpu.hl = new float[w*h*NrWaveletSteps];
        gpu.lh = new float[w*h*NrWaveletSteps];
        gpu.hh_avg = new float[w*h*NrWaveletSteps];
        gpu.hl_avg = new float[w*h*NrWaveletSteps];
        gpu.lh_avg = new float[w*h*NrWaveletSteps];
        gpu.hh_stddev = new float[w*h*NrWaveletSteps];
        gpu.hl_stddev = new float[w*h*NrWaveletSteps];
        gpu.lh_stddev = new float[w*h*NrWaveletSteps];
        gpu.int_params = new int[INT_PARAMS];

#endif

        gpu.int_params[NX] = w;
        gpu.int_params[NY] = h;
        gpu.int_params[WAVELET_STEPS] = NrWaveletSteps;
        gpu.int_params[WAVELET_WIN_SIZE] = TexWinSize/2.0f;


        for(unsigned n=0; n < NrWaveletSteps;n++ )
            for(int x=0; x < w; x++)
                for(int y=0; y < h; y++)
                {
                    int pos = w*h*n + y*w +x;
                    gpu.hh[pos] = (float)WaveletHH.at(n).at<float>(x,y);
                    gpu.hl[pos] = (float)WaveletHL.at(n).at<float>(x,y);
                    gpu.lh[pos] = (float)WaveletLH.at(n).at<float>(x,y);
                }


#ifdef CONST_GPU
        gpu_wavelet(gpu,const_gpu);
#else
        gpu_wavelet(gpu);
#endif
        /*

        if(LDATexSpace)
        {


            // Fill the matrices with texture values for each label
            mat Call = zeros(scribbles_orig.size(),NrWaveletSteps*6);
            vector<mat> mats;
            for(unsigned i=0; i < nr_labels;i++)
                mats.push_back(zeros(label_count_orig[i],NrWaveletSteps*6));
            vec off = zeros(nr_labels+1);
            for(unsigned i=0; i < scribbles_orig.size(); i++)
            {
                int label = scribbles_orig.at(i).label;
                int scribble_pos = scribbles_orig.at(i).y*w +scribbles_orig.at(i).x;
                for(unsigned n=0; n < NrWaveletSteps;n++)
                {
                    int pos = w*h*n + scribble_pos;
                    int o = off.at(label);
                    mats.at(label)(o,n*6+0) = gpu.hh_avg[pos];
                    mats.at(label)(o,n*6+1) = gpu.hl_avg[pos];
                    mats.at(label)(o,n*6+2) = gpu.lh_avg[pos];
                    mats.at(label)(o,n*6+3) = gpu.hh_stddev[pos];
                    mats.at(label)(o,n*6+4) = gpu.hl_stddev[pos];
                    mats.at(label)(o,n*6+5) = gpu.lh_stddev[pos];

                    o = off.at(nr_labels);
                    Call(o,n*6+0) = gpu.hh_avg[pos];
                    Call(o,n*6+1) = gpu.hl_avg[pos];
                    Call(o,n*6+2) = gpu.lh_avg[pos];
                    Call(o,n*6+3) = gpu.hh_stddev[pos];
                    Call(o,n*6+4) = gpu.hl_stddev[pos];
                    Call(o,n*6+5) = gpu.lh_stddev[pos];
                }
                off.at(label)++;
                off.at(nr_labels)++;
            }


            mat LDATexMatrix;
            if (!OLDA(LDATexMatrix,Call,mats, scribbles_orig,label_count_orig)) return false;

            // Project into TexDim Space
            vec point(NrWaveletSteps*6);
            for(int x=0; x < w; x++)
                for(int y=0; y < h; y++)
                    for(unsigned n = 0; n < NrWaveletSteps;n++)
                        for(unsigned t=0; t < TexDim;t++)
                        {
                            int pos = w*h*n + y*w+x;
                            point(n*6 +0) = gpu.hh_avg[pos];
                            point(n*6 +1) = gpu.hl_avg[pos];
                            point(n*6 +2) = gpu.lh_avg[pos];
                            point(n*6 +3) = gpu.hh_stddev[pos];
                            point(n*6 +4) = gpu.hl_stddev[pos];
                            point(n*6 +5) = gpu.lh_stddev[pos];
                            Textures.at(t).at(x,y) = as_scalar(LDATexMatrix.row(t)*point);
                        }

        }
        else
        {
            for(int x=0; x < w; x++)
                for(int y=0; y < h; y++)
                    for(unsigned n=0; n < NrWaveletSteps;n++)
                    {
                        int pos = w*h*n + y*w+x;
                        Textures.at(n*6+0).at(x,y) = gpu.hh_avg[pos];
                        Textures.at(n*6+1).at(x,y) = gpu.hl_avg[pos];
                        Textures.at(n*6+2).at(x,y) = gpu.lh_avg[pos];
                        Textures.at(n*6+3).at(x,y) = gpu.hh_stddev[pos];
                        Textures.at(n*6+4).at(x,y) = gpu.hl_stddev[pos];
                        Textures.at(n*6+5).at(x,y) = gpu.lh_stddev[pos];

                    }

            // Visualize each channel if needed
            if(TexDim < 6*NrWaveletSteps+1 && show)
            {
                QImage wavelet(w,h, QImage::Format_RGB32);
                float m = max(max(abs(Textures.at(TexDim-1))));
                for(int i=0; i < w;i++)
                    for (int j=0;j < h;j++)
                    {
                        int val = abs(255.0f*Textures.at(TexDim-1).at(i,j) /m);
                        wavelet.setPixel(i,j,QColor(val,val,val).rgb());
                    }

                out->setImage(wavelet);
            }
        }
        */

        TextureDone=true;
        delete img;
        delete transform;

#ifndef CONST_GPU

        delete gpu.lh;
        delete gpu.hh;
        delete gpu.hl;
        delete gpu.lh_avg;
        delete gpu.hh_avg;
        delete gpu.hl_avg;
        delete gpu.hh_stddev;
        delete gpu.hl_stddev;
        delete gpu.lh_stddev;
        delete gpu.int_params;

#endif

        return true;

    }
    catch (const std::exception &error)
    {
        QMessageBox msgBox;
        msgBox.addButton(QMessageBox::Ok);
        QString lol("Wavelet error in analysis: ");
        lol.append(error.what());
        msgBox.setText(lol);
        msgBox.exec();
        return false;
    }



}

bool Tool::LDA(Mat &Projector,Mat Call, vector<Mat> Ceach)
{

    unsigned dim = Call.cols;
    Mat eigvec,temp;

    /*
    Projector = mat(dim,dim);
    mat Sw = zeros(dim,dim),Sb = cov(Call,1);  // Norm type=1
    for(unsigned i=0; i < nr_labels;i++)
        Sw = Sw + cov(Ceach.at(i),1);
    cx_vec eigval;


    if(!eig_gen(eigval,temp, eigvec, inv(Sw)*Sb)) return false;
*/


    /*
    // Do an index sort for the eigenvectors
    vector<double> eigen;
    vector<int> index;
    for(int n = 0; n < NrWaveletSteps*6;n++)
        eigen.push_back(eigval.at(n).real());
    sort(eigen.begin(),eigen.end());
    int currInd=NrWaveletSteps*6-1,counter=0;
    while(index.size() != NrWaveletSteps*6)
    {
        if (eigval.at(counter).real() == eigen.at(currInd))
        {
            index.push_back(counter);
            currInd--;
        }
        if (counter++ == NrWaveletSteps*6-1) counter=0;
    }

    // Bring the projection matrix into right order (descending eigenvalues)
    for(int i=0; i < NrWaveletSteps*6; i++)
    {
        int eig_i = index.at(i);
        for(int j=0; j < NrWaveletSteps*6; j++)
            LDATexMatrix.at(j,i) = eigvec.at(j,eig_i).real();
    }


    for(unsigned i=0; i < dim; i++)
        for(unsigned j=0; j < dim; j++)
            Projector.at<float>(i,j) = eigvec.at<float>(j,i);
            */

    return true;
}

bool Tool::OLDA(Mat &Projector,Mat Call, vector<Mat> Ceach,  vector<Scribble> &scrib, vector<unsigned> &count)
{

    /*
    // Compute all means
    unsigned dim = Call.n_cols;
    rowvec centroid = mean(Call);
    vector<rowvec> means(nr_labels);
    for(unsigned i=0; i < nr_labels;i++)
        means.at(i) = mean(Ceach.at(i));

    // Compute Ht
    mat Ht(dim,scrib.size());
    double factor = 1.0/(sqrt(scrib.size()));
    for(unsigned i=0; i < scrib.size(); i++)
        for(unsigned j=0; j < dim; j++)
            Ht.at(j,i) = (Call.at(i,j)-centroid.at(j))*factor;

    // Compute Hb
    mat Hb(dim,nr_labels);
    for(unsigned i=0; i < nr_labels; i++)
    {
        double label_factor = factor/count[i];
        for(unsigned j=0; j < dim; j++)
            Hb.at(j,i) = (means.at(i).at(j)-centroid.at(j))*label_factor;
    }

    mat U,temp,P,G;
    vec Sigma,nada;  // rank of St=t ?
    if(!svd(U,Sigma,temp,Ht)) return false;
    if(!svd(P,nada,temp,diagmat(Sigma).i() * U.t()*Hb))return false;
    if (!qr(G,temp,U*diagmat(Sigma).i()*P))return false;
    Projector = trans(G);
*/
    return true;
}

void Tool::doWavelet()
{
    Wavelet(true);
}


void Tool::doContSeg()
{
    QImage im(in->back);
    theta.clear();
    eta.clear();
    xi.clear();
}

void Tool::loadCImgData()
{
}

void Tool::doTest()
{
    LoadIcgBenchFile("icg/all/image_0002_9050.gt");
    doKernelEstimation(false);
}

void Tool::doDumpImage()
{    
    out->back.save("dump_right.png");

}


void Tool::doDiceScoreWithOutput()
{
    cerr << "Score: " << doDiceScore() <<endl;
}

double Tool::doDiceScore()
{

    unsigned w = out->back.width();
    unsigned h = out->back.height();

    if(groundtruth.empty())
    {
        cout << "No GT file loaded..." << endl;
        return 0;
    }
    int border = TexWinSize/2;
    border=0;

    float score=0;
    for (unsigned label = 0; label < nr_labels; label++){
        int area_ab = 0;
        int area_a = 0;
        int area_b = 0;
        for (unsigned int x = border; x < w-border; x++)
            for (unsigned int y = border; y < h-border; y++){
                unsigned l_a = labeling.at<float>(x,y);
                unsigned l_b =  groundtruth.at<float>(x,y);
                if (label == l_a)area_a++;
                if (label == l_b)
                {
                    area_b++;
                    if (label == l_a) area_ab++;
                }
            }
        score += 2.0 * area_ab / (area_a + area_b);
    }

    score /= nr_labels;
    return score;

}

void Tool::LoadIcgBenchFile(QString file)
{
    try
    {
        IcgBench::IcgBenchFileIO image(file.toStdString());
        IcgBench::LabelImage* gt = image.getLabels();

        string path = "icg/images/" + image.getFileName();

        in->loadImage(QString(path.c_str()));

        unsigned w = in->back.width();
        unsigned h = in->back.height();

        if(gt->width() != w ||gt->height() != h)
        {
            cout << "WRONG DIMENSIONS!" << endl;
            return;
        }

        groundtruth = Mat(w,h,CV_32F);
        for (unsigned int x = 0; x < w; x++)
            for (unsigned int y = 0; y < h; y++)
                groundtruth.at<float>(x,y) = gt->get(x,y);

        vector<IcgBench::Seed> seeds = image.getSeeds();
        int brush = BrushSize;

        for (IcgBench::Seed s : seeds)
            in->setIcgLabel(s.x,s.y,s.label,brush, groundtruth);


        readScribbles();
        delete gt;
        in->update();

    }
    catch(...)
    {
        cout << "FAIL!" <<endl;
    }
}

void Tool::doIcgBench()
{

    QString file = QFileDialog::getOpenFileName(
                0, tr("Open GT"), "../../..", tr("GT Files (*.gt)"));

    LoadIcgBenchFile(file);

}

void Tool::doBenchIcgFile()
{
    QString file = QFileDialog::getOpenFileName(
                0, tr("Open GT"), "../../..", tr("GT Files (*.gt)"));

    cout << BenchIcgFile(file) << endl;
}

double Tool::BenchIcgFile(QString file)
{
    LoadIcgBenchFile(file);

    double max=0;
    float best_sigma,best_beta,best_alpha,best_lambda;
    int best_steps, best_winsize, best_texdim;
    int currRun=0;

    if(!ColorLDA(false,1))
    {
        cerr << "ColorLDA FAIL" << endl;
        return 0;
    }

    for(int steps=1; steps < 2; steps++)
        for(int winsize=5; winsize < 6; winsize += 3)
            for(int texdim=5; texdim < 6; texdim += 2)
            {
                TexWinSize = winsize;
                TexDim = texdim;
                NrWaveletSteps = steps;

                if (!Wavelet(false))
                {
                    cerr << "Wavelet FAIL" << endl;
                    return 0;
                }

                for(float sigma=0.05f; sigma < 0.25f; sigma += 0.05f)
                    for(float beta=0.05f; beta < 0.25f; beta += 0.05f)
                        for(float alpha=0.1f; alpha < 5; alpha += 0.25f)
                            for(float lambda=1; lambda < 5; lambda += 1)
                            {

                                kernel_sigma = sigma;
                                kernel_beta = beta;
                                kernel_alpha = alpha;
                                seg_lambda = lambda;

                                if (!doKernelEstimation(false))
                                {
                                    cerr << "Kernel FAIL" << endl;
                                    return 0;
                                }
                                if(!GPUSegmentation(false))
                                {
                                    cerr << "Segmentation FAIL" << endl;
                                    return 0;
                                }
                                cout << currRun++ << " ";
                                double score = doDiceScore();
                                if (score > max)
                                {
                                    max = score;
                                    best_sigma = sigma;
                                    best_beta = beta;
                                    best_alpha = alpha;
                                    best_lambda = lambda;
                                    best_steps = steps;
                                    best_winsize = winsize;
                                    best_texdim = texdim;
                                }

                            }
            }
    cout << "Best score: " << max << "   " << best_sigma << " " << best_alpha<< " " << best_beta<< " "
         << best_lambda<< " " << best_steps << " " << best_winsize << " " << best_texdim << endl;
    return max;
}

void Tool::doBenchmark()
{

    QString directory ;
    directory = QFileDialog::getExistingDirectory(0, tr("Open Benchmark directory"), ".");
    doBenchmark(directory.toStdString());

}

void Tool::doBenchmark(string dir)
{

#ifndef __APPLE__

    double total_score=0;
    int number_tests=0;
    vector<string> files;
    DIR *pDIR;
    struct dirent *entry;
    if( (pDIR=opendir(dir.c_str())) ){
        while((entry = readdir(pDIR))){
            string str(entry->d_name);
            if( str.find(".gt")!=string::npos)
                files.push_back(str);
        }
        closedir(pDIR);
    }
    else
    {
        cerr << "Couldn't open directory!" << endl;
        return;
    }


    time_t t = time(0);
    tm *now = localtime(&t);
    stringstream timestring;
    timestring << "/" << now->tm_hour << "-" << now->tm_min << "__" << now->tm_mday << "-" << now->tm_mon << ".txt";


    for(vector<string>::iterator it = files.begin(); it != files.end(); ++it)
    {
        string path = dir.append("/")+*it;
        LoadIcgBenchFile(QString(path.c_str()));
        ColorLDA(false,ColorMode==2);
        Wavelet(false);
        doKernelEstimation(false);
        GPUSegmentation(false);
        double score = doDiceScore();
        total_score += score;
        number_tests++;
    }

    double result = total_score/number_tests;
    stringstream r;
    r << result;
    cerr << "   Total score: " << result << endl;
    ofstream output(("test/" + r.str()).c_str());
    output << "space " << kernel_alpha << " " << kernel_delta << " " << Anisotropy << endl;
    output << "color " << kernel_sigma <<  " " << ColorMode << endl;
    output << "tex " << kernel_beta <<  " " << WaveletFilter.toStdString()
           << " " << LDATexSpace << " " << TexDim <<  " " << NrWaveletSteps << endl;
    output << "segmentation " << seg_tau << " " << seg_lambda << " " << GMode << " " << seg_eta << endl;
    output << "brush " << BrushSize << endl;
    output.close();
#endif

}


void Tool::doFull()
{
    ColorLDA(false,ColorMode==2);
    Wavelet(false);
    doKernelEstimation(false);
    GPUSegmentation(false);
    doDiceScoreWithOutput();
}

