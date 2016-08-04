#ifndef TOOL_H
#define TOOL_H


#include <QObject>
#include <QImage>

#include <vector>
//#include <armadillo>
//#include <Wave.hh>
#include "icgbench.h"
#include "gpu.h"

#include <opencv2/core.hpp>


using namespace std;
//using namespace arma;
using namespace cv;

class ScribbleArea;
#include <scribblearea.h>

struct Scribble
{
    int label,x,y;
    Vec3f rgb;
};

class Tool : public QObject
{
    Q_OBJECT

public:
    Tool();
    ~Tool();

    ScribbleArea *in,*out;
    std::vector<Scribble> scribbles, scribbles_orig;
    vector<unsigned> label_count,label_count_orig;
    unsigned TexWinSize,NrWaveletSteps,BrushSize,currentIt, nr_labels,eig,TexDim;
    float kernel_alpha,kernel_sigma,kernel_beta, kernel_delta,seg_it, seg_tau,seg_lambda, seg_eta;

    vector<Mat> likely;

    vector<Mat> data_term;
    vector<Mat> theta, eta;
    vector<Mat> xi;

    Mat labeling, groundtruth, LDAimage, G;


    vector<Mat> Colors,Textures, primal, dual;
    vector<Mat> WaveletHH,WaveletLH,WaveletHL, WaveletLL;
    QString WaveletFilter;


    int ColorMode, GMode, Anisotropy;
    bool EstimateGPU;
    bool Color,Space,Texture;
    bool ColorDone, TextureDone;
    bool LDATexSpace, AutoParam;

    GPU_DATA gpu;
    GPU_DATA const_gpu;

    bool LDA(Mat &Projector,Mat Call, vector<Mat> Ceach);
    bool OLDA(Mat &Projector,Mat Call, vector<Mat> Ceach, vector<Scribble> &scrib, vector<unsigned> &count);

    void readScribbles();
    double DiceScore(QString file);
    bool ColorLDA(bool show,bool ortho);
    bool GTerm(bool show);
    bool Wavelet(bool show);
    bool GPUSegmentation(bool stepwise);
    bool LBP(int radius, int samples);
    bool ColorLBP(int radius, int samples);

    void projectDijkstra();
    void projectSimplex(vector<double> &in);
    void getKNN(int x, int y, vector<Scribble> out);
    void printPixelInformation(unsigned x, unsigned y);

    Mat makeBeautifulSegmentation();

    public slots:
        void doLBP();
        void doTest();
        void doColorLDA();
        void doColorOLDA();
        bool doKernelEstimation();
        bool doKernelEstimation(bool doAll);
        void doGTerm();
        void setEigenVector(int i);
        void setColor(int i);
        void setGMode(int i);
        void setSpace(int i);
        void setTexture(int i);
        void setSigma(double s);
        void setAlpha(double a);
        void setTau(double t);
        void setBeta(double b);
        void setEta(double e);
        void setLambda(double l);
        void setIterations(int it);
        void setColorMode(int i);
        void setTextureMode(int i);
        void setEstimateGPU(int i);
        void setAnisotropy(int i);
        void setWaveletSteps(int i);
        void setTexDim(int i);
        void setTexWin(int i);
        void doContSeg();
        void doGPUSegmentationStepwise();
        void doGPUSegmentationConvergence();
        void doDumpImage();
        void setAutoParam(int i);
        void setDelta(double d);

        void doBenchIcgFile();
        double BenchIcgFile(QString file);

        void doBenchmark();
        void doBenchmark(string dir);

        void doWavelet();
        void setWaveletFilter(QString str);
        void setBrushSize(int i);
        void doIcgBench();
        void LoadIcgBenchFile(QString file);
        double doDiceScore();
        void doDiceScoreWithOutput();
        void loadCImgData();

        void doDumpEstimation();
        void doFull();

};

#endif // TOOL_H

