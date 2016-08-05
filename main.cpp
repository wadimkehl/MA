
#include <QApplication>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QLabel>

#include <scribblearea.h>
#include <tool.h>
#include <sstream>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    Tool *tool = new Tool();

    bool doBench=false;
    string benchDir;

    if (argc > 2) {
        for (int i = 1; i < argc-1; i=i+2) {

            stringstream name(argv[i]);
            stringstream value(argv[i+1]);

            if(name.str() == "-alpha") value >> tool->kernel_alpha;
            if(name.str() == "-beta")  value >> tool->kernel_beta ;
            if(name.str() == "-sigma")  value >> tool->kernel_sigma ;
            if(name.str() == "-brush")  value >> tool->BrushSize ;
            if(name.str() == "-lambda")  value >> tool->seg_lambda ;
            if(name.str() == "-colormode")  value >> tool->ColorMode ;
            if(name.str() == "-ldatex")  value >> tool->LDATexSpace;
            if(name.str() == "-texdim")  value >> tool->TexDim ;
            if(name.str() == "-texsteps")  value >> tool->NrWaveletSteps;
            if(name.str() == "-auto")  value >> tool->AutoParam;
            if(name.str() == "-iso")  value >> tool->Anisotropy;

            if(name.str() == "-bench")
            {
                doBench=true;
                 value >> benchDir;
            }

        }
    }



    // first box
    QPushButton *testButton = new QPushButton("Test");
    QPushButton *loadButton = new QPushButton("Load Image");  
    QPushButton *cimgButton = new QPushButton("Load Dataterm");
    QPushButton *icgButton = new QPushButton("Load IcgBench");
    QPushButton *benchButton = new QPushButton("Benchmark!");
    QPushButton *benchFileButton = new QPushButton("Bench File!");
    QPushButton *dumpLButton = new QPushButton("Dump Image");
    QVBoxLayout *vbox0 = new QVBoxLayout;
    vbox0->addWidget(testButton);
    vbox0->addWidget(loadButton);
    vbox0->addWidget(cimgButton);
    vbox0->addWidget(icgButton);
    vbox0->addWidget(benchButton);
    vbox0->addWidget(benchFileButton);

    vbox0->addWidget(dumpLButton);

    QGroupBox *groupBox0 = new QGroupBox();
    groupBox0->setLayout(vbox0);



    QPushButton *loadScribblesButton = new QPushButton("Load");
    QPushButton *saveScribblesButton = new QPushButton("Save");
    QPushButton *clearScribblesButton = new QPushButton("Clear");
    QSpinBox *labelSpinBox = new QSpinBox;
    labelSpinBox->setRange(0, 9);
    labelSpinBox->setSingleStep(1);
    labelSpinBox->setValue(0);
    QSpinBox *brushSpinBox = new QSpinBox;
    brushSpinBox->setRange(1, 15);
    brushSpinBox->setSingleStep(1);
    brushSpinBox->setValue(1);
    QComboBox *aniCombo = new QComboBox();
    aniCombo->addItem("Iso");
    aniCombo->addItem("Maha");
    aniCombo->addItem("NormEuc");
    QVBoxLayout *scribbleLayout = new QVBoxLayout();
    scribbleLayout->addWidget(loadScribblesButton);
    scribbleLayout->addWidget(saveScribblesButton);
    scribbleLayout->addWidget(clearScribblesButton);
    //scribbleLayout->addWidget(scribbleLabel1);
    scribbleLayout->addWidget(labelSpinBox);
    //scribbleLayout->addWidget(scribbleLabel2);
    scribbleLayout->addWidget(brushSpinBox);
    scribbleLayout->addWidget(aniCombo);
    QGroupBox *scribbleBox = new QGroupBox("Scribbles");
    scribbleBox->setLayout(scribbleLayout);


    QPushButton *ldaButton = new QPushButton("LDA");
    QPushButton *oldaButton = new QPushButton("OLDA");
    QComboBox *colorCombo = new QComboBox();
    colorCombo->addItem("RGB");
    colorCombo->addItem("LDA");
    colorCombo->addItem("OLDA");
    colorCombo->setCurrentIndex(tool->ColorMode);
    QSpinBox *eigenSpinBox = new QSpinBox;
    eigenSpinBox->setRange(0, 4);
    eigenSpinBox->setSingleStep(1);
    eigenSpinBox->setValue(0);
    QGroupBox *ldaBox = new QGroupBox("Color");
    QVBoxLayout *ldaLayout = new QVBoxLayout;
    ldaLayout->addWidget(ldaButton);
    ldaLayout->addWidget(oldaButton);
    ldaLayout->addWidget(eigenSpinBox);
    ldaLayout->addWidget(colorCombo);
    ldaBox->setLayout(ldaLayout);

    QComboBox *waveletFilter = new QComboBox();
    waveletFilter->addItem("Haar");
    waveletFilter->addItem("Daub4");
    waveletFilter->addItem("Daub6");
    waveletFilter->addItem("Daub8");
    waveletFilter->addItem("Antonini");
    waveletFilter->addItem("Brislawn");
    waveletFilter->addItem("Villa1");
    waveletFilter->addItem("Villa2");
    waveletFilter->addItem("Villa3");
    waveletFilter->addItem("Villa4");
    waveletFilter->addItem("Villa5");
    waveletFilter->addItem("Villa6");
    waveletFilter->addItem("Odegard");
    waveletFilter->setCurrentIndex(3);
    tool->setWaveletFilter(QString("Daub8"));
    QPushButton *waveletButton = new QPushButton("Do!");
    QCheckBox *ldaTexCheckBox = new QCheckBox("LDA?");
    ldaTexCheckBox->setChecked(tool->LDATexSpace);
    QSpinBox *waveletSpinBox = new QSpinBox;
    waveletSpinBox->setRange(1, 5);
    waveletSpinBox->setSingleStep(1);
    waveletSpinBox->setValue(tool->NrWaveletSteps);
    QSpinBox *texdimSpinBox = new QSpinBox;
    texdimSpinBox->setRange(1, 36);
    texdimSpinBox->setSingleStep(1);
    texdimSpinBox->setValue(tool->TexDim);
    QSpinBox *texwinSpinBox = new QSpinBox;
    texwinSpinBox->setRange(1, 30);
    texwinSpinBox->setSingleStep(1);
    texwinSpinBox->setValue(tool->TexWinSize);

    QGroupBox *waveletBox = new QGroupBox("Wavelets");
    QVBoxLayout *waveletLayout = new QVBoxLayout;
    waveletLayout->addWidget(waveletFilter);
    waveletLayout->addWidget(waveletSpinBox);
    waveletLayout->addWidget(texwinSpinBox);
    waveletLayout->addWidget(waveletButton);
    waveletLayout->addWidget(ldaTexCheckBox);
    waveletLayout->addWidget(texdimSpinBox);
    waveletBox->setLayout(waveletLayout);



    QComboBox *gMode = new QComboBox();
    gMode->addItem("Off");
    gMode->addItem("Color");
    gMode->addItem("Texture");
    gMode->setCurrentIndex(0);
    QDoubleSpinBox *etaSpinBox = new QDoubleSpinBox;
    etaSpinBox->setRange(0, 10);
    etaSpinBox->setDecimals(4);
    etaSpinBox->setSingleStep(0.05);
    etaSpinBox->setValue(tool->seg_eta);
    QPushButton *gButton = new QPushButton("Show");
    QPushButton *LBPButton = new QPushButton("LBP");
    QVBoxLayout *gLayout = new QVBoxLayout();
    gLayout->addWidget(gMode);
    gLayout->addWidget(etaSpinBox);
    gLayout->addWidget(gButton);
    gLayout->addWidget(LBPButton);

    QGroupBox *gBox = new QGroupBox("Weighted TV");
    gBox->setLayout(gLayout);



    QCheckBox *colorCheckBox = new QCheckBox("Color? "  + QString::fromLocal8Bit("σ") + ":");
    colorCheckBox->setChecked(true);
    QDoubleSpinBox *sigmaSpinBox = new QDoubleSpinBox;
    sigmaSpinBox->setRange(0, 10000);
    sigmaSpinBox->setDecimals(2);
    sigmaSpinBox->setSingleStep(0.1);
    sigmaSpinBox->setValue(tool->kernel_sigma);
    QHBoxLayout *colorlayout = new QHBoxLayout;
    colorlayout->addWidget(colorCheckBox);
    colorlayout->addWidget(sigmaSpinBox);
    QGroupBox *colorBox = new QGroupBox();
    colorBox->setLayout(colorlayout);

    QCheckBox *spaceCheckBox = new QCheckBox("Space? " + QString::fromLocal8Bit("α") + ":");
    spaceCheckBox->setChecked(true);
    QDoubleSpinBox *alphaSpinBox = new QDoubleSpinBox;
    alphaSpinBox->setRange(0, 10000);
    alphaSpinBox->setDecimals(2);
    alphaSpinBox->setSingleStep(0.1);
    alphaSpinBox->setValue(tool->kernel_alpha);
    QHBoxLayout *spacelayout = new QHBoxLayout;
    spacelayout->addWidget(spaceCheckBox);
    spacelayout->addWidget(alphaSpinBox);
    QGroupBox *spaceBox = new QGroupBox();
    spaceBox->setLayout(spacelayout);

    QCheckBox *textureCheckBox = new QCheckBox("Texture? "  + QString::fromLocal8Bit("β") + ":");
    textureCheckBox->setChecked(true);
    QDoubleSpinBox *betaSpinBox = new QDoubleSpinBox;
    betaSpinBox->setRange(0, 10000);
    betaSpinBox->setDecimals(2);
    betaSpinBox->setSingleStep(0.1);
    betaSpinBox->setValue(tool->kernel_beta);
    QHBoxLayout *texturelayout = new QHBoxLayout;
    texturelayout->addWidget(textureCheckBox);
    texturelayout->addWidget(betaSpinBox);
    QGroupBox *textureBox = new QGroupBox();
    textureBox->setLayout(texturelayout);

    QCheckBox *estimateCheckBox = new QCheckBox("GPU");
    estimateCheckBox->setChecked(true);
    QCheckBox *paramCheckBox = new QCheckBox("AutoParam");
    QPushButton *estimateButton = new QPushButton("Estimation");
    QPushButton *dumpEstimateButton = new QPushButton("Dump");


    QHBoxLayout *kernelbox = new QHBoxLayout;
    kernelbox->addWidget(colorCheckBox);
    kernelbox->addWidget(sigmaSpinBox);

    kernelbox->addWidget(spaceCheckBox);
    kernelbox->addWidget(alphaSpinBox);

    kernelbox->addWidget(textureCheckBox);
    kernelbox->addWidget(betaSpinBox);

   // kernelbox->addWidget(estimateCheckBox);
    kernelbox->addWidget(paramCheckBox);
    kernelbox->addWidget(estimateButton);
    kernelbox->addWidget(dumpEstimateButton);
    QGroupBox *kernelb = new QGroupBox();
    kernelb->setTitle("Kernel Density Estimation");
    kernelb->setLayout(kernelbox);


    QHBoxLayout *segbox = new QHBoxLayout;
    QLabel *tauLabel = new QLabel(QString::fromLocal8Bit("τ") + ":");
    QDoubleSpinBox *tauSpinBox = new QDoubleSpinBox;
    tauSpinBox->setRange(0, 1);
    tauSpinBox->setDecimals(4);
    tauSpinBox->setSingleStep(0.1);
    tauSpinBox->setValue(0.5);
    QLabel *lambdaLabel = new QLabel(QString::fromLocal8Bit("λ") + ":");
    QDoubleSpinBox *lambdaSpinBox = new QDoubleSpinBox;
    lambdaSpinBox->setRange(0, 100);
    lambdaSpinBox->setDecimals(4);
    lambdaSpinBox->setSingleStep(0.1);
    lambdaSpinBox->setValue(tool->seg_lambda);
    QPushButton *allButton = new QPushButton("Do all");
    QPushButton *varButton = new QPushButton("Stepwise");
    QPushButton *convergeButton = new QPushButton("Convergence");
    QPushButton *diceButton = new QPushButton("Dice Score");
    QPushButton *dumpButton = new QPushButton("Dump Image");

    segbox->addWidget(tauLabel);
    segbox->addWidget(tauSpinBox);
    segbox->addWidget(lambdaLabel);
    segbox->addWidget(lambdaSpinBox);
    segbox->addWidget(allButton);
    segbox->addWidget(varButton);
    segbox->addWidget(convergeButton);
    segbox->addWidget(diceButton);
    segbox->addWidget(dumpButton);

    QGroupBox *segb = new QGroupBox();
    segb->setLayout(segbox);
    segb->setTitle("Segmentation");



    QVBoxLayout *testlayout = new QVBoxLayout;
    testlayout->addWidget(kernelb);
    testlayout->addWidget(segb);
    QGroupBox *testbox = new QGroupBox();
    testbox->setLayout(testlayout);



    QHBoxLayout *grouplayout = new QHBoxLayout;
    grouplayout->addWidget(groupBox0);
    grouplayout->addWidget(scribbleBox);
    grouplayout->addWidget(ldaBox);
    grouplayout->addWidget(waveletBox);
    grouplayout->addWidget(gBox);
    grouplayout->addWidget(testbox);
    QGroupBox *groupBox = new QGroupBox();
    groupBox->setLayout(grouplayout);
    groupBox->setGeometry(0,0,1000,200);


    QHBoxLayout *viewlayout= new QHBoxLayout;
    ScribbleArea *scribbleArea = new ScribbleArea();
    ScribbleArea *gView = new ScribbleArea();
    viewlayout->addWidget(scribbleArea);
    viewlayout->addWidget(gView);
    scribbleArea->setGeometry(0,0,500,500);
    gView->setGeometry(0,0,500,500);
    gView->readOnly = true;
    tool->in = scribbleArea;
    tool->out = gView;
    gView->tool = tool;
    scribbleArea->tool = tool;
    tool->setBrushSize(tool->BrushSize);
    QGroupBox *viewBox = new QGroupBox();
    viewBox->setLayout(viewlayout);


    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(groupBox);
    layout->addWidget(viewBox);
    QWidget window;
    window.setLayout(layout);
    window.show();


    QObject::connect(testButton, SIGNAL(clicked()), tool, SLOT( doTest()));
    QObject::connect(loadButton, SIGNAL(clicked()), scribbleArea, SLOT(loadImageDialog()));
    QObject::connect(cimgButton, SIGNAL(clicked()), tool, SLOT(loadCImgData()));
    QObject::connect(icgButton, SIGNAL(clicked()), tool, SLOT(doIcgBench()));
    QObject::connect(benchButton, SIGNAL(clicked()), tool, SLOT(doBenchmark()));


    QObject::connect(dumpLButton, SIGNAL(clicked()), scribbleArea, SLOT(dumpImageWithScribbles()));


    QObject::connect(clearScribblesButton, SIGNAL(clicked()), scribbleArea, SLOT(clearScribbles()));
    QObject::connect(loadScribblesButton, SIGNAL(clicked()), scribbleArea, SLOT(loadScribblesDialog()));
    QObject::connect(saveScribblesButton, SIGNAL(clicked()), scribbleArea, SLOT(saveScribblesDialog()));
    QObject::connect(labelSpinBox, SIGNAL(valueChanged(int)), scribbleArea, SLOT(setCurrLabel(int)));
    QObject::connect(brushSpinBox, SIGNAL(valueChanged(int)), tool, SLOT(setBrushSize(int)));
    QObject::connect(aniCombo, SIGNAL(activated(int)), tool, SLOT(setAnisotropy(int)));

    QObject::connect(sigmaSpinBox, SIGNAL(valueChanged(double)), tool, SLOT(setSigma(double)));
    QObject::connect(alphaSpinBox, SIGNAL(valueChanged(double)), tool, SLOT(setAlpha(double)));
    QObject::connect(betaSpinBox, SIGNAL(valueChanged(double)), tool, SLOT(setBeta(double)));

    QObject::connect(colorCheckBox, SIGNAL(stateChanged(int)), tool, SLOT(setColor(int)));
    QObject::connect(spaceCheckBox, SIGNAL(stateChanged(int)), tool, SLOT(setSpace(int)));
    QObject::connect(textureCheckBox, SIGNAL(stateChanged(int)), tool, SLOT(setTexture(int)));


    QObject::connect(ldaButton, SIGNAL(clicked()), tool, SLOT(doColorLDA()));
    QObject::connect(oldaButton, SIGNAL(clicked()), tool, SLOT(doColorOLDA()));
    QObject::connect(colorCombo, SIGNAL(activated(int)), tool, SLOT(setColorMode(int)));
    QObject::connect(eigenSpinBox, SIGNAL(valueChanged(int)), tool, SLOT(setEigenVector(int)));


    QObject::connect(waveletButton, SIGNAL(clicked()), tool, SLOT(doWavelet()));
    QObject::connect(waveletFilter, SIGNAL(activated(QString)),tool, SLOT(setWaveletFilter(QString)));
    QObject::connect(ldaTexCheckBox, SIGNAL(stateChanged(int)), tool, SLOT(setTextureMode(int)));
    QObject::connect(texwinSpinBox, SIGNAL(valueChanged(int)), tool, SLOT(setTexWin(int)));
    QObject::connect(waveletSpinBox, SIGNAL(valueChanged(int)), tool, SLOT(setWaveletSteps(int)));
    QObject::connect(texdimSpinBox, SIGNAL(valueChanged(int)), tool, SLOT(setTexDim(int)));

    QObject::connect(gMode, SIGNAL(activated(int)), tool, SLOT(setGMode(int)));
    QObject::connect(gButton, SIGNAL(clicked()), tool, SLOT(doGTerm()));
    QObject::connect(LBPButton, SIGNAL(clicked()), tool, SLOT(doLBP()));

    QObject::connect(etaSpinBox, SIGNAL(valueChanged(double)), tool, SLOT(setEta(double)));




    QObject::connect(estimateCheckBox, SIGNAL(stateChanged(int)), tool, SLOT(setEstimateGPU(int)));
    QObject::connect(paramCheckBox, SIGNAL(stateChanged(int)), tool, SLOT(setAutoParam(int)));
    QObject::connect(estimateButton, SIGNAL(clicked()), tool, SLOT(doKernelEstimation()));
    QObject::connect(dumpEstimateButton, SIGNAL(clicked()), tool, SLOT(doDumpEstimation()));

    QObject::connect(tauSpinBox, SIGNAL(valueChanged(double)), tool, SLOT(setTau(double)));
    QObject::connect(lambdaSpinBox, SIGNAL(valueChanged(double)), tool, SLOT(setLambda(double)));

    QObject::connect(allButton, SIGNAL(clicked()), tool, SLOT(doFull()));
    QObject::connect(varButton, SIGNAL(clicked()), tool, SLOT(doGPUSegmentationStepwise()));
    QObject::connect(convergeButton, SIGNAL(clicked()), tool, SLOT(doGPUSegmentationConvergence()));

    QObject::connect(diceButton, SIGNAL(clicked()), tool, SLOT(doDiceScoreWithOutput()));
    QObject::connect(dumpButton, SIGNAL(clicked()), tool, SLOT(doDumpImage()));

    int result=0;
    if(!doBench)
        result = app.exec();
    else
        tool->doBenchmark(benchDir);

    return result;
}
