#ifndef SCRIBBLEAREA_H
#define SCRIBBLEAREA_H

#include <QGraphicsView>
#include <vector>

#include <opencv2/core.hpp>

using namespace cv;

Vec3f getRgbOfLabel(int label);

class Tool;
#include <tool.h>

class ScribbleArea : public QGraphicsView
{
    Q_OBJECT

public:
    ScribbleArea(QWidget *parent = 0);

    int **scribbles;
    int **scribbles_orig;

    int currLabel;
    bool readOnly;
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void paintEvent(QPaintEvent *event);

    void setLabel(int x, int y, int label);
    void setLabel(int x, int y, int label, int brushSize);
    void setIcgLabel(int x, int y, int label, int brushSize, Mat &gt);

    void removeLastLabel();

    Tool *tool;


    QSize sizeHint() const;
    bool scribbling;
    QImage back;
    int brushSize;

    QPoint lastClick;


public slots:
    void setCurrLabel(int l);
    void loadImageDialog();
    void loadImage(QString fileName);
    void loadScribbles(QString fileName);
    void loadScribblesDialog();
    void saveScribblesDialog();
    void setImage(QImage im);
    void clearScribbles();

    void dumpImageWithScribbles();


};

#endif // SCRIBBLEAREA_H
