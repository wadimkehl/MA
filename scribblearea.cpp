#include <QMouseEvent>
#include <QPen>
#include <QPainter>
#include <QFileDialog>
#include <QLayout>
#include <fstream>
#include <iostream>
#include "scribblearea.h"

using namespace std;
Vec3f getRgbOfLabel(int label)
{
    switch(label)
    {
    case 0:
        return Vec3f(1,0,0);
    case 1:
        return Vec3f(0,1,0);
    case 2:
        return Vec3f(0,0,1);
    case 3:
        return Vec3f(0,1,1);
    case 4:
        return Vec3f(1,1,1);
    case 5:
        return Vec3f(0.5,0.5,0.5);
    case 7:
        return Vec3f(0.5,1,1);
    case 8:
        return Vec3f(1,1,0.5);
    case 9:
        return Vec3f(1,0.5,1);

    }

    return Vec3f(0,0,0);
}

ScribbleArea::ScribbleArea(QWidget *parent)
    : QGraphicsView(parent)
{
    scribbling = false;
    readOnly = false;
    scribbles = new int*[1000];
    scribbles_orig = new int*[1000];

    for(int i =0; i < 1000; i++)
    {
        scribbles[i] = new int[1000];
        scribbles_orig[i] = new int[1000];
    }
    brushSize=10;
    currLabel = 0;
    clearScribbles();

}

void ScribbleArea::setLabel(int x, int y, int label)
{
    if(readOnly) return;

    for (int i = x-brushSize/2.0; i <= x+((brushSize/2.0));i++)
        for (int j = y-brushSize/2.0; j <= y+((brushSize/2.0));j++)
        {

            if (i >= back.size().width() || j >= back.size().height() || i < 0 || j<0)
                continue;

            scribbles[i][j] = label;
        }

    scribbles_orig[x][y] = label;

}


void ScribbleArea::dumpImageWithScribbles()
{
    /*
    QImage im(back);

    for(int i =0; i < 1000; i++)
        for(int j =0; j < 1000; j++)
            if(scribbles[i][j]> -1)
            {
                im.setPixel(i,j,getRgbOfLabel(scribbles[i][j]).rgb());
            }


    im.save("dump_left.png");
*/
}

void ScribbleArea::setLabel(int x, int y, int label, int brushSize)
{
    if(readOnly) return;

    scribbles_orig[x][y] = label;
    if(brushSize==1)
    {
        scribbles[x][y] = label;
        return;
    }

    for (int i = x-brushSize/2.0; i <= x+((brushSize/2.0));i++)
        for (int j = y-brushSize/2.0; j <= y+((brushSize/2.0));j++)
        {

            if (i >= back.size().width() || j >= back.size().height() || i < 0 || j<0)
                continue;

            scribbles[i][j] = label;
        }
}

void ScribbleArea::setIcgLabel(int x, int y, int label, int brushSize, Mat &gt)
{
    if(readOnly) return;

    scribbles_orig[x][y] = label;
    if(brushSize==1)
    {
        scribbles[x][y] = label;
        return;
    }

    for (int i = x-brushSize/2.0; i <= x+((brushSize/2.0));i++)
        for (int j = y-brushSize/2.0; j <= y+((brushSize/2.0));j++)
        {

            if (i >= back.size().width() || j >= back.size().height() || i < 0 || j<0)
                continue;

            if(gt.at<int>(i,j) != label) continue;


            scribbles[i][j] = label;
        }


}

void ScribbleArea::removeLastLabel()
{
    if(readOnly) return;
    for (int i = lastClick.x()-brushSize/2.0; i <= lastClick.x()+((brushSize/2.0));i++)
        for (int j = lastClick.y()-brushSize/2.0; j <= lastClick.y()+((brushSize/2.0));j++)
        {

            if (i >= back.size().width() || j >= back.size().height() || i < 0 || j<0)
                continue;

            scribbles[i][j] = -1;
            scribbles_orig[i][j] = -1;
        }
}

void ScribbleArea::mousePressEvent(QMouseEvent *event)
{
    scribbling = true;

    if (event->buttons() == Qt::LeftButton && scribbling)
    {

        if(readOnly)
        {
            tool->printPixelInformation(event->x(),event->y());
            return;
        }
        setLabel(event->x(),event->y(),currLabel);
        lastClick.setX(event->x());
        lastClick.setY(event->y());
        tool->ColorDone=false;
        tool->TextureDone = false;
    }
    else if(event->buttons() == Qt::RightButton)
    {
        //removeLastLabel();
        brushSize+=5;
        setLabel(event->x(),event->y(),-1);
        tool->ColorDone=false;
        tool->TextureDone = false;
        brushSize-=5;


    }
    update();
}

void ScribbleArea::mouseMoveEvent(QMouseEvent *event)
{
    //return;

    if (event->buttons() == Qt::LeftButton && scribbling)
    {

        if(readOnly)
        {
            tool->printPixelInformation(event->x(),event->y());
            return;
        }
        setLabel(event->x(),event->y(),currLabel);
        lastClick.setX(event->x());
        lastClick.setY(event->y());
        tool->ColorDone=false;
        tool->TextureDone = false;
    }
    else if(event->buttons() == Qt::RightButton)
    {
        //removeLastLabel();
        brushSize+=5;
        setLabel(event->x(),event->y(),-1);
        tool->ColorDone=false;
        tool->TextureDone = false;
        brushSize-=5;


    }
    update();
}


void ScribbleArea::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton && scribbling)
    {
        scribbling=false;
    }
    update();

}

void ScribbleArea::clearScribbles()
{

    for(int i =0; i < 1000; i++)
        for(int j =0; j < 1000; j++)
        {
            scribbles[i][j] = -1;
            scribbles_orig[i][j] = -1;
        }
    update();

}

void ScribbleArea::paintEvent(QPaintEvent *event)
{
    QPainter painter(viewport());
    painter.drawImage(0,0,back);
    QPen p;
    p.setWidth(1);
    event->accept();

    for(int i=0; i < 1000;i++)
        for(int j=0; j < 1000;j++)
        {
            int label = scribbles[i][j];
            if(label < 0) continue;
            Vec3f lab = getRgbOfLabel(label)*255;
            QRgb col = qRgb(lab[2],lab[1],lab[0]);
            p.setColor(col);
            painter.setPen(p);
            painter.drawPoint(i,j);
        }
    QGraphicsView::paintEvent(event);
}


QSize ScribbleArea::sizeHint() const
{
    QSize q(800,800);

    if(readOnly)
        return q;

    return back.size();
}



void ScribbleArea::setCurrLabel(int l)
{
    currLabel = l;
}

void ScribbleArea::setImage(QImage im)
{
    back = im;
    resize(back.size());
    updateGeometry();
}

void ScribbleArea::loadScribbles(QString fileName)
{

    ifstream file(fileName.toStdString().c_str());

    int oldBrush = brushSize;

    brushSize=1;
    while(file.good())
    {
        int x,y,label;
        file >> x;
        file >> y;
        file >> label;
        setLabel(x,y,label);
    }
    file.close();
    update();

    brushSize=oldBrush;

}

void ScribbleArea::loadScribblesDialog()
{

    QString fileName = QFileDialog::getOpenFileName(
                this,tr("Open Scribbles"), "d.scribbles", tr("Scribble Files (*.scribbles)"));

    loadScribbles(fileName);
}

void ScribbleArea::saveScribblesDialog()
{
    QString fileName = QFileDialog::getSaveFileName(
                this,tr("Save Scribbles"), "d.scribbles", tr("Scribble Files (*.scribbles)"));
    ofstream file(fileName.toStdString().c_str());

    for(int i=0; i < 1000;i++)
        for(int j=0; j < 1000;j++)
        {
            int label = scribbles[i][j];
            if(label < 0) continue;
            file << i << " " << j << " " << label << endl;
        }

    file.close();

}

void ScribbleArea::loadImage(QString fileName)
{
    back.load(fileName);
    resize(back.size());
    clearScribbles();
    tool->ColorDone=false;
    tool->TextureDone = false;
    updateGeometry();

}

void ScribbleArea::loadImageDialog()
{
    QString fileName = QFileDialog::getOpenFileName(
                this,tr("Open Image"), "d.JPG", tr("Image Files (*.png *.jpg *.bmp)"));

    loadImage(fileName);

}
