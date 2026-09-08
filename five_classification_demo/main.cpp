#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("血液分析仪-血球五分类(肖珲贤：1207162512@qq.com)");
    w.show();
    return QApplication::exec();
}
