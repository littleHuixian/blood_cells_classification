#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QModelIndex>
#include <QStringList>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class QStandardItemModel;
class QGraphicsScene;
class QResizeEvent;
class BloodCellClassifier;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void onOpenFile();
    void onRefresh();
    void onCurrentChanged(const QModelIndex &current);
    void onPredict();

private:
    bool loadDirectory(const QString &dirPath, const QString &highlightFile);
    void addImageItem(const QString &filePath);
    void selectItem(const QString &filePath);
    void displayImage(const QString &filePath);
    void showFileInfo(const QString &filePath, const QSize &imageSize);
    static bool isImageFile(const QString &filePath);

    Ui::MainWindow *ui;
    QStandardItemModel *fileModel;
    QGraphicsScene *imageScene;
    BloodCellClassifier *classifier;
};
#endif // MAINWINDOW_H
