#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "bloodcell_classifier.h"

#include <QAction>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QItemSelectionModel>
#include <QListView>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QResizeEvent>
#include <QTextCharFormat>
#include <QTextCursor>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

const char *kImageFilter =
        "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif);;"
        "所有文件 (*.*)";

const char *kModelPath =
        "E:/PythonFiles/OpenCVCodes/blood_cells_classification/models/"
        "best_blood_cell_model_ir9.onnx";

const QStringList &imageNameFilters()
{
    static const QStringList filters = {
        QStringLiteral("*.png"),
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff"),
        QStringLiteral("*.webp"),
        QStringLiteral("*.gif")
    };
    return filters;
}

QImage cvMatToQImage(const cv::Mat &mat)
{
    if (mat.empty())
        return {};

    switch (mat.type()) {
    case CV_8UC4: {
        cv::Mat rgba;
        cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(rgba.data, rgba.cols, rgba.rows,
                      static_cast<qsizetype>(rgba.step),
                      QImage::Format_RGBA8888).copy();
    }
    case CV_8UC3: {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows,
                      static_cast<qsizetype>(rgb.step),
                      QImage::Format_RGB888).copy();
    }
    case CV_8UC1:
        return QImage(mat.data, mat.cols, mat.rows,
                      static_cast<qsizetype>(mat.step),
                      QImage::Format_Grayscale8).copy();
    default:
        return {};
    }
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , fileModel(new QStandardItemModel(this))
    , imageScene(new QGraphicsScene(this))
    , classifier(new BloodCellClassifier)
{
    ui->setupUi(this);

    // 列表只做单选浏览，不进入编辑
    ui->lvFileName->setModel(fileModel);
    ui->lvFileName->setIconSize(QSize(64, 64));
    ui->lvFileName->setSelectionMode(QAbstractItemView::SingleSelection);
    ui->lvFileName->setEditTriggers(QAbstractItemView::NoEditTriggers);

    ui->graphicsView->setScene(imageScene);
    ui->graphicsView->setRenderHints(QPainter::Antialiasing
                                     | QPainter::SmoothPixmapTransform);
    ui->graphicsView->setBackgroundBrush(Qt::black);

    connect(ui->btnOpenFile, &QPushButton::clicked,
            this, &MainWindow::onOpenFile);
    connect(ui->btnRefresh, &QPushButton::clicked,
            this, &MainWindow::onRefresh);
    connect(ui->lvFileName->selectionModel(), &QItemSelectionModel::currentChanged,
            this, &MainWindow::onCurrentChanged);
    connect(ui->action_prediction, &QAction::triggered,
            this, &MainWindow::onPredict);

    if (!classifier->initialize(QString::fromUtf8(kModelPath))) {
        ui->teShowInfo->setPlainText(
                QStringLiteral("模型初始化失败：%1")
                        .arg(classifier->lastError()));
    }

    // 启动时加载 exe 所在目录下的 test_images；不存在则自动创建
    const QString testImagesDir =
            QDir(QApplication::applicationDirPath() + QStringLiteral("/test_images"))
                    .absolutePath();
    QDir().mkpath(testImagesDir);
    ui->leFilePath->setText(QDir::toNativeSeparators(testImagesDir));
    loadDirectory(testImagesDir, QString());
}

MainWindow::~MainWindow()
{
    delete classifier;
    delete ui;
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    if (!imageScene->items().isEmpty()) {
        ui->graphicsView->fitInView(imageScene->itemsBoundingRect(),
                                    Qt::KeepAspectRatio);
    }
}

void MainWindow::onOpenFile()
{
    // 默认从当前路径所在目录开始选择
    QString startDir;
    const QString currentText = ui->leFilePath->text().trimmed();
    if (!currentText.isEmpty()) {
        const QFileInfo currentInfo(currentText);
        startDir = currentInfo.isDir()
                ? currentInfo.absoluteFilePath()
                : currentInfo.absolutePath();
    }
    if (startDir.isEmpty() || !QDir(startDir).exists()) {
        startDir = QStandardPaths::writableLocation(
                QStandardPaths::PicturesLocation);
        if (startDir.isEmpty() || !QDir(startDir).exists()) {
            startDir = QStandardPaths::writableLocation(
                    QStandardPaths::HomeLocation);
        }
    }

    const QString filePath = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("打开图片"),
            startDir,
            QString::fromUtf8(kImageFilter));
    if (filePath.isEmpty())
        return;

    ui->leFilePath->setText(QDir::toNativeSeparators(filePath));
    loadDirectory(QFileInfo(filePath).absolutePath(), filePath);
}

void MainWindow::onRefresh()
{
    const QString pathText = ui->leFilePath->text().trimmed();
    if (pathText.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("请先输入或选择文件路径。"));
        return;
    }

    const QFileInfo info(pathText);
    if (info.isFile()) {
        if (!isImageFile(pathText)) {
            QMessageBox::warning(this, QStringLiteral("提示"),
                                 QStringLiteral("请选择图片文件。"));
            return;
        }
        ui->leFilePath->setText(QDir::toNativeSeparators(info.absoluteFilePath()));
        loadDirectory(info.absolutePath(), info.absoluteFilePath());
    } else if (info.isDir()) {
        ui->leFilePath->setText(QDir::toNativeSeparators(info.absoluteFilePath()));
        loadDirectory(info.absoluteFilePath(), QString());
    } else {
        QMessageBox::warning(this, QStringLiteral("提示"),
                             QStringLiteral("路径不存在：%1").arg(pathText));
    }
}

void MainWindow::onCurrentChanged(const QModelIndex &current)
{
    if (!current.isValid())
        return;

    const QString filePath = current.data(Qt::UserRole).toString();
    if (filePath.isEmpty())
        return;

    ui->leFilePath->setText(QDir::toNativeSeparators(filePath));
    displayImage(filePath);
}

void MainWindow::onPredict()
{
    const QModelIndex current = ui->lvFileName->currentIndex();
    if (!current.isValid()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("请先选择一张图片。"));
        return;
    }

    const QString filePath = current.data(Qt::UserRole).toString();
    if (filePath.isEmpty())
        return;

    if (!classifier->isReady()) {
        QMessageBox::warning(this, QStringLiteral("警告"),
                             QStringLiteral("模型尚未初始化：%1")
                                     .arg(classifier->lastError()));
        return;
    }

    BloodCellClassifier::Prediction prediction;
    if (!classifier->predictImage(filePath, &prediction)) {
        QMessageBox::warning(this, QStringLiteral("错误"),
                             QStringLiteral("预测失败：%1")
                                     .arg(classifier->lastError()));
        return;
    }

    QTextCursor cursor(ui->teShowInfo->textCursor());
    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock();

    QTextCharFormat titleFormat = cursor.charFormat();
    titleFormat.setFontWeight(QFont::Bold);
    cursor.insertText(QStringLiteral("预测结果：\n"), titleFormat);

    QTextCharFormat normalFormat = cursor.charFormat();
    QTextCharFormat redFormat = normalFormat;
    redFormat.setForeground(Qt::red);
    redFormat.setFontWeight(QFont::Bold);

    cursor.insertText(
            QStringLiteral("预测类别：%1（%2）\n")
                    .arg(prediction.className, prediction.classNameZh),
            normalFormat);

    cursor.insertText(QStringLiteral("置信度："), normalFormat);
    cursor.insertText(QStringLiteral("%1%")
                              .arg(prediction.confidence * 100.0, 0, 'f', 2),
                      redFormat);
    cursor.insertText(QStringLiteral("\nTop-3：\n"), normalFormat);

    for (int i = 0; i < prediction.topEntries.size(); ++i) {
        const auto &entry = prediction.topEntries.at(i);
        cursor.insertText(
                QStringLiteral("%1. %2（%3）")
                        .arg(i + 1)
                        .arg(entry.className, entry.classNameZh),
                normalFormat);
        cursor.insertText(QStringLiteral("（"), normalFormat);
        cursor.insertText(QStringLiteral("%1%")
                                  .arg(entry.probability * 100.0, 0, 'f', 2),
                          redFormat);
        cursor.insertText(QStringLiteral("）\n"), normalFormat);
    }

    ui->teShowInfo->setTextCursor(cursor);
}

bool MainWindow::loadDirectory(const QString &dirPath, const QString &highlightFile)
{
    QDir dir(dirPath);
    if (!dir.exists())
        return false;

    fileModel->clear();

    const QFileInfoList entries =
            dir.entryInfoList(imageNameFilters(),
                              QDir::Files | QDir::Readable,
                              QDir::Name | QDir::IgnoreCase);

    QString selectedHighlight;
    for (const QFileInfo &entry : entries) {
        addImageItem(entry.absoluteFilePath());
        if (selectedHighlight.isEmpty() && !highlightFile.isEmpty()) {
            if (QDir::cleanPath(entry.absoluteFilePath())
                    == QDir::cleanPath(highlightFile)) {
                selectedHighlight = entry.absoluteFilePath();
            }
        }
    }

    if (fileModel->rowCount() == 0) {
        imageScene->clear();
        ui->teShowInfo->setPlainText(
                QStringLiteral("目录中没有找到可显示的图片：\n%1")
                        .arg(QDir::toNativeSeparators(dir.absolutePath())));
        return false;
    }

    ui->teShowInfo->setPlainText(
            QStringLiteral("目录：%1\n共 %2 张图片")
                    .arg(QDir::toNativeSeparators(dir.absolutePath()))
                    .arg(fileModel->rowCount()));

    selectItem(selectedHighlight.isEmpty() ? highlightFile : selectedHighlight);
    return true;
}

void MainWindow::addImageItem(const QString &filePath)
{
    const QFileInfo info(filePath);
    QStandardItem *item = new QStandardItem(info.fileName());
    item->setData(filePath, Qt::UserRole);
    item->setToolTip(QDir::toNativeSeparators(filePath));
    fileModel->appendRow(item);
}

void MainWindow::selectItem(const QString &filePath)
{
    if (filePath.isEmpty()) {
        if (fileModel->rowCount() > 0) {
            ui->lvFileName->setCurrentIndex(fileModel->index(0, 0));
        }
        return;
    }

    const QString target = QDir::cleanPath(filePath);
    for (int row = 0; row < fileModel->rowCount(); ++row) {
        const QModelIndex index = fileModel->index(row, 0);
        if (QDir::cleanPath(index.data(Qt::UserRole).toString()) == target) {
            ui->lvFileName->setCurrentIndex(index);
            return;
        }
    }

    if (fileModel->rowCount() > 0) {
        ui->lvFileName->setCurrentIndex(fileModel->index(0, 0));
    }
}

void MainWindow::displayImage(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, QStringLiteral("错误"),
                             QStringLiteral("无法读取图片：%1")
                                     .arg(filePath));
        return;
    }
    const QByteArray bytes = file.readAll();
    file.close();

    // OpenCV 从内存解码，避免中文路径下 imread 失效
    const cv::Mat encoded(1, bytes.size(), CV_8UC1,
                          const_cast<char *>(bytes.constData()));
    const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
    QImage image;
    if (!decoded.empty()) {
        image = cvMatToQImage(decoded);
    }

    if (image.isNull()) {
        // 退回 Qt 自带解码
        image = QImage(filePath);
    }
    if (image.isNull()) {
        QMessageBox::warning(this, QStringLiteral("错误"),
                             QStringLiteral("无法解码图片：%1").arg(filePath));
        return;
    }

    imageScene->clear();
    QGraphicsPixmapItem *pixmapItem =
            imageScene->addPixmap(QPixmap::fromImage(image));
    pixmapItem->setTransformationMode(Qt::SmoothTransformation);

    imageScene->setSceneRect(pixmapItem->boundingRect());
    ui->graphicsView->fitInView(pixmapItem->boundingRect(),
                                Qt::KeepAspectRatio);

    showFileInfo(filePath, image.size());
}

void MainWindow::showFileInfo(const QString &filePath, const QSize &imageSize)
{
    const QFileInfo info(filePath);
    const qint64 sizeBytes = info.size();
    QString sizeText;
    if (sizeBytes >= 1024 * 1024) {
        sizeText = QString::number(sizeBytes / (1024.0 * 1024.0), 'f', 2)
                + QStringLiteral(" MB");
    } else {
        sizeText = QString::number(sizeBytes / 1024.0, 'f', 1)
                + QStringLiteral(" KB");
    }

    ui->teShowInfo->setPlainText(
            QStringLiteral("文件名：%1\n"
                           "文件大小：%2\n"
                           "图片尺寸：%3 × %4")
                    .arg(info.fileName())
                    .arg(sizeText)
                    .arg(imageSize.width())
                    .arg(imageSize.height()));
}

bool MainWindow::isImageFile(const QString &filePath)
{
    const QString suffix = QFileInfo(filePath).suffix();
    for (const QString &filter : imageNameFilters()) {
        if (filter.contains(suffix, Qt::CaseInsensitive))
            return true;
    }
    return false;
}
