#ifndef BLOODCELL_CLASSIFIER_H
#define BLOODCELL_CLASSIFIER_H

#include <QList>
#include <QString>

class BloodCellClassifier
{
public:
    BloodCellClassifier();
    ~BloodCellClassifier();

    // 初始化 ONNX Runtime 会话；成功返回 true
    bool initialize(const QString &modelPath);
    bool isReady() const;

    struct Prediction {
        int classIndex = -1;
        QString className;      // 英文/原始类别
        QString classNameZh;    // 中文类别
        float confidence = 0.0f;

        struct TopEntry {
            QString className;
            QString classNameZh;
            float probability = 0.0f;
        };
        QList<TopEntry> topEntries;
    };

    bool predictImage(const QString &imagePath, Prediction *prediction);
    QString lastError() const;

private:
    class Impl;
    Impl *d;
};

#endif // BLOODCELL_CLASSIFIER_H
