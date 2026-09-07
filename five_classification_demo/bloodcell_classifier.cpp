#include "bloodcell_classifier.h"

#include <QByteArray>
#include <QFile>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// MinGW 下把 MSVC 风格 _stdcall 映射为 GCC 的 __stdcall
#if defined(__MINGW32__) || defined(__MINGW64__)
#ifndef _stdcall
#define _stdcall __stdcall
#endif
#endif

#if defined(__APPLE__)
#include <onnxruntime_cxx_api.h>
#else
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#endif

namespace {

const int kImageHeight = 224;
const int kImageWidth = 224;

const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

struct ClassLabel
{
    const char *english;
    const char *chinese;
};

// 与 bloodcells_dataset 目录名排序一致
const ClassLabel kClassLabels[] = {
    {"basophil",  "嗜碱性粒细胞"},
    {"eosinophil", "嗜酸性粒细胞"},
    {"erythroblast", "成红细胞"},
    {"ig", "未成熟粒细胞（早幼粒细胞、中幼粒细胞和晚幼粒细胞）"},
    {"lymphocyte", "淋巴细胞"},
    {"monocyte", "单核细胞"},
    {"neutrophil", "中性粒细胞"},
    {"platelet", "血小板或血栓细胞"}
};

constexpr int kClassLabelCount = static_cast<int>(
        sizeof(kClassLabels) / sizeof(kClassLabels[0]));

QString englishLabel(int index)
{
    if (index >= 0 && index < kClassLabelCount)
        return QString::fromUtf8(kClassLabels[index].english);
    return QStringLiteral("Class %1").arg(index);
}

QString chineseLabel(int index)
{
    if (index >= 0 && index < kClassLabelCount)
        return QString::fromUtf8(kClassLabels[index].chinese);
    return englishLabel(index);
}

void labelsForIndex(int index, QString *english, QString *chinese)
{
    if (english)
        *english = englishLabel(index);
    if (chinese)
        *chinese = chineseLabel(index);
}

bool decodeImageFile(const QString &filePath, cv::Mat *bgr)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
        return false;
    const QByteArray bytes = file.readAll();
    file.close();

    const cv::Mat encoded(1, bytes.size(), CV_8UC1,
                          const_cast<char *>(bytes.constData()));
    *bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
    return !bgr->empty();
}

} // namespace

class BloodCellClassifier::Impl
{
public:
    Impl()
        : env(ORT_LOGGING_LEVEL_WARNING, "blood_cell_classifier")
    {
    }

    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::string outputName;
    QString error;
    bool ready = false;
};

BloodCellClassifier::BloodCellClassifier()
    : d(new Impl)
{
}

BloodCellClassifier::~BloodCellClassifier()
{
    delete d;
}

bool BloodCellClassifier::initialize(const QString &modelPath)
{
    d->session.reset();
    d->ready = false;
    d->error.clear();

    if (!QFileInfo::exists(modelPath)) {
        d->error = QStringLiteral("模型文件不存在：%1").arg(modelPath);
        return false;
    }

    try {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

#if defined(_WIN32)
        const std::wstring nativeModelPath = modelPath.toStdWString();
#else
        const std::string nativeModelPath = modelPath.toStdString();
#endif
        d->session = std::make_unique<Ort::Session>(
                d->env, nativeModelPath.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = d->session->GetInputNameAllocated(0, allocator);
        d->inputName = inputName.get();
        auto outputName = d->session->GetOutputNameAllocated(0, allocator);
        d->outputName = outputName.get();

        Ort::TypeInfo inputTypeInfo = d->session->GetInputTypeInfo(0);
        const auto inputShape =
                inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        // 输出类别数 = logits 的最后一维
        Ort::TypeInfo outputTypeInfo = d->session->GetOutputTypeInfo(0);
        const auto outputShape =
                outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        const int outputCount = outputShape.empty()
                ? 0
                : static_cast<int>(outputShape.back());
        if (outputCount <= 0) {
            d->error = QStringLiteral("无法获取模型输出类别数");
            return false;
        }

        // 只校验常见形状；异常时使用默认 224×224
        if (inputShape.size() == 4) {
            const int h = static_cast<int>(inputShape[2]);
            const int w = static_cast<int>(inputShape[3]);
            if (h <= 0 || w <= 0) {
                d->error = QStringLiteral("模型输入尺寸无效");
                return false;
            }
        }

        d->ready = true;
        return true;
    } catch (const Ort::Exception &e) {
        d->session.reset();
        d->error = QString::fromUtf8(e.what());
        return false;
    }
}

bool BloodCellClassifier::isReady() const
{
    return d->ready && d->session;
}

bool BloodCellClassifier::predictImage(const QString &imagePath,
                                       Prediction *prediction)
{
    if (!isReady())
        return false;
    if (!prediction)
        return false;

    cv::Mat bgr;
    if (!decodeImageFile(imagePath, &bgr)) {
        d->error = QStringLiteral("无法读取/解码图片：%1").arg(imagePath);
        return false;
    }

    // 与 Python val_test_transforms 对应：RGB、Resize(224,224)、
    // ToTensor、Normalize(mean,std)
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(kImageWidth, kImageHeight),
               0, 0, cv::INTER_LINEAR);
    cv::Mat floatImage;
    resized.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    const size_t channelSize =
            static_cast<size_t>(kImageHeight) * kImageWidth;
    std::vector<float> blob(channelSize * 3);

    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);
    for (int c = 0; c < 3; ++c) {
        cv::Mat normalized;
        cv::subtract(channels[c], cv::Scalar(kMean[c]), normalized);
        cv::divide(normalized, cv::Scalar(kStd[c]), normalized);
        std::copy_n(normalized.ptr<float>(), channelSize,
                    blob.begin() + c * channelSize);
    }

    try {
        const std::vector<int64_t> inputShape = {
            1, 3, kImageHeight, kImageWidth
        };
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, blob.data(), blob.size(),
                inputShape.data(), inputShape.size());

        const char *inputNames[] = {d->inputName.c_str()};
        const char *outputNames[] = {d->outputName.c_str()};
        std::vector<Ort::Value> outputs = d->session->Run(
                Ort::RunOptions{nullptr},
                inputNames, &inputTensor, 1,
                outputNames, 1);

        const Ort::Value &output = outputs.at(0);
        const float *logits = output.GetTensorData<float>();
        const int64_t count =
                output.GetTensorTypeAndShapeInfo().GetElementCount();
        if (count <= 0 || logits == nullptr) {
            d->error = QStringLiteral("模型输出为空");
            return false;
        }

        // logits -> softmax 概率
        const int classCount = static_cast<int>(count);
        std::vector<float> probs(classCount);
        const float maxLogit =
                *std::max_element(logits, logits + classCount);
        float sum = 0.0f;
        for (int i = 0; i < classCount; ++i) {
            probs[i] = std::exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        for (float &p : probs)
            p /= sum;

        // Top-3
        std::vector<int> order(classCount);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(),
                          order.begin() + std::min(3, classCount),
                          order.end(),
                          [&probs](int a, int b) {
                              return probs[a] > probs[b];
                          });

        prediction->classIndex = order.front();
        labelsForIndex(order.front(),
                       &prediction->className,
                       &prediction->classNameZh);
        prediction->confidence = probs[order.front()];

        const int topCount = std::min(3, classCount);
        prediction->topEntries.clear();
        for (int i = 0; i < topCount; ++i) {
            const int idx = order.at(i);
            Prediction::TopEntry entry;
            labelsForIndex(idx, &entry.className, &entry.classNameZh);
            entry.probability = probs[idx];
            prediction->topEntries.append(entry);
        }
        d->error.clear();
        return true;
    } catch (const Ort::Exception &e) {
        d->error = QString::fromUtf8(e.what());
        return false;
    }
}

QString BloodCellClassifier::lastError() const
{
    return d->error;
}
