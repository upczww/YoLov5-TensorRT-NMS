#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "yololayer.h"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

using namespace nvinfer1;

cv::Mat preprocess_img(cv::Mat &img)
{
    int w, h, x, y;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w)
    {
        w = Yolo::INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo::INPUT_H - h) / 2;
    }
    else
    {
        w = r_h * img.cols;
        h = Yolo::INPUT_H;
        x = (Yolo::INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo::INPUT_H, Yolo::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat &img, float *bbox)
{
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w)
    {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[3] - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox[0] - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[2] - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
    float *gamma = (float *)weightMap[lname + ".weight"].values;
    float *beta = (float *)weightMap[lname + ".bias"].values;
    float *mean = (float *)weightMap[lname + ".running_mean"].values;
    float *var = (float *)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer *convBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int g, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 2;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // hard_swish = x * hard_sigmoid
    auto hsig = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    auto ew = network->addElementWise(*bn1->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer *focus(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int ksize, std::string lname)
{
    ISliceLayer *s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ISliceLayer *s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ISliceLayer *s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ISliceLayer *s4 = network->addSlice(input, Dims3{0, 1, 1}, Dims3{inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ITensor *inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer *bottleneck(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, bool shortcut, int g, float e, std::string lname)
{
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2)
    {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer *bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{1, 1}, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++)
    {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{1, 1}, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor *inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer *bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer *SPP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int k1, int k2, int k3, std::string lname)
{
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k1, k1});
    pool1->setPaddingNd(DimsHW{k1 / 2, k1 / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k2, k2});
    pool2->setPaddingNd(DimsHW{k2 / 2, k2 / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k3, k3});
    pool3->setPaddingNd(DimsHW{k3 / 2, k3 / 2});
    pool3->setStrideNd(DimsHW{1, 1});

    ITensor *inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr)
    {
        return -1;
    }

    struct dirent *p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr)
    {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0)
        {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

std::vector<float> getAnchors(std::map<std::string, Weights> &weightMap)
{
    std::vector<float> anchors_yolo;
    Weights Yolo_Anchors = weightMap["model.24.anchor_grid"];
    assert(Yolo_Anchors.count == 18);
    int each_yololayer_anchorsnum = Yolo_Anchors.count / 3;
    const float *tempAnchors = (const float *)(Yolo_Anchors.values);
    for (int i = 0; i < Yolo_Anchors.count; i++)
    {
        if (i < each_yololayer_anchorsnum)
        {
            anchors_yolo.push_back(const_cast<float *>(tempAnchors)[i]);
        }
        if ((i >= each_yololayer_anchorsnum) && (i < (2 * each_yololayer_anchorsnum)))
        {
            anchors_yolo.push_back(const_cast<float *>(tempAnchors)[i]);
        }
        if (i >= (2 * each_yololayer_anchorsnum))
        {
            anchors_yolo.push_back(const_cast<float *>(tempAnchors)[i]);
        }
    }
    return anchors_yolo;
}

IPluginV2Layer *addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, IConvolutionLayer *det0, IConvolutionLayer *det1, IConvolutionLayer *det2)
{
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    std::vector<float> anchors_yolo = getAnchors(weightMap);
    PluginField pluginMultidata[4];
    int NetData[4];
    NetData[0] = Yolo::CLASS_NUM;
    NetData[1] = Yolo::INPUT_W;
    NetData[2] = Yolo::INPUT_H;
    NetData[3] = Yolo::MAX_OUTPUT_BBOX_COUNT;
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 3;
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[3] = {8, 16, 32};
    int plugindata[3][8];
    std::string names[3];
    for (int k = 1; k < 4; k++)
    {
        plugindata[k - 1][0] = Yolo::INPUT_W / scale[k - 1];
        plugindata[k - 1][1] = Yolo::INPUT_H / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8;
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }
    PluginFieldCollection pluginData;
    pluginData.nbFields = 4;
    pluginData.fields = pluginMultidata;
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", &pluginData);
    ITensor *inputTensors_yolo[] = {det2->getOutput(0), det1->getOutput(0), det0->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    yolo->setName("yolo_layer");
    return yolo;
}

IPluginV2Layer *addBatchedNMSLayer(INetworkDefinition *network, IPluginV2Layer *yolo, int num_classes, int top_k, int keep_top_k, float score_thresh, float iou_thresh, bool is_normalized = false, bool clip_boxes = false)
{
    auto creator = getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1");
    // Set plugin fields and the field collection
    const bool share_location = true;
    const int background_id = -1;
    PluginField fields[9] = {
        PluginField{"shareLocation", &share_location,
                    PluginFieldType::kINT32, 1},
        PluginField{"backgroundLabelId", &background_id,
                    PluginFieldType::kINT32, 1},
        PluginField{"numClasses", &num_classes,
                    PluginFieldType::kINT32, 1},
        PluginField{"topK", &top_k, PluginFieldType::kINT32,
                    1},
        PluginField{"keepTopK", &keep_top_k,
                    PluginFieldType::kINT32, 1},
        PluginField{"scoreThreshold", &score_thresh,
                    PluginFieldType::kFLOAT32, 1},
        PluginField{"iouThreshold", &iou_thresh,
                    PluginFieldType::kFLOAT32, 1},
        PluginField{"isNormalized", &is_normalized,
                    PluginFieldType::kINT32, 1},
        PluginField{"clipBoxes", &clip_boxes,
                    PluginFieldType::kINT32, 1},
    };
    PluginFieldCollection pfc{9, fields};
    IPluginV2 *pluginObj = creator->createPlugin("batchednms", &pfc);
    ITensor *inputTensors[] = {yolo->getOutput(0), yolo->getOutput(1)};
    auto batchednmslayer = network->addPluginV2(inputTensors, 2, *pluginObj);
    batchednmslayer->setName("nms_layer");
    assert(batchednmslayer);
    return batchednmslayer;
}
/*
IPluginV2Layer *addBatchedNMSLayer(INetworkDefinition *network, IPluginV2Layer *yolo, int num_classes, int top_k, int keep_top_k, float score_thresh, float iou_thresh, bool is_normalized = false, bool clip_boxes = false)
{
    nvinfer1::plugin::NMSParameters param;
    // Set plugin fields and the field collection
    const bool share_location = true;
    const int background_id = -1;
    param.backgroundLabelId = background_id;
    param.iouThreshold = iou_thresh;
    param.isNormalized = is_normalized;
    param.keepTopK = keep_top_k;
    param.numClasses = num_classes;
    param.scoreThreshold = score_thresh;
    param.shareLocation = share_location;
    param.topK = top_k;

    IPluginV2 *pluginObj = createBatchedNMSPlugin(param);

    ITensor *inputTensors[] = {yolo->getOutput(0), yolo->getOutput(1)};
    auto batchednmslayer = network->addPluginV2(inputTensors, 2, *pluginObj);
    batchednmslayer->setName("nms_layer");
    assert(batchednmslayer);
    return batchednmslayer;
}
*/
#endif
