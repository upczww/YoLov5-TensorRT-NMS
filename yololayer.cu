#include <assert.h>
#include "yololayer.h"
#include "utils.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel> &vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void *)));
        size_t AnchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto &yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    YoloLayerPlugin::~YoloLayerPlugin()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void *data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void *)));
        size_t AnchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto &yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void *buffer) const
    {
        using namespace Tn;
        char *d = static_cast<char *>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloLayerPlugin::initialize()
    {
        return 0;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
    {   assert(index < 2);
        //output the result to channel
        if (index == 0)
        {
            return Dims3(mMaxOutObject, 1, 4);
        }
        return DimsHW(mMaxOutObject, mClassCount);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char *pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char *YoloLayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() {}

    const char *YoloLayerPlugin::getPluginType() const
    {
        return "YoloLayer_TRT";
    }

    const char *YoloLayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void YoloLayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt *YoloLayerPlugin::clone() const
    {
        YoloLayerPlugin *p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *bboxData, float *scoreData, int *countData, int noElements,
                                 const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements)
            return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes;                                                    // 85
        const float *curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT); // b*h*w*3*85

        for (int k = 0; k < 3; ++k)
        {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH)
                continue;
            int *res_count = countData + bnIdx;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject)
                return;

            float *curBbox = bboxData + bnIdx * maxoutobject * 4 + count * 4;
            float *curScore = scoreData + bnIdx * maxoutobject * classes + count * classes;

            for (int i = 5; i < info_len_i; ++i)
            {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                curScore[i - 5] = p * box_prob;
            }
            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            // pytorch:
            //  y = x[i].sigmoid()
            //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth
            float cx = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            float cy = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth
            // v5: https://github.com/ultralytics/yolov5/issues/471
            float w = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            w = w * w * anchors[2 * k];
            float h = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            h = h * h * anchors[2 * k + 1];
            // cx,cy,w,h to x1,y1,x2,y2
            curBbox[0] = cx - 0.5 * w;
            curBbox[1] = cy - 0.5 * h;
            curBbox[2] = cx + 0.5 * w;
            curBbox[3] = cy + 0.5 * h;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const *inputs, void **outputs, void *workspace, cudaStream_t stream, int batchSize)
    {
        float *bboxData = (float *)outputs[0];
        float *scoreData = (float *)outputs[1];
        int *countData = (int *)workspace;

        for (int idx = 0; idx < batchSize; ++idx)
        {
            CUDA_CHECK(cudaMemset(countData + idx, 0, sizeof(int)));
        }
        CUDA_CHECK(cudaMemset(bboxData, 0, sizeof(float) * mMaxOutObject * 4 * batchSize));
        CUDA_CHECK(cudaMemset(scoreData, 0, sizeof(float) * mMaxOutObject * mClassCount * batchSize));

        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
        {
            const auto &yolo = mYoloKernel[i];
            numElem = yolo.width * yolo.height * batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CalDetection<<<(yolo.width * yolo.height * batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>(inputs[i], bboxData, scoreData, countData, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount);
        }
    }

    int YoloLayerPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
    {
        forwardGpu((const float *const *)inputs, outputs, workspace, stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char *YoloPluginCreator::getPluginName() const
    {
        return "YoloLayer_TRT";
    }

    const char *YoloPluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection *YoloPluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt *YoloPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
    {
        int class_count = 80;
        int input_w = 416;
        int input_h = 416;
        int max_output_object_count = 1000;
        std::vector<Yolo::YoloKernel> yolo_kernels(3);

        const PluginField *fields = fc->fields;
        for (int i = 0; i < fc->nbFields; i++)
        {
            if (strcmp(fields[i].name, "netdata") == 0)
            {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int *)(fields[i].data);
                class_count = tmp[0];
                input_w = tmp[1];
                input_h = tmp[2];
                max_output_object_count = tmp[3];
            }
            else if (strstr(fields[i].name, "yolodata") != NULL)
            {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int *)(fields[i].data);
                YoloKernel kernel;
                kernel.width = tmp[0];
                kernel.height = tmp[1];
                for (int j = 0; j < fields[i].length - 2; j++)
                {
                    kernel.anchors[j] = tmp[j + 2];
                }
                yolo_kernels[2 - (fields[i].name[8] - '1')] = kernel;
            }
        }
        YoloLayerPlugin *obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, yolo_kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt *YoloPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin *obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
} // namespace nvinfer1
