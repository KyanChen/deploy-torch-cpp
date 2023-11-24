// include the libtorch
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
// include the opencv
#include <opencv2/opencv.hpp>
// include the filesystem
#include <filesystem>
// include the gdal
#include <gdal_priv.h>
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
// include tensorrt
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

// use the namespace
using namespace std;
using namespace cv;
using namespace at;
using namespace nvinfer1;


// define the path separator
#ifdef _WIN32
constexpr auto PATH_SEPARATOR = "\\";
#else
constexpr auto PATH_SEPARATOR = "/";
#endif

// define the queue for the output
queue<torch::Tensor> queue_output;
// define the mutex for the queue
std::mutex mtx;
// define the condition variable for the queue
std::condition_variable cv_queue;


// define the logger class
class MyLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


// create the tensorrt engine
nvinfer1::ICudaEngine* createEngine(const std::string& onnxModelPath, int maxBatchSize, int half) {
    // if there is a serialized engine file, use it instead of building a new one
    // else build a new one and save it to the disk
    // if the serialized engine file exists, use it
    string serialized_engine_path = onnxModelPath + ".engine";
    if (filesystem::exists(serialized_engine_path)) {
        // deserialize the engine
        ifstream serialized_engine_file(serialized_engine_path, ios::binary);
        if (serialized_engine_file.good()) {
            serialized_engine_file.seekg(0, ios::end);
            size_t serialized_engine_size = serialized_engine_file.tellg();
            serialized_engine_file.seekg(0, ios::beg);
            unique_ptr<char[]> serialized_engine(new char[serialized_engine_size]);
            serialized_engine_file.read(serialized_engine.get(), serialized_engine_size);
            serialized_engine_file.close();
            // deserialize the engine and return the engine for forward pass
            nvinfer1::ICudaEngine* engine;
            nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
            engine = runtime->deserializeCudaEngine(serialized_engine.get(), serialized_engine_size, nullptr);
            return engine;
        }
    }
    // if the serialized engine file does not exist, build a new one and save it
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    // create the config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // set config
    config->setMaxWorkspaceSize(1 << 20);
    // set the fp16 mode
    if (half) {
        config->setFlag(BuilderFlag::kFP16);
    }
    // set the dynamic batch size
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{ 1, 3, 512, 512 });
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{ maxBatchSize, 3, 512, 512 });
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{ 2*maxBatchSize, 3, 512, 512 });
//    profile->setDimensions("output", OptProfileSelector::kMIN, Dims4{ 1, 3, 1024, 1024 });
//    profile->setDimensions("output", OptProfileSelector::kOPT, Dims4{ maxBatchSize, 3, 1024, 1024 });
//    profile->setDimensions("output", OptProfileSelector::kMAX, Dims4{ maxBatchSize * 2, 3, 1024, 1024 });
    config->addOptimizationProfile(profile);

    // create the network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // create the parser
    auto parser = nvonnxparser::createParser(*network, gLogger);
    // parse the onnx model
    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        cout << "Failed to parse the onnx model" << endl;
        return nullptr;
    }

    // set the input and output name of the tensorrt
    auto input = network->getInput(0);
    input->setName("input");
    auto output = network->getOutput(0);
    output->setName("output");

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // serialize the engine
    nvinfer1::IHostMemory* serialized_engine = engine->serialize();
    // save the serialized engine to the disk
    ofstream serialized_engine_file(serialized_engine_path, ios::binary);
    if (!serialized_engine_file.good()) {
        cout << "Failed to open the serialized engine file" << endl;
        return nullptr;
    }
    serialized_engine_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    serialized_engine_file.close();
    // destroy the builder
    builder->destroy();
    // destroy the network
    network->destroy();
    // destroy the parser
    parser->destroy();
    // destroy the serialized engine
    serialized_engine->destroy();
    // return the engine
    return engine;

}


// predict the data in loop, and push the result to the queue
void predict_thread(nvinfer1::ICudaEngine* engine, torch::Tensor& tensor_patches, int batch_size, int crop_size, int scale_factor, int is_log) {
    int channel = tensor_patches.size(1);

    // create inconsistent tensorrt context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    // create input and output buffer on device
    void* buffers[2];
    int channel_tmp = 3;
    // set the batch size
    int inputIndex = engine->getBindingIndex("input");
    context->setBindingDimensions(inputIndex, Dims4(batch_size, channel_tmp, crop_size, crop_size));
//    context->setBindingDimensions(1, Dims4(batch_size, channel_tmp, crop_size*scale_factor, crop_size*scale_factor));
    int input_size = batch_size * channel_tmp * crop_size * crop_size;
    int output_size = batch_size * channel_tmp * crop_size * crop_size * scale_factor * scale_factor;
    cudaMalloc(&buffers[0], input_size * sizeof(float));
    cudaMalloc(&buffers[1], output_size * sizeof(float));
    // create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int64_t exact_batch_size;
    // loop to forward the model
    for (int i = 0; i < tensor_patches.size(0); i += batch_size) {
        auto tensor_output = torch::zeros({ batch_size, channel_tmp, crop_size*scale_factor, crop_size*scale_factor }, torch::TensorOptions(torch::kFloat));
        // get the batch data
        auto tensor_batch = tensor_patches.slice(0, i, i + batch_size);
        // repeat 1 channel to 3 channels if the input is 1 channel
        if (channel == 1) {
            tensor_batch = tensor_batch.repeat({ 1, 3, 1, 1 });
        }
        exact_batch_size = tensor_batch.size(0);

        if (channel == 1) {
            tensor_batch = tensor_batch.repeat({ 1, 3, 1, 1 });
        }
        // pad to desired batch size by repeating the last patch
        if (exact_batch_size < batch_size) {
            auto tensor_batch_tmp = tensor_batch[-1].unsqueeze(0).repeat({ batch_size - tensor_batch.size(0), 1, 1, 1 });
            tensor_batch = torch::cat({ tensor_batch, tensor_batch_tmp }, 0);
        }
        // copy the input tensor to device
        cudaMemcpyAsync(buffers[0], tensor_batch.data_ptr(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        // run inference
        context->enqueueV2(buffers, stream, nullptr);
        // copy the output tensor to host
        cudaMemcpyAsync(tensor_output.data_ptr(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

        if (exact_batch_size < batch_size) {
            tensor_output = tensor_output.slice(0, 0, exact_batch_size);
        }
        // if channel == 3, change to 1 by mean
        if (channel == 1) {
            tensor_output = tensor_output.mean(1, true);
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            // push the result to the queue
            queue_output.push(tensor_output);
        }
        cv_queue.notify_one();
        if (is_log) {
            if (i % (batch_size * 50) == 0) {
                cout << "Total: " << tensor_patches.size(0) / batch_size << " Current: " << i / batch_size << endl;
            }
        }
    }
    // destroy the context
    context->destroy();
    // destroy the buffers
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    // destroy the stream
    cudaStreamDestroy(stream);
    // destroy the engine
    engine->destroy();
}


// write the data in loop using GDAL, and pop the result from the queue
void write_thread(int dst_width, int dst_height, int channel, int dst_patch_size, int value_range, string& output_image_name, int is_log) {
    // create the output image
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL) {
        cout << "Error getting the driver!" << endl;
        return;
    }
    // create the output image
    GDALDataset* poDstDS = poDriver->Create(output_image_name.c_str(), dst_width, dst_height, channel, GDT_Int16, NULL);
    if (poDstDS == NULL) {
        cout << "Error creating the output image!" << endl;
        return;
    }
    // loop to write the data
    int w_start = 0;
    int h_start = 0;
    while (true) {
        torch::Tensor tensor_output;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_queue.wait(lock, [] { return !queue_output.empty(); });
            tensor_output = queue_output.front();
            queue_output.pop();
        }
        // preprocess the data
        tensor_output = tensor_output * value_range;
        tensor_output = tensor_output.to(torch::kInt16);
        // tensor_output is B C H W, we need to loop on the B and C to write the data
        for (int b = 0; b < tensor_output.size(0); b++) {
            for (int c = 0; c < tensor_output.size(1); c++) {
                auto current_data = tensor_output[b][c];
                // get the band
                GDALRasterBand *poBand = poDstDS->GetRasterBand(c + 1);
                // remove the padding
                if (w_start + dst_patch_size > dst_width) {
                    current_data = current_data.slice(1, 0, dst_width - w_start);
                }
                if (h_start + dst_patch_size > dst_height) {
                    current_data = current_data.slice(0, 0, dst_height - h_start);
                }
                // write the data
                poBand->RasterIO(GF_Write, w_start, h_start, current_data.size(1), current_data.size(0),
                                 current_data.data_ptr(), current_data.size(1), current_data.size(0), GDT_Int16, 0, 0);
            }

            // update the w_start and h_start
            w_start += dst_patch_size;
            if (w_start >= dst_width) {
                w_start = 0;
                h_start += dst_patch_size;
            }
        }
        // check if the loop is finished
        if (h_start >= dst_height) {
            break;
        }
    }
    // close the dataset
    GDALClose(poDstDS);
    if (is_log) {
        cout << "Image written!" << endl;
    }
}


// main function with argc and argv
int main(int argc, const char* argv[]) {
    // check the input arguments, it needs input_image_path, output_folder, model_path, is_log
    // each argument has a default value
    try {
        if (argc < 1) {
            cout << "Input arguments: input_image_path, output_folder, model_path, is_log, crop_size, batch_size, scale_factor, value_range, half" << endl;
            throw "Error: No input arguments!";
        }
        if (argc > 9) {
            cout << "Input arguments: input_image_path, output_folder, model_path, is_log, crop_size, batch_size, scale_factor, value_range, half" << endl;
            throw "Error: Too many input arguments!";
        }
    }
    catch (const char* msg) {
        cerr << msg << endl;
        return -1;
    }

    string input_image_path;
    string output_folder;
    string model_path;
    int is_log;
    int crop_size;
    int batch_size;
    int scale_factor;
    int value_range;
    int half;

    try {
        input_image_path = argc > 1 ? argv[1] : R"(I:\deploy_sr\gf7_1_pan\GF7_DLC_E117.7_N38.8_20231003_L1A0001283650-BWDPAN.tiff)";
//        input_image_path = argc > 1 ? argv[1] : R"(I:\deploy_sr\sample.tiff)";
        output_folder = argc > 2 ? argv[2] : "output";
        model_path = argc > 3 ? argv[3] : R"(I:\deploy_sr\rrdbnet.onnx)";
        batch_size = argc > 4 ? stoi(argv[4]) : 16;
        is_log = argc > 5 ? stoi(argv[5]) : 1;
        crop_size = argc > 6 ? stoi(argv[6]) : 512;
        scale_factor = argc > 7 ? stoi(argv[7]) : 2;
        value_range = argc > 8 ? stoi(argv[8]) : 255*8;
        half = argc > 9 ? stoi(argv[9]) : 1;
    }
    catch (const char* msg) {
        cout << "Error: Input arguments type error!" << endl;
        cout << "Input arguments: input_image_path, output_folder, model_path, is_log, crop_size, batch_size, scale_factor, value_range, half" << endl;
        cerr << msg << endl;
        return -1;
    }

    if (is_log) {
        cout << "Input image path: " << input_image_path << endl;
        cout << "Output folder: " << output_folder << endl;
        cout << "Model path: " << model_path << endl;
        cout << "Is log: " << is_log << endl;
        cout << "Crop size: " << crop_size << endl;
        cout << "Batch size: " << batch_size << endl;
        cout << "Scale factor: " << scale_factor << endl;
        cout << "Value range: " << value_range << endl;
        cout << "Half: " << half << endl;
    }

    // log the start time for calculate the running time
    auto start = std::chrono::steady_clock::now();

    // check the input image path
    if (!filesystem::exists(input_image_path)) {
        cout << "Input image path does not exist!" << endl;
        return -1;
    }

    // check the output folder, if it does not exist, create it, otherwise, do nothing
    if (!filesystem::exists(output_folder)) {
        filesystem::create_directory(output_folder);
    }

    // check the model path
    if (!filesystem::exists(model_path)) {
        cout << "Model path does not exist!" << endl;
        return -1;
    }

    // create the tensorrt engine
    auto engine = createEngine(model_path, batch_size, half);

    // load the tif using gdal
    GDALAllRegister();
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(input_image_path.c_str(), GA_ReadOnly);
    if (poDataset == NULL) {
        if (is_log) {
            cout << "Error loading the image!" << endl;
            return -1;
        }
    }

    // get the image width, height, and channel
    int width = poDataset->GetRasterXSize();
    int height = poDataset->GetRasterYSize();
    int channel = poDataset->GetRasterCount();

    if (is_log) {
        cout << "Image width: " << width << endl;
        cout << "Image height: " << height << endl;
        cout << "Image channel: " << channel << endl;
    }

    // convert the image to torch tensor
    torch::TensorOptions options(torch::kInt16);
    torch::Tensor tensor_image = torch::zeros({ 1, channel, height, width }, options);
    for (int i = 0; i < channel; i++) {
        GDALRasterBand* poBand = poDataset->GetRasterBand(i + 1);
        poBand->RasterIO(GF_Read, 0, 0, width, height, tensor_image[0][i].data_ptr(), width, height, GDT_Int16, 0, 0);
    }

    if (is_log) {
        cout << "Image converted to tensor!" << endl;
        cout << "tensor size: " << tensor_image.sizes() << endl;
    }

    // zero pad the image to be divisible by crop_size
    int pad_width = (crop_size - width % crop_size) % crop_size;
    int pad_height = (crop_size - height % crop_size) % crop_size;
    torch::nn::ZeroPad2d zero_pad(torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({ 0, pad_width, 0, pad_height })));
    tensor_image = zero_pad->forward(tensor_image);

    if (is_log) {
        cout << "Image zero padded!" << endl;
        cout << "tensor size: " << tensor_image.sizes() << endl;
    }
    // convert B C H W -> (B n_patch_h n_patch_w) C patch_h patch_w
    auto n_patch_h = tensor_image.size(2) / crop_size;
    auto n_patch_w = tensor_image.size(3) / crop_size;
    auto tensor_patches = tensor_image.view({ -1, channel, n_patch_h, crop_size, n_patch_w, crop_size });
    tensor_patches = tensor_patches.permute({ 0, 2, 4, 1, 3, 5 }).contiguous();
    tensor_patches = tensor_patches.view({ -1, channel, crop_size, crop_size });
    tensor_patches = tensor_patches / value_range;

    if (is_log) {
        cout << "Image to patches!" << endl;
        cout << "tensor size: " << tensor_patches.sizes() << endl;
    }

    if (is_log) {
        // print the data preparation time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Data preparation time: " << duration.count() << " seconds" << endl;
    }

    // predict thread
    std::thread predict(predict_thread, std::ref(engine), std::ref(tensor_patches), batch_size, crop_size, scale_factor, is_log);
    // write thread
    // out_filename is the original filename + 'sr'
    string output_image_name = output_folder + PATH_SEPARATOR + filesystem::path(input_image_path).stem().string() + "_sr.tif";
    int dst_width = width * scale_factor;
    int dst_height = height * scale_factor;
    int dst_patch_size = crop_size * scale_factor;
    std::thread write(write_thread, dst_width, dst_height, channel, dst_patch_size, value_range, std::ref(output_image_name), is_log);
    // join the thread
    predict.join();
    write.join();

    // close the dataset
    GDALClose(poDataset);
    // close gdal
    GDALDestroyDriverManager();

    if (is_log) {
        // data post processing time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Data post processing time: " << duration.count() << " seconds" << endl;
    }
    system("pause");
}

//        ////write the image using opencv
//        auto zz = tensor_output[0][i];
//        zz = zz / value_range * 255;
//        zz = zz.to(torch::kByte);
//        // convert the tensor to opencv mat
//        cv::Mat mat_output = cv::Mat(zz.size(1), zz.size(0), CV_8UC1, zz.data_ptr());
//        // write the mat to a new gdal file
//        cv::imwrite("output.png", mat_output);

