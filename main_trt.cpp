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
    // devisiable by 255
    tensor_patches = tensor_patches / value_range;

    if (is_log) {
        cout << "Image to patches!" << endl;
        cout << "tensor size: " << tensor_patches.sizes() << endl;
    }

    // loop to forward the model
    int n_batch = ceil(float(tensor_patches.size(0)) / float(batch_size));
    auto tensor_output = torch::zeros({ tensor_patches.size(0), channel, crop_size*scale_factor, crop_size*scale_factor }, torch::TensorOptions(torch::kFloat));
    if (is_log) {
        // print the data preparation time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Data preparation time: " << duration.count() << " seconds" << endl;
    }

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

    // loop to forward the model
    int64_t exact_batch_size;
    for (int i = 0; i < n_batch; i++) {
        auto tensor_out_tmp = torch::zeros({ batch_size, channel_tmp, crop_size*scale_factor, crop_size*scale_factor }, torch::TensorOptions(torch::kFloat));
        auto tensor_batch = tensor_patches.slice(0, i * batch_size, (i + 1) * batch_size);
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
        cudaMemcpyAsync(tensor_out_tmp.data_ptr(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

        if (exact_batch_size < batch_size) {
            tensor_out_tmp = tensor_out_tmp.slice(0, 0, exact_batch_size);
        }
        // if channel == 3, change to 1 by mean
        if (channel == 1) {
            tensor_out_tmp = tensor_out_tmp.mean(1, true);
        }
        // copy the output tensor to the output tensor
        tensor_output.slice(0, i * batch_size, (i + 1) * batch_size) = tensor_out_tmp;
        if (is_log) {
            if (i % 30 == 0){
                cout << "Total batch: " << n_batch << ", current batch: " << i << endl;
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


    if (is_log) {
        // print the model forward time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Model forward time: " << duration.count() << " seconds" << endl;
    }
    // clamp the output
    tensor_output = tensor_output.clamp(0, 1);
    // convert to float
    tensor_output = tensor_output.to(torch::kFloat);
    tensor_output = tensor_output * value_range;
    tensor_output = tensor_output.to(torch::kInt16);

    // restore to the original whole image
    tensor_output = tensor_output.view({ -1, n_patch_h, n_patch_w, channel, crop_size*scale_factor, crop_size*scale_factor });
    tensor_output = tensor_output.permute({ 0, 3, 1, 4, 2, 5 }).contiguous();

    tensor_output = tensor_output.view({ -1, channel, n_patch_h * crop_size*scale_factor, n_patch_w * crop_size*scale_factor });

    tensor_output = tensor_output.slice(2, 0, height*scale_factor);
    tensor_output = tensor_output.slice(3, 0, width*scale_factor);

    // write the output image to a new gdal file
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL) {
        if (is_log) {
            cout << "Error getting the driver!" << endl;
            return -1;
        }
    }
    // get the input image file name without suffix
    string input_image_name = filesystem::path(input_image_path).stem().string();
    // get the output image file name
    string output_image_name = output_folder + PATH_SEPARATOR + input_image_name + "_sr.tif";
    // create the output image
    GDALDataset* poDstDS = poDriver->Create(output_image_name.c_str(), width*scale_factor, height*scale_factor, channel, GDT_Int16, NULL);
    if (poDstDS == NULL) {
        if (is_log) {
            cout << "Error creating the output image!" << endl;
            return -1;
        }
    }
    tensor_output = tensor_output.contiguous();
    // write the tensor data to gdal
    for (int i = 0; i < channel; i++) {
        GDALRasterBand* poBand = poDstDS->GetRasterBand(i + 1);
        poBand->RasterIO(GF_Write, 0, 0, width*scale_factor, height*scale_factor, tensor_output[0][i].data_ptr(), width*scale_factor, height*scale_factor, GDT_Int16, 0, 0);
    }
    // close the dataset
    GDALClose(poDstDS);
    // close the dataset
    GDALClose(poDataset);
    // close the dataset
    GDALDestroyDriverManager();

    if (is_log) {
        cout << "Image written!" << endl;
        // data post processing time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Data post processing time: " << duration.count() << " seconds" << endl;
    }
    system("pause");
}

//auto zz = tensor_out_tmp[0][0];
//zz = zz / value_range * 255;
//zz = zz.to(torch::kByte);
//// convert the tensor to opencv mat
//cv::Mat mat_output = cv::Mat(zz.size(1), zz.size(0), CV_8UC1, zz.data_ptr());
//// write the mat to a new gdal file
//cv::imwrite("output.png", mat_output);

