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
// use the namespace
using namespace std;
using namespace cv;
using namespace at;

// define the path separator
#ifdef _WIN32
constexpr auto PATH_SEPARATOR = "\\";
#else
constexpr auto PATH_SEPARATOR = "/";
#endif

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
        output_folder = argc > 2 ? argv[2] : "output";
        model_path = argc > 3 ? argv[3] : R"(I:\deploy_sr\torch_script_model.pt)";
        batch_size = argc > 4 ? stoi(argv[4]) : 8;
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


    // load the model
    torch::jit::script::Module module;
    try{
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        cout << "Error loading the model!" << endl;
        return -1;
    }

    // check the device type
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        if (is_log) {
            std::cout << "CUDA available! Predicting on GPU." << std::endl;
        }
        device_type = torch::kCUDA;
    }
    else
    {
        if (is_log) {
            std::cout << "Predicting on CPU." << std::endl;
        }
        device_type = torch::kCPU;
    }
    auto device = torch::Device(device_type);

    // convert the model to the device type
    module.to(device);
    module.eval();

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

    auto tensorType = torch::kFloat;
    if (half) {
        tensorType = torch::kHalf;
    }
    tensor_patches = tensor_patches.to(tensorType);
    module.to(tensorType);

    // loop to forward the model
    auto n_batch = ceil(tensor_patches.size(0) / batch_size);
    auto tensor_output = torch::zeros({ tensor_patches.size(0), channel, crop_size*scale_factor, crop_size*scale_factor }, torch::TensorOptions(tensorType));
    if (is_log) {
        // print the data preparation time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Data preparation time: " << duration.count() << " seconds" << endl;
    }

    for (int i = 0; i < n_batch; i++) {
        auto tensor_batch = tensor_patches.slice(0, i * batch_size, (i + 1) * batch_size);
        tensor_batch = tensor_batch.to(device);
        // if channel == 1, change to 3
        if (channel == 1) {
            tensor_batch = tensor_batch.repeat({1, 3, 1, 1});
        }
        // forward with no grad
        torch::NoGradGuard no_grad;
        auto tensor_batch_output = module.forward({ tensor_batch }).toTensor();
        // if channel == 3, change to 1 by mean
        if (channel == 1) {
            tensor_batch_output = tensor_batch_output.mean(1, true);
        }
        tensor_batch_output = tensor_batch_output.to(torch::kCPU);
        tensor_output.slice(0, i * batch_size, (i + 1) * batch_size) = tensor_batch_output;
        if (is_log) {
            cout << "Total batch: " << n_batch << ", current batch: " << i << endl;
        }
    }
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

//        ////write the image using opencv
//        auto zz = tensor_output[0][i];
//        zz = zz / value_range * 255;
//        zz = zz.to(torch::kByte);
//        // convert the tensor to opencv mat
//        cv::Mat mat_output = cv::Mat(zz.size(1), zz.size(0), CV_8UC1, zz.data_ptr());
//        // write the mat to a new gdal file
//        cv::imwrite("output.png", mat_output);

