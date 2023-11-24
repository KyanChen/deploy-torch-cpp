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

// use the namespace
using namespace std;
using namespace cv;

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


// predict the data in loop, and push the result to the queue
void predict_thread(torch::jit::script::Module& module, const torch::Tensor& tensor_patches, int batch_size, torch::Device& device, int is_log) {
    int channel = tensor_patches.size(1);
    // no grad
    // loop to forward the model
    for (int i = 0; i < tensor_patches.size(0); i += batch_size) {
        // get the batch data
        auto tensor_batch = tensor_patches.slice(0, i, i + batch_size);
        // to device
        tensor_batch = tensor_batch.to(device);
        // repeat 1 channel to 3 channels if the input is 1 channel
        if (channel == 1) {
            tensor_batch = tensor_batch.repeat({ 1, 3, 1, 1 });
        }
        // forward the model
        torch::NoGradGuard no_grad;
        auto tensor_output = module.forward({ tensor_batch }).toTensor();
        // to cpu
        tensor_output = tensor_output.to(torch::kCPU);
        // clamp the output
        tensor_output = tensor_output.clamp(0, 1);
        // convert to float
        tensor_output = tensor_output.to(torch::kFloat);
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
            cout << "Total: " << tensor_patches.size(0) / batch_size << " Current: " << i / batch_size << endl;
        }
    }

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
        model_path = argc > 3 ? argv[3] : R"(I:\deploy_sr\torch_script_model.pt)";
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


    if (is_log) {
        // print the data preparation time using seconds
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Data preparation time: " << duration.count() << " seconds" << endl;
    }

    // predict thread
    std::thread predict(predict_thread, std::ref(module), std::ref(tensor_patches), batch_size, std::ref(device), is_log);
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

