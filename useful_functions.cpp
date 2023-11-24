#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 
#include <time.h>
#include <vector>

#ifdef _WIN32 
#include <io.h>
#include <windows.h>
#include <objbase.h>
#define MKDIR(a) _mkdir((a))
#define ACCESS(a, b) _access((a), (b))
#define SLEEP(a) Sleep((a))
constexpr auto PATH_SEPARATOR = "\\";

#else
// linux
#include "dirent.h"
#include <sys/stat.h>
#include <uuid/uuid.h>
#define MKDIR(a) mkdir((a),0755)
#define ACCESS(a, b) access((a), (b))
#define SLEEP(a) usleep((a))
constexpr auto PATH_SEPARATOR = "/";
#endif

using namespace cv;
using namespace std;

#ifdef _WIN32
string GetInputFileName(string root, string suffix)
{
	long long handle = 0;
	struct _finddata_t fileinfo;

	if ((handle = _findfirst((root + "/*" + suffix).c_str(), &fileinfo)) != -1)
	{
		return root + "/" + fileinfo.name;
	}
	else
	{
		return "";
	}
}
#else
/**
     * 判断是否是一个文件
     */
static bool is_file(std::string filename) 
{
	struct stat buffer;
	return (stat (filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}
 
/**
 * 判断是否是一个文件夹,
 * */
static bool is_dir(std::string filefodler) 
{
	struct stat buffer;
	return (stat (filefodler.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

vector<string> GetInputFileName(string root, string suffix)
{
	DIR* dirptr = NULL;
	struct dirent* dirp;
	vector<string> files;

	if (is_file(root)){
		if (0 == suffix.compare(root.substr(root.size() - 3, 3))){
			files.push_back(root);
		}
	}
	else if (is_dir(root)){
		if (!((dirptr = opendir(root.c_str())) == NULL)){
			while ((dirp = readdir(dirptr)) != NULL)
			{
				string file_name(dirp->d_name);
				if ((dirp->d_type == DT_REG) && (0 == suffix.compare(file_name.substr(file_name.size() - 3, 3))))
				{
					files.push_back(root + "/" + string(dirp->d_name));
				}
			}
		closedir(dirptr);
		}
	}
	return files;
}
#endif


int main(int argc, char *argv[])
{
	string input_file = "/Users/chenkeyan/MyCode/Torch_Deploy/Examples/SV2-01-PMS-126-691-L00000022796.jpg";
	std::string model_pb = "/Users/chenkeyan/MyCode/Torch_Deploy/Weights_Deploy/torch_scritp_model.pt";
	int is_output_mask = 0;
	int is_quiet = 0;
	int img_size = 1024;

	if (argc == 1) {
		if (!is_quiet) {
		cout << "Load default image!" << endl;
	}
	}
	else if (argc == 2) {
		input_file = argv[1];
	}
	else if (argc == 3) {
		input_file = argv[1];
		model_pb = argv[2];
	}
	else if (argc == 4) {
		input_file = argv[1];
		model_pb = argv[2];
		is_output_mask = stoi(argv[3]);
	}
	else if (argc == 5) {
		input_file = argv[1];
		model_pb = argv[2];
		is_output_mask = stoi(argv[3]);
		is_quiet = stoi(argv[4]);
	}
	else
	{
		cout << "Should Be image_path:string  model_path:string  is_output_mask:int is_quiet:int" << endl;
		throw "Usage: " + string(argv[0]) + " Fail!";
		return -1;
	}

	if (!is_quiet) {
		cout << "Load image!" << endl;
	}
	// torch::DeviceType device;
	// if (torch::cuda::is_available()) {
	// 	std::cout << "CUDA available! Predicting on GPU." << std::endl;
	// 	device = torch::kCUDA;
	// }
	// else
	// {
	// 	std::cout << "Predicting on CPU." << std::endl;
	// 	device = torch::kCPU;
	// }
	// auto device_type = torch::Device(device);

	auto img_files = GetInputFileName(input_file, "jpg");
	if (img_files.size() == 0){
		if (!is_quiet) {
		cout << "No Image Found!" << endl;
		}
		return 0;
	}
	if (!is_quiet) {
		cout << "Load Model!" << endl;
	}
	auto device_type = torch::Device(torch::kCPU);
	auto model = torch::jit::load(model_pb);
	model.to(device_type);
	model.eval();

	for(int i=0; i<img_files.size(); i++){
		// const clock_t begin_time = clock();
		string img_path = img_files.at(i);
		int start = img_path.rfind(PATH_SEPARATOR) + strlen(PATH_SEPARATOR);
		int end = img_path.rfind(".");
		string img_name_no_sufix = img_path.substr(start, end - start);

		int last_slash_idx = img_path.rfind(PATH_SEPARATOR);
		string folder = img_path.substr(0, last_slash_idx);

		if (!is_quiet) {
			cout << "Procesing " << i << " image, name: " << img_name_no_sufix << ".jpg" << endl;
		}

		Mat image = cv::imread(img_path);
		cvtColor(image, image, COLOR_BGR2RGB);
		int src_w = image.cols;
		int src_h = image.rows;
		float h_w_ratio = src_h / src_w;
		cv::Mat img_patch;
		if (h_w_ratio < 1.55){
			cv::resize(image, img_patch, cv::Size(img_size, img_size));
		}
		else{
			cv::resize(image, img_patch, cv::Size(img_size, img_size*(int(src_h / src_w))));
		}
		auto input_tensor = torch::from_blob(img_patch.data, { img_patch.rows, img_patch.cols, 3 }, torch::kByte);
		input_tensor = input_tensor.view({ -1, img_size, img_size,  3 });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).toType(torch::kFloat32);
		auto output = model.forward({ input_tensor.to(device_type) }).toTensor();
		output = torch::argmax(output, 1);
		output = output.view({ -1, img_size });

		auto pred_label = output.mul_(128).toType(torch::kByte);
		cv::Mat pred_label_mat(cv::Size(pred_label.size(1), pred_label.size(0)), CV_8U, pred_label.data_ptr());
		cv::resize(pred_label_mat, pred_label_mat, {src_w, src_h}, cv::InterpolationFlags::INTER_NEAREST);
		cv::imwrite(folder+"/"+img_name_no_sufix+".png", pred_label_mat);
		// cv::imshow("show", pred_label_mat);
		// cv::waitKey(0);
	}
	
	// pred_label = pred_label.permute({ 1,0 });
	// cv::imshow("show", img_patch);
	// cv::waitKey(0);

	// int width_step = width / width_num;
	// int height_step = height / height_num;

	// cv::Mat img_patch_transformed;
	// for (int i = 0; i <= width_num; i++) {
	// 	auto width_start = i * width_step;
	// 	auto width_end = ((i + 1) * width_step < width ? ((i + 1) * width_step) : width);
	// 	for (int j = 0; j <= height_num; j++) {
	// 		// cout << j << endl;
	// 		auto height_start = j * height_step;
	// 		auto height_end = ((j + 1) * height_step < height ? ((j + 1) * height_step) : height);
			// auto img_patch = image(cv::Range(height_start, height_end), cv::Range(width_start, width_end)).clone();
	// 		/*cv::imshow("show", img_patch);
	// 		cv::waitKey(0);*/

	// 		//������ָ����С
	// 		cv::resize(img_patch, img_patch_transformed, cv::Size(80, 160));
			
	// 		//ת������
	// 		auto input_tensor = torch::from_blob(img_patch_transformed.data, { img_patch_transformed.rows, img_patch_transformed.cols, 3 }, 
	// 			torch::kByte);
	// 		input_tensor = input_tensor.permute({ 2,0,1 }).toType(torch::kFloat32);
	// 		// std::cout << input_tensor.size(0) << input_tensor.size(1);
	// 		input_tensor[0] = input_tensor[0].div_(255).sub_(0.2619703004633249).div_(0.021910381946245575);
	// 		input_tensor[1] = input_tensor[1].div_(255).sub_(0.16063937884372356).div_(0.015363815458932483);
	// 		input_tensor[2] = input_tensor[2].div_(255).sub_(0.11861206329461361).div_(0.01316140398841059);
	// 		// std::cout << input_tensor[0][1];
	// 		input_tensor = input_tensor.unsqueeze(0);
	// 		//ǰ�򴫲�
	// 		auto output = model.forward({ input_tensor.to(device_type) }).toTensor();
	// 		//std::cout << output.size(0) << output.size(1) << output.size(2) << output.size(3);
	// 		output = torch::argmax(output, 1);
	// 		auto pred_label = output.mul_(255).squeeze_(0).toType(torch::kByte);
	// 		pred_label = pred_label.permute({ 1,0 });
			
	// 		cv::Mat pred_label_mat(cv::Size(40, 80), CV_8U, pred_label.data_ptr());
	// 		/*cv::imshow("show", pred_label_mat);
	// 		cv::waitKey(0);*/

	// 		cv::resize(pred_label_mat, pred_label_mat, cv::Size(img_patch.cols, img_patch.rows));
	// 		cv::morphologyEx(pred_label_mat, pred_label_mat, cv::MORPH_CLOSE, cv::Mat(3, 3, CV_8U, cv::Scalar(1)));
	// 		std::vector< std::vector< cv::Point> > contours;
	// 		cv::findContours(pred_label_mat, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	// 		int area = 0;
	// 		int id = 0;
	// 		for (int k = 0; k < contours.size(); k++) {
	// 			auto area_tmp = cv::contourArea(contours.at(k));
	// 			if (area_tmp > area) {
	// 				area = area_tmp;
	// 				id = k;
	// 			}
	// 		}
	// 		cv::drawContours(img_patch, contours, id, cv::Scalar(255, 0, 0), 1, 8);
	// 		Mat tmp = image(cv::Range(height_start, height_end), cv::Range(width_start, width_end));
	// 		img_patch.copyTo(tmp);
	// 		cv::cvtColor(image, image, cv::COLOR_RGB2BGR); 
			
	// 	}
	// }
	// float seconds = float(clock() - begin_time) / 1000; //�˴�1000ָ����ÿ��Ϊ1000��ʱ�����ڣ�����Ҫ��õ�����Ϊ��λ��ʱ�䣬��Ҫ����1000.
	// cout << "Time:" << seconds << endl;
	// system("Pause");
	// cv::imwrite("show.bmp", image);
	return 0;
}