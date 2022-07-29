#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>



int main()
{
    cv::ocl::setUseOpenCL(true);
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL is not available..." << std::endl;
        return 1;
    }

    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_ALL))
    {
        std::cout << "Failed creating the context..." << std::endl;
        return 1;
    }

    std::cout << context.ndevices() << " GPU devices are detected." << std::endl; //This bit provides an overview of the OpenCL devices you have in your computer
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        std::cout << "name:              " << device.name() << std::endl;
        std::cout << "available:         " << device.available() << std::endl;
        std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << std::endl;
        std::cout << std::endl;
    }

    cv::Mat src = cv::Mat::ones(50, 50, CV_32F);
    cv::UMat u_src = src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

    cv::Mat dst = cv::Mat::ones(50, 50, CV_32F);
    cv::UMat u_dst = dst.getUMat(cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::Mat A_init = cv::Mat::ones(50, 50, CV_32F);
    cv::Mat A = A_init.mul(2);
    cv::Mat B_init = cv::Mat::ones(50, 50, CV_32F);
    cv::Mat B = B_init.mul(2);
    cv::Mat C_init = cv::Mat::ones(50, 50, CV_32F);
    cv::Mat C = C_init.mul(2);
    
    float scale = 2.;

    // Define where src Code is stored
    //
    std::ifstream ifs("matmul.cl");
	if(!ifs.is_open())
	{
		std::cout << "Unable to open matmul.cl" << std::endl;
		return 1;
	}
	std::string kernel_source((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    cv::ocl::ProgramSource program_source(kernel_source);

	// Compile the kernel code
    cv::String errmsg;
	std::cout << __LINE__ << std::endl;
    cv::String buildopt = cv::format("-D dstT=%s", cv::ocl::typeToStr(u_dst.depth())); // "-D dstT=float"
	std::cout << __LINE__ << std::endl;
    cv::ocl::Program program = context.getProg(program_source, buildopt, errmsg);
	std::cout << __LINE__ << std::endl;
    
    // Transfer data to device
    // get kernel by name 
    cv::ocl::Kernel k("mat_mul", program);
	std::cout << __LINE__ << std::endl;
    if (k.empty())
    {
        std::cout << "Can't get OpenCL kernel" << std::endl;
        return 1;
    }
	k.args(A.rows, A.cols, B.rows, B.cols, A, B, dst);
    // k.args(u_src,  cv::ocl::KernelArg::ReadWrite(u_dst));

	size_t gl[2] = {A.cols, A.rows};

	auto t_start = std::chrono::steady_clock::now();
    bool executionResult = k.run(2, gl, NULL, true);
    auto t_end = std::chrono::steady_clock::now();
    
    std::chrono::duration<float> elapsed_seconds = t_end - t_start;


	auto t_start_cpu = std::chrono::steady_clock::now();
	C = A * B;
    auto t_end_cpu = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed_seconds_cpu = t_end_cpu - t_start_cpu;

	cv::Mat mat_dst = u_dst.getMat(cv::ACCESS_READ);
    std::cout << "Compare Results: CPU: " << C << "GPU: " << mat_dst << std::endl;
    std::cout << "CPU Time: " << elapsed_seconds_cpu.count() << std::endl;
    std::cout << "GPU Time: " << elapsed_seconds.count() << std::endl;

    if (!executionResult)
    {
        std::cout << "OpenCL kernel launch failed" << std::endl;
        return 1;
    }


	return 0;
}
