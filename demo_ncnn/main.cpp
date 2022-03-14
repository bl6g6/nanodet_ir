#include "nanodet.h"
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const int color_list[1][1] =
{
    //{255 ,255 ,255}, //bg
    {255},
};

typedef struct BoxInfo 
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

void draw_bboxes(const cv::Mat& bgr, std::vector<nanodet::BoxInfo>& bboxes)
{
    static const char* class_names[] = {"standing", "seated", "lying", "other"};

    cv::Mat image = bgr.clone();
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        nanodet::BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Rect(cv::Point(bbox.x1, bbox.y1),
                                        cv::Point(bbox.x2, bbox.y2)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0));
    }
    cv::imwrite("result.png", image);
}

int main(int argc, char* argv[])
{
    std::string input_file(argv[1]);
    std::string bin_path = "masike_nanodet_V20211215_opt.bin";
    std::string param_path = "masike_nanodet_V20211215_opt.param";

    cv::Mat image = cv::imread(input_file, 0);  //  将3通道数据转成单通道
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", input_file.c_str());
        return -1;
    }
    const unsigned char *pixels = image.data;
    auto net = nanodet::NanoDet();
    net.init(bin_path, param_path);
    auto start2 =  std::chrono::steady_clock::now();
    for(int i = 0; i < 1; i++){
        auto start = std::chrono::steady_clock::now();
        auto results = net.detect(pixels, 320, 240, 320, 0.35, 0.4, false);
        auto end = std::chrono::steady_clock::now();
        auto cost_time = std::chrono::duration<double, std::milli>(end - start).count();
        printf("inference  time: %7.2f ms. \n", cost_time);
        draw_bboxes(image, results);
    }
    auto end2 = std::chrono::steady_clock::now();
    auto cost_time_total = std::chrono::duration<double, std::milli>(end2 - start2).count();
    printf("average inference time: %f ms. \n", cost_time_total/1000);
    return 0;
}


//
//#include <sys/types.h>
//#include <dirent.h>
//#include <errno.h>
//#include <vector>
//#include <string>
//#include <iostream>
//
//using namespace std;
//
///*function... might want it in some class?*/
//int getdir (string dir, vector<string> &files)
//{
//    DIR *dp;
//    struct dirent *dirp;
//    if((dp  = opendir(dir.c_str())) == NULL) {
//        cout << "Error(" << errno << ") opening " << dir << endl;
//        return errno;
//    }
//
//    while ((dirp = readdir(dp)) != NULL) {
//        files.push_back(string(dirp->d_name));
//    }
//    closedir(dp);
//    return 0;
//}
//
//int main(int argc, char* argv[])
//{
//    std::string input_file(argv[1]);
//    std::string bin_path = "V20210615B_nanodet.bin";
//    std::string param_path = "V20210615B_nanodet.param";
//
//    auto net = nanodet::NanoDet();
//    net.init(bin_path, param_path);
//
//    vector<string> files = vector<string>();
//    getdir(input_file,files);
//    string files_path;
//    for (unsigned int i = 0;i < files.size();i++) {
//        files_path = input_file + files[i];
////        cout << files_path << endl;
////        cout << files[i] << endl;
//        cv::Mat image = cv::imread(files_path, CV_LOAD_IMAGE_GRAYSCALE);  //  将3通道数据转成单通道
//        if (image.empty())
//        {
//            fprintf(stderr, "cv::imread %s failed\n", input_file);
//            return -1;
//        }
//        const unsigned char *pixels = image.data;
//        clock_t start = clock();
//        auto results = net.detect(pixels, 320, 240, 320, 0.4, 0.5, false);
//        clock_t end = clock();
//        double cost_time = (double)(end - start)/CLOCKS_PER_SEC;
//        printf("Total used time: %f ms. \n", cost_time*1000);
//
//    }
//    return 0;
//}
