#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#define IMAGE_PATH "/Users/admin/Desktop/Task/Project/opencvTest1/norazo/OpenCV_Test_KYS/OpenCV_Test_KYS/test.jpeg"
#define COLOR_IMAGE_WINDOW_NAME "color image"
#define GRAY_IMAGE_WINDOW_NAME "gray image"

int main(int argc, char** argv) {
    
    cv::Mat imageColor;
    cv::Mat imageGray;

    imageColor = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    imageGray = cv::imread(IMAGE_PATH, cv::IMREAD_GRAYSCALE);

    if (imageColor.empty() || imageGray.empty()){
        std::cout << IMAGE_PATH
            <<" 이미지를 불러오는 데 문제가 생겼습니다." << std::endl;
        return -1;
    }

    cv::namedWindow(COLOR_IMAGE_WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::namedWindow(COLOR_IMAGE_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::imshow(COLOR_IMAGE_WINDOW_NAME, imageColor);
    cv::imshow(GRAY_IMAGE_WINDOW_NAME, imageGray);

    cv::waitKey(0);

    cv::destroyWindow(COLOR_IMAGE_WINDOW_NAME);
    cv::destroyWindow(GRAY_IMAGE_WINDOW_NAME);

    return 0;
}
