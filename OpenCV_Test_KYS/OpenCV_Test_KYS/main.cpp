#include <iostream>
#include "opencv2/highgui.hpp"

#define IMAGE_PATH "test.jpeg"
#define COLOR_IMAGE_WINDOW_NAME "color image"
#define GRAY_IMAGE_WINDOW_NAME "gray image"

using namespace cv;
using namespace std;
int main(int argc, char* argv[]) {
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "OpenCV major version : " << CV_MAJOR_VERSION << endl;
    cout << "OpenCV minor version : " << CV_MINOR_VERSION << endl;
    cout << "OpenCV subminor version : " << CV_SUBMINOR_VERSION << endl;
    
    Mat Image = imread("test.jpeg");
    if(Image.empty()){
        cout << "Could not open find the image" << endl;
        return -1;
    }
    
    namedWindow("test", WINDOW_NORMAL);
    namedWindow("test", WINDOW_AUTOSIZE);
    imshow("test", Image);
    waitKey(0);
    
    destroyWindow("test");
    imwrite("output.png", Image);
    return 0;
}
