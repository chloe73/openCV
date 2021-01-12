#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

    Mat origin = imread("test.jpeg");
    if(origin.empty()){
        cout << "이미지를 불러올 수 없습니다." << endl;
        return -1;
    }
    resize(origin, origin, Size(720, 720));
    
    Mat img;
    cvtColor(origin, img, COLOR_BGR2GRAY);
    GaussianBlur(img, img, Size(9, 9), 0);
    threshold(img, img, 100, 255, 1);
    imshow("binar", img);
    Mat label, stats, centroid;
    int cnt = connectedComponentsWithStats(img, label, stats, centroid, 8, CV_32S);
    int count = 0;
    
    for(int i = 1; i < cnt; i++){
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);
        
        if (area < 200) continue;
        
        rectangle(origin, Rect(x-2, y-2, width+4, height+4), Scalar(0, 255, 255));
        count++;
        stringstream str;
        str << count;
        putText(origin, str.str(), Point(x-2, y+13), FONT_HERSHEY_PLAIN, 1.2, Scalar(255, 255, 255));
    }
    
    imshow("Labeling", origin);
    waitKey();
    return 0;
}
