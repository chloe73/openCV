#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

#define NUMBER "CAR.jpeg"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[]){
    // 이미지 원본
    Mat img = imread(NUMBER, IMREAD_COLOR);
    // 원본 흑백화
    Mat gray ;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // 블러
    Mat blur;
    GaussianBlur(gray, blur, Size(7, 7), 0);
    // 외곽선
    Mat canny;
    Canny(blur, canny, 90, 180);
    // 외곽선 찾기
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    int index_count = 0;
    Rect* temp = (Rect*)malloc(sizeof(Rect)*contours.size());
//    Rect temp[contours.size()];
    for(int i = 0; i < contours.size(); i++){
        vector<Point> cnt = contours[i];
        double area = contourArea(cnt);
        Rect rect = boundingRect(cnt);
        double aspect_ratio = (double)rect.width/rect.height;
        
        if ((aspect_ratio>=0.3)&&(aspect_ratio<=2)&&(rect.area() >= 150)&&(rect.area() <= 1000)){
            rectangle(img, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(0,255,0), 1);
            temp[index_count++] = boundingRect(cnt);
        }
    }
    Rect box[index_count];
    for(int i = 0; i < index_count; i++){
        box[i] = temp[i];
    }
    free(temp);
    
    for(int i = 0; i < index_count; i++){
        for(int j = 0; j < index_count - (i+1); j++){
            Rect temp = box[j];
            box[j] = box[j+1];
            box[j+1] = temp;
        }
    }
    int f_count = 0;
    int select = 0;
    int plate_whidth = 0;
    
    for(int i = 0; i < index_count; i++){
        int count = 0;
        for(int j = i + 1; j < index_count - 1; j++){
            double delta_x = abs(box[j+1].x - box[i].x);
            if (delta_x > 150) break;
            
            double delta_y = abs(box[j+1].y - box[i].y);
            if(delta_x == 0) delta_x = 1;
            if(delta_y == 0) delta_y = 1;
            
            double gradient = delta_y/delta_x;
            if(gradient < 0.25) count++;
            
            if (count > f_count){
                select = i;\
                f_count = count;
                plate_whidth = delta_x;
            }
        }
    }
    
    //Mat number_plate = gray(Range(box[select.y]-10,box[sel]

    imshow("img", img);
//    imshow("gray", gray);
//    imshow("blur", blur);
//    imshow("canny", canny);
    //cout << "contours len : " << contours.size() << endl;
    
    waitKey(0);
    
    return 0;
}
