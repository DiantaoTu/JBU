#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

cv::Mat DownSample(cv::Mat source)
{
    cv::Mat dst;
    cv::pyrDown(source, dst);
    return dst;
}

//test

cv::Mat JointBilateralUpsample(cv::Mat high, cv::Mat low, const int halfWindow = 5,const float sigma_d = 0.5, const float sigma_r = 0.1)
{
    // https://www.jianshu.com/p/ce4afe599d6a 
    int width = high.cols;
    int height = high.rows;
    float factor = 2;   // 高像素与低像素图像尺度之比为2
    cv::Mat upSampled = cv::Mat::zeros(high.size(), CV_8UC3);   // 上采样后的图像是彩色的
    for(int i=0; i < height; i++)
        for(int j=0; j < width; j++)
        {
            cv::Vec3b p = high.at<cv::Vec3b>(i,j);
            // 确定当前像素支撑窗口的边界
            float low_i = i/factor;
            float low_j = j/factor;
            int iMax = floor(min(low.rows - 1.f, low_i + halfWindow));
            int iMin = ceil(max(0.f, low_i - halfWindow));
            int jMax = floor(min(low.cols - 1.f, low_j + halfWindow));
            int jMin = ceil(max(0.f, low_j - halfWindow));
            // 计算f函数，结果存储在spatial中
            cv::Mat lowWindow = low.rowRange(iMin, iMax+1).colRange(jMin, jMax+1).clone();
            cv::Mat spatial = cv::Mat::zeros(lowWindow.size(), CV_32F);
            for(int m=0; m<spatial.rows; m++)
                for(int n=0; n<spatial.cols; n++)
                {
                    float x = iMin + m - low_i;
                    float y = jMin + n - low_j;
                    spatial.at<float>(m,n) = exp(-(x*x+y*y)/(2*sigma_d*sigma_d));
                }
            // highWindow是像素P在高分辨率下的支撑窗口，如果低分辨率时窗口为5*5
            // 高分辨率下应该就是10*10，但公式中要求二者一致，因此高分辨率下窗口也是5*5
            // 这就需要对窗口的数据降采样，隔一行采样一行，隔一列采样一列
            cv::Mat highWindow = cv::Mat::zeros(spatial.size(), CV_8UC3);
            for(int m=0; m < spatial.rows; m++)
                for(int n=0; n < spatial.cols; n++)
                {
                    highWindow.at<cv::Vec3b>(m,n) = high.at<cv::Vec3b>((iMin+m)*factor, (jMin+n)*factor);
                }
            
            // 计算g函数，结果存储在range里
            cv::Mat range = cv::Mat::zeros(highWindow.size(), CV_32F);  // range kernel filter 也就是g函数
            cv::MatIterator_<cv::Vec3b> highBegin = highWindow.begin<cv::Vec3b>();
            cv::MatIterator_<cv::Vec3b> highEnd = highWindow.end<cv::Vec3b>();
            cv::MatIterator_<float> rangeBegin = range.begin<float>();
            while(highBegin != highEnd)
            {
                float B = ((*highBegin)[0] - p[0]) / 255.f;
                float G = ((*highBegin)[1] - p[1]) / 255.f;
                float R = ((*highBegin)[2] - p[2]) / 255.f;
                *rangeBegin = exp(-(B*B + G*G + R*R)/(2*sigma_r*sigma_r));
                highBegin++;
                rangeBegin++;
            }

            cv::Mat spatial_range = cv::Mat::zeros(range.size(), CV_32F);
            spatial_range = spatial.mul(range);
            float Kp = cv::sum(spatial_range)[0];
            cv::Vec3f new_p(0,0,0);
            cv::MatIterator_<float> sumBegin = spatial_range.begin<float>();
            // cv::MatIterator_<float> rangeBegin = range.begin<float>();
            cv::MatIterator_<float> sumEnd = spatial_range.end<float>();
            cv::MatIterator_<cv::Vec3b> lowBegin = lowWindow.begin<cv::Vec3b>();
            while(sumBegin != sumEnd)
            {
                new_p += (*sumBegin) * (*lowBegin);
                sumBegin++;
                lowBegin++;
            }
            // cv::Mat_<cv::Vec3b> sumWeight = lowWindow.mul(spatial_range);
            // cv::Vec4f new_p = cv::sum(sumWeight)/Kp;

            new_p /= Kp;

            upSampled.at<Vec3b>(i,j)[0] = (uchar)new_p[0];
            upSampled.at<Vec3b>(i,j)[1] = (uchar)new_p[1];
            upSampled.at<Vec3b>(i,j)[2] = (uchar)new_p[2];

        }
    return upSampled;
}

int main(int argc, char** argv)
{
    Mat img = imread("../raw.jpg");
    cout<<"before sample rows:"<<img.rows<<" cols:"<<img.cols<<endl;

    Mat downSample = DownSample(img);
    cout<<"after down sample rows:"<<downSample.rows<<" cols:"<<downSample.cols<<endl;
    imwrite("../downSample.jpg", downSample);

    Mat upSample = JointBilateralUpsample(img, downSample);
    cout<<"after up sample rows:"<<upSample.rows<<" cols:"<<upSample.cols<<endl;
    imwrite("../upSample.jpg",upSample);
    return 0;
}


