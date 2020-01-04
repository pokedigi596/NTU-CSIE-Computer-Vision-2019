#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

void kernal(Mat img, Mat imgproc, int y, int x, int mode)
{
	int target_pixel = (mode == 0 ? 0 : 255);
	for (int i = -2; i < 3; i++)
	{
		for (int j = -2; j < 3; j++)
		{
			target_pixel = ((i == -2 && j == -2) ||
							(i == -2 && j == 2) ||
							(i == 2 && j == -2) ||
							(i == 2 && j == 2) || 
							(mode == 0 ? (img.at<uchar>(i + y, j + x) < target_pixel) : (img.at<uchar>(i + y, j + x) > target_pixel))) ?
							target_pixel : img.at<uchar>(i + y, j + x);
		}
	}

	for (int i = -2; i < 3; i++)
	{
		for (int j = -2; j < 3; j++)
		{
			imgproc.at<uchar>(i + y, j + x) = ((i == -2 && j == -2) ||
											   (i == -2 && j == 2) ||
											   (i == 2 && j == -2) ||
											   (i == 2 && j == 2)) ? imgproc.at<uchar>(i + y, j + x) : target_pixel;
			
		}
	}
}

void dilation(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(img, imgproc, i, j, 0);
		}
	}

	imshow("Dilation Image", imgproc);

	imwrite("Dilation Image.jpg", imgproc);
}

void erosion(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(img, imgproc, i, j, 1);
		}
	}

	imshow("Erosion Image", imgproc);

	imwrite("Erosion Image.jpg", imgproc);
}

void opening(Mat img)
{
	Mat imgproc0 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(img, imgproc0, i, j, 1);
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(imgproc0, imgproc1, i, j, 0);
		}
	}

	imshow("Opening Image", imgproc1);

	imwrite("Opening Image.jpg", imgproc1);
}

void closing(Mat img)
{
	Mat imgproc0 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(img, imgproc0, i, j, 0);
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(imgproc0, imgproc1, i, j, 1);
		}
	}

	imshow("Closing Image", imgproc1);

	imwrite("Closing Image.jpg", imgproc1);
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	dilation(img);
	erosion(img);
	opening(img);
	closing(img);
	
	waitKey(0);
	return 0;
}