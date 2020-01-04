#include<cstdio>
#include<cstdlib>
#include<cmath>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void upside_down(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	// upside-down
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			imgproc.at<uchar>(i, j) = img.at<uchar>(img.rows - 1 - i, j);
		}

	// show image
	imshow("upside-down", imgproc);

	// write image 
	imwrite("upside-down Lena.jpg", imgproc);
}

void right_side_left(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	// right-side-left
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			imgproc.at<uchar>(i, j) = img.at<uchar>(i, img.cols - 1 - j);
		}

	// show image
	imshow("right-side-left", imgproc);

	// write image 
	imwrite("right-side-left Lena.jpg", imgproc);
}

void diagonally_mirrored(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	// diagonally mirrored
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			imgproc.at<uchar>(i, j) = img.at<uchar>(j, i);
		}

	// show image
	imshow("diagonally mirrored", imgproc);

	// write image 
	imwrite("diagonally mirrored Lena.jpg", imgproc);
}

void img_rotation_45(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	int ox = img.cols / 2;
	int oy = img.rows / 2;
	int new_x, new_y;
	double vsin = sin(45 * 0.01745329252);
	double vcos = cos(45 * 0.01745329252);
	// rotate
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			new_x = j - ox;
			new_y = i - oy;
			int x = new_x * vcos + new_y * vsin + ox;
			int y = new_y * vcos + -new_x * vsin + oy;
			if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
			{
				imgproc.at<uchar>(i, j) = img.at<uchar>(y, x);
			}
			else
			{
				imgproc.at<uchar>(i, j) = 0;
			}
		}
	// show image
	imshow("rotate 45 degree", imgproc);

	// write image 
	imwrite("rotate 45 degree Lena.jpg", imgproc);
}

void shrink(Mat img)
{
	Mat imgproc = Mat(img.rows / 2, img.cols / 2, CV_8UC1);
	// shrink
	for (int i = 0; i < imgproc.rows; i++)
		for (int j = 0; j < imgproc.cols; j++)
		{
			imgproc.at<uchar>(i, j) = (img.at<uchar>(2 * i, 2 * j)
				                     + img.at<uchar>(2 * i, 2 * j + 1) 
				                     + img.at<uchar>(2 * i + 1, 2 * j) 
				                     + img.at<uchar>(2 * i + 1, 2 * j + 1)) / 4;
		}
	// show image
	imshow("shrink", imgproc);

	// write image 
	imwrite("shrink Lena.jpg", imgproc);
}

void binarize(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	// binarize
	for (int i = 0; i < imgproc.rows; i++)
		for (int j = 0; j < imgproc.cols; j++)
		{
			if (img.at<uchar>(i, j) >= 128)
			{
				imgproc.at<uchar>(i, j) = 255;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 0;
			}
		}
	// show image
	imshow("binarize", imgproc);

	// write image 
	imwrite("binarize Lena.jpg", imgproc);
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);
	// Part 1.
	upside_down(img);
	right_side_left(img);
	diagonally_mirrored(img);
	// Part 2.
	img_rotation_45(img);
	shrink(img);
	binarize(img);

	waitKey(0);
	return 0;
}