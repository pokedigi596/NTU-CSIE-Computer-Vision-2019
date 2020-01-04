#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

void binarize(Mat img)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<uchar>(i, j) = img.at<uchar>(i, j) < 128 ? 0 : 255;
		}
	}
}

bool kernal(Mat img, Mat imgproc, int y, int x, int mode)
{
	bool filter = true;
	Mat temp = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = -2; i < 3; i++)
	{
		for (int j = -2; j < 3; j++)
		{
			if (mode == 0)
			{
				imgproc.at<uchar>(y + i, x + j) = ((i == -2 && j == -2) ||
											   (i == -2 && j == 2) ||
											   (i == 2 && j == -2) ||
											   (i == 2 && j == 2)) ? imgproc.at<uchar>(y + i, x + j) : 255;
			}
			if (mode == 1)
			{
				filter = filter & (((i == -2 && j == -2) || 
									(i == -2 && j == 2) || 
									(i == 2 && j == -2) || 
									(i == 2 && j == 2)) ? true : img.at<uchar>(y + i, x + j) == 255);
			}
		}
	}
	return filter;
}

void dilation(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			if (img.at<uchar>(i, j) == 255)
			{
				kernal(img, imgproc, i, j, 0);
			}
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
			if (img.at<uchar>(i, j) == 255)
			{
				imgproc.at<uchar>(i, j) = kernal(img, imgproc, i, j, 1) ? 255 : 0;
			}
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
			if (img.at<uchar>(i, j) == 255)
			{
				imgproc0.at<uchar>(i, j) = kernal(img, imgproc0, i, j, 1) ? 255 : 0;
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			if (imgproc0.at<uchar>(i, j) == 255)
			{
				kernal(imgproc0, imgproc1, i, j, 0);
			}
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
			if (img.at<uchar>(i, j) == 255)
			{
				kernal(img, imgproc0, i, j, 0);
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			if (imgproc0.at<uchar>(i, j) == 255)
			{
				imgproc1.at<uchar>(i, j) = kernal(imgproc0, imgproc1, i, j, 1) ? 255 : 0;
			}
		}
	}

	imshow("Closing Image", imgproc1);

	imwrite("Closing Image.jpg", imgproc1);
}

void hit_and_miss(Mat img)
{
	Mat img_inv0 = Mat(img.rows, img.cols, CV_8UC1);
	Mat img_inv1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img_inv0.at<uchar>(i, j) = img.at<uchar>(i, j) == 255 ? 0 : 255;
		}
	}
	for (int i = 1; i < img_inv0.rows; i++)
	{
		for (int j = 0; j < img_inv0.cols - 1; j++)
		{
			if (img_inv0.at<uchar>(i, j) == 0)
			{
				bool filter = true;
				for (int x = -1; x < 1; x++)
				{
					for (int y = 0; y < 2; y++)
					{
						filter = filter & ((x == 0 && y == 0) ? true : img_inv0.at<uchar>(x + i, y + j) == 255);
					}
				}
				img_inv1.at<uchar>(i, j) = filter ? 255 : 0;
			}
		}
	}

	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	
	for (int i = 0; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 255)
			{
				bool filter = true;
				for (int x = 0; x < 2; x++)
				{
					for (int y = -1; y < 1; y++)
					{
						filter = filter & ((x == 1 && y == -1) ? true : img.at<uchar>(x + i, y + j) == 255);
					}
				}
				imgproc.at<uchar>(i, j) = filter ? 255 : 0;
			}
		}
	}

	Mat final_image = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			final_image.at<uchar>(i, j) = (imgproc.at<uchar>(i, j) == 255 && img_inv1.at<uchar>(i, j) == 255) ? 255 : 0;
		}
	}

	imshow("Hit and Miss Image", final_image);

	imwrite("Hit and Miss Image.jpg", final_image);
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	binarize(img);

	dilation(img);
	erosion(img);
	opening(img);
	closing(img);
	hit_and_miss(img);
	
	waitKey(0);
	return 0;
}