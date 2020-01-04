#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<string>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

int gaussian_kernel[11][11] = { {0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0},
							    {0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0},
							    {0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0},
							    {-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1},
							    {-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1},
							    {-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2},
							    {-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1},
							    {-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1},
							    {0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0},
							    {0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0},
							    {0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0} };

int difference_kernel[11][11] = { {-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1},
								{-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3},
								{-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4},
								{-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6},
								{-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7},
								{-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8},
								{-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7},
								{-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6},
								{-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4},
								{-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3},
								{-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1} };

Mat Laplacian0(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel[3][3];
			for (int x = 0; x < 3; x++)
			{
				for (int y = 0; y < 3; y++)
				{
					kernel[x][y] = (i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= img.rows || j - 1 + y >= img.cols) ? 0 : img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int pixel = kernel[1][1] * (-4) + kernel[0][1] + kernel[1][0] + kernel[1][2] + kernel[2][1];
			if (pixel >= threshold)
			{
				imgproc.at<uchar>(i, j) = 2;
			}
			else if(pixel <= -threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (imgproc.at<uchar>(i, j) == 2)
			{
				bool img_flag = true;
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						if ((i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= imgproc.rows || j - 1 + y >= imgproc.cols)) continue;
						else
						{
							if (imgproc.at<uchar>(i - 1 + x, j - 1 + y) == 0)
							{
								img_flag = img_flag & false;
							}
							else
							{
								img_flag = img_flag & true;
							}
						}
					}
				}
				imgproc1.at<uchar>(i, j) = (img_flag) ? 255 : 0;
			}
			else 
			{
				imgproc1.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc1;
}

Mat Laplacian1(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel[3][3];
			for (int x = 0; x < 3; x++)
			{
				for (int y = 0; y < 3; y++)
				{
					kernel[x][y] = (i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= img.rows || j - 1 + y >= img.cols) ? 0 : img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int pixel = (kernel[1][1] * (-8) + kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[1][0] + kernel[1][2] + kernel[2][0] + kernel[2][1] + kernel[2][2]) / 3;
			if (pixel >= threshold)
			{
				imgproc.at<uchar>(i, j) = 2;
			}
			else if (pixel <= -threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (imgproc.at<uchar>(i, j) == 2)
			{
				bool img_flag = true;
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						if ((i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= imgproc.rows || j - 1 + y >= imgproc.cols)) continue;
						else
						{
							if (imgproc.at<uchar>(i - 1 + x, j - 1 + y) == 0)
							{
								img_flag = img_flag & false;
							}
							else
							{
								img_flag = img_flag & true;
							}
						}
					}
				}
				imgproc1.at<uchar>(i, j) = (img_flag) ? 255 : 0;
			}
			else
			{
				imgproc1.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc1;
}

Mat Minimum_variance(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel[3][3];
			for (int x = 0; x < 3; x++)
			{
				for (int y = 0; y < 3; y++)
				{
					kernel[x][y] = (i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= img.rows || j - 1 + y >= img.cols) ? 0 : img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int pixel = (kernel[1][1] * (-4) + kernel[0][0] * 2 - kernel[0][1] + kernel[0][2] * 2 - kernel[1][0] - kernel[1][2] + kernel[2][0] * 2 - kernel[2][1] + kernel[2][2] * 2) / 3;
			if (pixel >= threshold)
			{
				imgproc.at<uchar>(i, j) = 2;
			}
			else if (pixel <= -threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (imgproc.at<uchar>(i, j) == 2)
			{
				bool img_flag = true;
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						if ((i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= imgproc.rows || j - 1 + y >= imgproc.cols)) continue;
						else
						{
							if (imgproc.at<uchar>(i - 1 + x, j - 1 + y) == 0)
							{
								img_flag = img_flag & false;
							}
							else
							{
								img_flag = img_flag & true;
							}
						}
					}
				}
				imgproc1.at<uchar>(i, j) = (img_flag) ? 255 : 0;
			}
			else
			{
				imgproc1.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc1;
}

Mat Gaussian(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel_sum = 0;
			for (int x = 0; x < 11; x++)
			{
				for (int y = 0; y < 11; y++)
				{
					int temp = (i - 11 + x < 0 || j - 11 + y < 0 || i - 11 + x >= img.rows || j - 11 + y >= img.cols) ? 0 : img.at<uchar>(i - 11 + x, j - 11 + y);
					kernel_sum += temp * gaussian_kernel[x][y];
				}
			}
			if (kernel_sum >= threshold)
			{
				imgproc.at<uchar>(i, j) = 2;
			}
			else if (kernel_sum <= -threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (imgproc.at<uchar>(i, j) == 2)
			{
				bool img_flag = true;
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						if ((i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= imgproc.rows || j - 1 + y >= imgproc.cols)) continue;
						else
						{
							if (imgproc.at<uchar>(i - 1 + x, j - 1 + y) == 0)
							{
								img_flag = img_flag & false;
							}
							else
							{
								img_flag = img_flag & true;
							}
						}
					}
				}
				imgproc1.at<uchar>(i, j) = (img_flag) ? 255 : 0;
			}
			else
			{
				imgproc1.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc1;
}

Mat Difference(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel_sum = 0;
			for (int x = 0; x < 11; x++)
			{
				for (int y = 0; y < 11; y++)
				{
					int temp = (i - 11 + x < 0 || j - 11 + y < 0 || i - 11 + x >= img.rows || j - 11 + y >= img.cols) ? 0 : img.at<uchar>(i - 11 + x, j - 11 + y);
					kernel_sum += temp * difference_kernel[x][y];
				}
			}
			if (kernel_sum >= threshold)
			{
				imgproc.at<uchar>(i, j) = 2;
			}
			else if (kernel_sum <= -threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (imgproc.at<uchar>(i, j) == 2)
			{
				bool img_flag = true;
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						if ((i - 1 + x < 0 || j - 1 + y < 0) || (i - 1 + x >= imgproc.rows || j - 1 + y >= imgproc.cols)) continue;
						else
						{
							if (imgproc.at<uchar>(i - 1 + x, j - 1 + y) == 0)
							{
								img_flag = img_flag & false;
							}
							else
							{
								img_flag = img_flag & true;
							}
						}
					}
				}
				imgproc1.at<uchar>(i, j) = (img_flag) ? 255 : 0;
			}
			else
			{
				imgproc1.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc1;
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	Mat result = Mat(img.rows, img.cols, CV_8UC1);

	result = Laplacian0(img, 15);
	imshow("Laplacian_4", result);
	imwrite("Laplacian_4.jpg", result);
	result = Laplacian1(img, 15);
	imshow("Laplacian_8", result);
	imwrite("Laplacian_8.jpg", result);
	result = Minimum_variance(img, 20);
	imshow("minimum-variance Laplacian", result);
	imwrite("minimum-variance Laplacian.jpg", result);
	result = Gaussian(img, 3000);
	imshow("Laplacian of Gaussian", result);
	imwrite("Laplacian of Gaussian.jpg", result);
	result = Difference(img, 1);
	imshow("Difference of Gaussian", result);
	imwrite("Difference of Gaussian.jpg", result);

	waitKey(0);
	return 0;
}