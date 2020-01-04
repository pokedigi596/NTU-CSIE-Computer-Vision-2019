#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<string>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

Mat Roberts(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel[2][2];
			for (int x = 0; x < 2; x++)
			{
				for (int y = 0; y < 2; y++)
				{
					kernel[x][y] = (i + x >= img.rows || j + y >= img.cols) ? 0 : img.at<uchar>(i + x, j + y);
				}
			}
			double f1 = (double)kernel[0][0] - (double)kernel[1][1];
			double f2 = (double)kernel[0][1] - (double)kernel[1][0];
			double gradient = pow(pow(f1, 2) + pow(f2, 2), 0.5);
			if (gradient > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

Mat Prewitt(Mat img, int threshold)
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
					if (i - 1 + x < 0 || j - 1 + y < 0) kernel[x][y] = 0;
					else kernel[x][y] = img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int f1 = (kernel[2][0] + kernel[2][1] + kernel[2][2]) - (kernel[0][0] + kernel[0][1] + kernel[0][2]);
			int f2 = (kernel[0][2] + kernel[1][2] + kernel[2][2]) - (kernel[0][0] + kernel[1][0] + kernel[2][0]);
			double gradient = pow(pow(f1, 2) + pow(f2, 2), 0.5);
			if (gradient > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

Mat Sobel(Mat img, int threshold)
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
					if (i - 1 + x < 0 || j - 1 + y < 0) kernel[x][y] = 0;
					else kernel[x][y] = img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int f1 = (kernel[2][0] + kernel[2][1] * 2 + kernel[2][2]) - (kernel[0][0] + kernel[0][1] * 2 + kernel[0][2]);
			int f2 = (kernel[0][2] + kernel[1][2] * 2 + kernel[2][2]) - (kernel[0][0] + kernel[1][0] * 2 + kernel[2][0]);
			double gradient = pow(pow(f1, 2) + pow(f2, 2), 0.5);
			if (gradient > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

Mat Frei_and_Chen(Mat img, int threshold)
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
					if (i - 1 + x < 0 || j - 1 + y < 0) kernel[x][y] = 0;
					else kernel[x][y] = img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int f1 = (kernel[2][0] + pow(kernel[2][1], 0.5) + kernel[2][2]) - (kernel[0][0] + pow(kernel[0][1], 0.5) + kernel[0][2]);
			int f2 = (kernel[0][2] + pow(kernel[1][2], 0.5) + kernel[2][2]) - (kernel[0][0] + pow(kernel[1][0], 0.5) + kernel[2][0]);
			double gradient = pow(pow(f1, 2) + pow(f2, 2), 0.5);
			if (gradient > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

Mat Kirsch(Mat img, int threshold)
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
					if (i - 1 + x < 0 || j - 1 + y < 0) kernel[x][y] = 0;
					else kernel[x][y] = img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int k[8], maxval = -INT_MAX - 1;
			k[0] = 5 * (kernel[0][0] + kernel[1][0] + kernel[2][0]) - 3 * (kernel[0][0] + kernel[0][1] + kernel[1][0] + kernel[2][0] + kernel[2][1]);
			k[1] = 5 * (kernel[0][1] + kernel[0][2] + kernel[1][2]) - 3 * (kernel[0][0] + kernel[1][0] + kernel[2][0] + kernel[2][1] + kernel[2][2]);
			k[2] = 5 * (kernel[0][0] + kernel[0][1] + kernel[0][2]) - 3 * (kernel[1][0] + kernel[1][2] + kernel[2][0] + kernel[2][1] + kernel[2][2]);
			k[3] = 5 * (kernel[0][0] + kernel[0][1] + kernel[1][0]) - 3 * (kernel[0][2] + kernel[1][2] + kernel[2][0] + kernel[2][1] + kernel[2][2]);
			k[4] = 5 * (kernel[0][0] + kernel[1][0] + kernel[2][0]) - 3 * (kernel[0][1] + kernel[0][2] + kernel[1][2] + kernel[2][1] + kernel[2][2]);
			k[5] = 5 * (kernel[1][0] + kernel[2][0] + kernel[2][1]) - 3 * (kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[1][2] + kernel[2][2]);
			k[6] = 5 * (kernel[2][0] + kernel[2][1] + kernel[2][2]) - 3 * (kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[1][0] + kernel[1][2]);
			k[7] = 5 * (kernel[1][2] + kernel[2][1] + kernel[2][2]) - 3 * (kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[1][0] + kernel[2][0]);
			for (int x = 0; x < 8; x++)
			{
				maxval = max(maxval, k[x]);
			}
			if (maxval > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

Mat Robinson(Mat img, int threshold)
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
					if (i - 1 + x < 0 || j - 1 + y < 0) kernel[x][y] = 0;
					else kernel[x][y] = img.at<uchar>(i - 1 + x, j - 1 + y);
				}
			}
			int k[8], maxval = INT_MIN;
			k[0] = (kernel[0][2] + kernel[1][2] * 2 + kernel[2][2]) - (kernel[0][0] + kernel[1][0] * 2 + kernel[2][0]);
			k[1] = (kernel[0][1] + kernel[0][2] * 2 + kernel[1][2]) - (kernel[1][0] + kernel[2][0] * 2 + kernel[2][1]);
			k[2] = (kernel[0][0] + kernel[0][1] * 2 + kernel[0][2]) - (kernel[2][0] + kernel[2][1] * 2 + kernel[2][2]);
			k[3] = (kernel[0][1] + kernel[0][0] * 2 + kernel[1][0]) - (kernel[2][1] + kernel[2][2] * 2 + kernel[1][2]);
			k[4] = (kernel[0][0] + kernel[1][0] * 2 + kernel[2][0]) - (kernel[0][2] + kernel[1][2] * 2 + kernel[2][2]);
			k[5] = (kernel[1][0] + kernel[2][0] * 2 + kernel[2][1]) - (kernel[0][1] + kernel[0][2] * 2 + kernel[1][2]);
			k[6] = (kernel[2][0] + kernel[2][1] * 2 + kernel[2][2]) - (kernel[0][0] + kernel[0][1] * 2 + kernel[0][2]);
			k[7] = (kernel[1][2] + kernel[2][2] * 2 + kernel[2][1]) - (kernel[0][1] + kernel[0][0] * 2 + kernel[1][0]);
			for (int x = 0; x < 8; x++)
			{
				maxval = max(maxval, k[x]);
			}
			if (maxval > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

Mat Nevatia_Babu(Mat img, int threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel[5][5];
			for (int x = 0; x < 5; x++)
			{
				for (int y = 0; y < 5; y++)
				{
					if (i - 2 + x < 0 || j - 2 + y < 0) kernel[x][y] = 0;
					else kernel[x][y] = img.at<uchar>(i - 2 + x, j - 2 + y);
				}
			}
			int k[6], maxval = INT_MIN;
			k[0] = 100 * ((kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[0][3] + kernel[0][4] + kernel[1][0] + kernel[1][1] + kernel[1][2] + kernel[1][3] + kernel[1][4]) -
						  (kernel[3][0] + kernel[3][1] + kernel[3][2] + kernel[3][3] + kernel[3][4] + kernel[4][0] + kernel[4][1] + kernel[4][2] + kernel[4][3] + kernel[4][4]));
			k[1] = 100 * ((kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[0][3] + kernel[0][4] + kernel[1][0] + kernel[1][1] + kernel[1][2] + kernel[2][0]) -
						  (kernel[2][4] + kernel[3][2] + kernel[3][3] + kernel[3][4] + kernel[4][0] + kernel[4][1] + kernel[4][2] + kernel[4][3] + kernel[4][4])) +
				   32 * (kernel[3][0] - kernel[1][4]) + 92 * (kernel[2][1] - kernel[2][3]) + 78 * (kernel[1][3] - kernel[3][1]);
			k[2] = 100 * ((kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[1][0] + kernel[1][1] + kernel[2][0] + kernel[2][1] + kernel[3][0] + kernel[4][0]) - 
						  (kernel[0][4] + kernel[1][4] + kernel[2][3] + kernel[2][4] + kernel[3][3] + kernel[3][4] + kernel[4][2] + kernel[4][3] + kernel[4][4]));
			k[3] = 100 * ((kernel[0][3] + kernel[0][3] + kernel[1][3] + kernel[1][4] + kernel[2][3] + kernel[2][4] + kernel[3][3] + kernel[3][4] + kernel[4][3] + kernel[4][4]) -
						  (kernel[0][0] + kernel[0][1] + kernel[1][0] + kernel[1][1] + kernel[2][0] + kernel[2][1] + kernel[3][0] + kernel[3][1] + kernel[4][0] + kernel[4][1]));
			k[4] = 100 * ((kernel[0][2] + kernel[0][3] + kernel[0][4] + kernel[1][3] + kernel[1][4] + kernel[2][3] + kernel[2][4] + kernel[3][4] + kernel[4][4]) - 
						  (kernel[0][0] + kernel[1][0] + kernel[2][0] + kernel[2][1] + kernel[3][0] + kernel[3][1] + kernel[4][0] + kernel[4][1] + kernel[4][2])) + 
				   32 * (kernel[0][1] - kernel[4][3]) + 92 * (kernel[1][2] - kernel[3][2]) + 78 * (kernel[3][3] - kernel[1][1]);
			k[5] = 100 * ((kernel[0][0] + kernel[0][1] + kernel[0][2] + kernel[0][3] + kernel[0][4] + kernel[1][2] + kernel[1][3] + kernel[1][4] + kernel[2][4]) -
						  (kernel[2][0] + kernel[3][0] + kernel[3][1] + kernel[3][2] + kernel[4][0] + kernel[4][1] + kernel[4][2] + kernel[4][3] + kernel[4][4])) +
				   32 * (kernel[3][4] - kernel[1][0]) + 92 * (kernel[2][3] - kernel[2][1]) + 78 * (kernel[1][1] - kernel[3][3]);
			for (int x = 0; x < 6; x++)
			{
				maxval = max(maxval, k[x]);
			}
			if (maxval > threshold)
			{
				imgproc.at<uchar>(i, j) = 0;
			}
			else
			{
				imgproc.at<uchar>(i, j) = 255;
			}
		}
	}

	return imgproc;
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	Mat result = Mat(img.rows, img.cols, CV_8UC1);

	result = Roberts(img, 15);
	imshow("Roberts", result);
	imwrite("Roberts.jpg", result);
	result = Prewitt(img, 50);
	imshow("Prewitt", result);
	imwrite("Prewitt.jpg", result);
	result = Sobel(img, 60);
	imshow("Sobel", result);
	imwrite("Sobel.jpg", result);
	result = Frei_and_Chen(img, 30);
	imshow("Frei and Chen", result);
	imwrite("Frei and Chen.jpg", result);
	result = Kirsch(img, 175);
	imshow("Kirsch", result);
	imwrite("Kirsch.jpg", result);
	result = Robinson(img, 60);
	imshow("Robinson", result);
	imwrite("Robinson.jpg", result);
	result = Nevatia_Babu(img, 14000);
	imshow("Nevatia Babu", result);
	imwrite("Nevatia Babu.jpg", result);

	waitKey(0);
	return 0;
}