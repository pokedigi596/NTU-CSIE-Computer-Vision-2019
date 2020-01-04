#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

std::string matrix[64][64];

Mat binarize_downsampling(Mat img)
{
	Mat tmp = Mat(img.rows, img.cols, CV_8UC1);
	Mat imgproc = Mat(img.rows / 8, img.cols / 8, CV_8UC1);
	// binarize
	for (int i = 0; i < tmp.rows; i++)
	{
		for (int j = 0; j < tmp.cols; j++)
		{
			tmp.at<uchar>(i, j) = img.at<uchar>(i, j) >= 128 ? 255 : 0;
		}
	}
	for (int i = 0; i < imgproc.rows; i++)
	{
		for (int j = 0; j < imgproc.cols; j++)
		{
			imgproc.at<uchar>(i, j) = tmp.at<uchar>(i * 8, j * 8);
		}
	}

	return imgproc;
}

void Yokoi(Mat img)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 255)
			{
				unsigned char kernal[3][3];
				int labelQ = 0;
				int labelR = 0;
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						kernal[x][y] = (i - 1 + x < 0 || j - 1 + y < 0 || i - 1 + x >= img.rows || j - 1 + y >= img.cols) ? 0 : img.at<uchar>(i - 1 + x, j - 1 + y);
					}
				}
				if (kernal[1][1] == kernal[1][2] && (kernal[0][2] != kernal[1][1] || kernal[0][1] != kernal[1][1])) labelQ++;
				else if (kernal[1][1] == kernal[1][2] && (kernal[0][2] == kernal[1][1] && kernal[0][1] == kernal[1][1])) labelR++;
				if (kernal[1][1] == kernal[0][1] && (kernal[0][0] != kernal[1][1] || kernal[1][0] != kernal[1][1])) labelQ++;
				else if (kernal[1][1] == kernal[0][1] && (kernal[0][0] == kernal[1][1] && kernal[1][0] == kernal[1][1])) labelR++;
				if (kernal[1][1] == kernal[1][0] && (kernal[2][0] != kernal[1][1] || kernal[2][1] != kernal[1][1])) labelQ++;
				else if (kernal[1][1] == kernal[1][0] && (kernal[2][0] == kernal[1][1] && kernal[2][1] == kernal[1][1])) labelR++;
				if (kernal[1][1] == kernal[2][1] && (kernal[2][2] != kernal[1][1] || kernal[1][2] != kernal[1][1])) labelQ++;
				else if (kernal[1][1] == kernal[2][1] && (kernal[2][2] == kernal[1][1] && kernal[1][2] == kernal[1][1])) labelR++;
				matrix[i][j] = (labelR == 4) ? '5' : labelQ + '0';
			}
			else matrix[i][j] = '0';
		}
	}
}

void draw_Yokoi()
{
	Mat Yokoi_img = Mat(512, 512, CV_8UC1);
	for (int i = 0; i < 512; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			Yokoi_img.at<uchar>(i, j) = 255;
		}
	}
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			if (matrix[i][j] != "0")
			{
				putText(Yokoi_img, matrix[i][j], Point(7.5 * (j + 1), 7.5 * (i + 2)), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, Scalar(0, 0, 0));
			}
		}
	}

	imshow("Yokoi", Yokoi_img);

	imwrite("Yokoi.jpg", Yokoi_img);
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	img = binarize_downsampling(img);
	Yokoi(img);
	draw_Yokoi();
	
	waitKey(0);
	return 0;
}