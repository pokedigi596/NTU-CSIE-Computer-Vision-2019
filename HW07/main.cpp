#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<string>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

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

int H(int b, int c, int d, int e)
{
	if (b == c && (d != b || e != b)) return 1;
	if (b == c && (d == b && e == b)) return 2;
	return 0;
}

int F(int a1, int a2, int a3, int a4)
{
	if (a1 == a2 && a2 == a3 && a3 == a4 && a4 == 2) return 5;
	int n = 0;
	if (a1 == 1) n++;
	if (a2 == 1) n++;
	if (a3 == 1) n++;
	if (a4 == 1) n++;
	return n;
}

void Yokoi(Mat img, Mat matrix)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 255)
			{
				unsigned char kernal[3][3];
				int a[4];
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						kernal[x][y] = (i - 1 + x < 0 || j - 1 + y < 0 || i - 1 + x >= img.rows || j - 1 + y >= img.cols) ? 0 : img.at<uchar>(i - 1 + x, j - 1 + y);
					}
				}
				a[0] = H(kernal[1][1], kernal[1][2], kernal[0][2], kernal[0][1]);
				a[1] = H(kernal[1][1], kernal[0][1], kernal[0][0], kernal[1][0]);
				a[2] = H(kernal[1][1], kernal[1][0], kernal[2][0], kernal[2][1]);
				a[3] = H(kernal[1][1], kernal[2][1], kernal[2][2], kernal[1][2]);
				matrix.at<uchar>(i, j) = F(a[0], a[1], a[2], a[3]);
			}
			else matrix.at<uchar>(i, j) = 6;
		}
	}
}

void pair_relationship(Mat matrix)
{
	for (int i = 0; i < matrix.rows; i++)
	{
		for (int j = 0; j < matrix.cols; j++)
		{
			if (matrix.at<uchar>(i, j) != 6)
			{
				int a[4];
				int n = 0;
				a[0] = (i - 1) < 0 ? 0 : matrix.at<uchar>(i - 1, j);
				a[1] = (i + 1) > matrix.rows - 1 ? 0 : matrix.at<uchar>(i + 1, j);
				a[2] = (j - 1) < 0 ? 0 : matrix.at<uchar>(i, j - 1);
				a[3] = (j + 1) > matrix.cols - 1 ? 0 : matrix.at<uchar>(i, j + 1);
				for (int k = 0; k < 4; k++)
				{
					if (a[k] > 0 && a[k] != 6) n++;
				}
				if (n > matrix.at<uchar>(i, j) && matrix.at<uchar>(i, j) == 1)
				{
					matrix.at<uchar>(i, j) = 'p';
				}
				else
				{
					matrix.at <uchar>(i, j) = 'q';
				}
			}
		}
	}
}

bool thinned_image(Mat matrix, Mat img)
{
	bool flag = false;
	for (int i = 0; i < matrix.rows; i++)
	{
		for (int j = 0; j < matrix.cols; j++)
		{
			if (matrix.at<uchar>(i, j) == 'p')
			{
				unsigned char kernal[3][3];
				int a[4];
				for (int x = 0; x < 3; x++)
				{
					for (int y = 0; y < 3; y++)
					{
						kernal[x][y] = (i - 1 + x < 0 || j - 1 + y < 0 || i - 1 + x >= matrix.rows || j - 1 + y >= matrix.cols) ? 6 : matrix.at<uchar>(i - 1 + x, j - 1 + y);
						kernal[x][y] = (kernal[x][y] != 6 && kernal[x][y] != 0) ? 1 : 0;
					}
				}
				a[0] = H(kernal[1][1], kernal[1][2], kernal[0][2], kernal[0][1]);
				a[1] = H(kernal[1][1], kernal[0][1], kernal[0][0], kernal[1][0]);
				a[2] = H(kernal[1][1], kernal[1][0], kernal[2][0], kernal[2][1]);
				a[3] = H(kernal[1][1], kernal[2][1], kernal[2][2], kernal[1][2]);
				if (F(a[0], a[1], a[2], a[3]) == 1)
				{
					matrix.at<uchar>(i, j) = 0;
				}
				else
				{
					matrix.at<uchar>(i, j) = 255;
				}
				if (matrix.at<uchar>(i, j) != img.at<uchar>(i, j))
				{
					flag = true;
				}
			}
			else if (matrix.at<uchar>(i, j) == 'q')
			{
				matrix.at<uchar>(i, j) = 255;
			}
			else matrix.at<uchar>(i, j) = 0;
		}
	}
	return flag;
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);
	Mat Yokoi_matrix = Mat(img.rows / 8, img.cols / 8, CV_8UC1);
	Mat matrix = Mat(img.rows / 8, img.cols / 8, CV_8UC1);
	bool flag = true;

	img = binarize_downsampling(img);
	while (flag)
	{
		Yokoi(img, Yokoi_matrix);
		pair_relationship(Yokoi_matrix);
		flag = thinned_image(Yokoi_matrix, img);
		Yokoi_matrix.copyTo(img);
	}
	//resize(img, matrix, Size(512, 512));
	imshow("Thin Image", img);
	imwrite("Thin Image.jpg", img);
	waitKey(0);
	return 0;
}