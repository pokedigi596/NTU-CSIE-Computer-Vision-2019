#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

int* binary_image;
int label = 1;

void binarize(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC3);
	// binarize
	for (int i = 0; i < imgproc.rows; i++)
		for (int j = 0; j < imgproc.cols; j++)
		{
			if (img.at<Vec3b>(i, j)[0] >= 128)
			{
				imgproc.at<Vec3b>(i, j)[0] = 255;
				imgproc.at<Vec3b>(i, j)[1] = 255;
				imgproc.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				imgproc.at<Vec3b>(i, j)[0] = 0;
				imgproc.at<Vec3b>(i, j)[1] = 0;
				imgproc.at<Vec3b>(i, j)[2] = 0;
			}
		}
	// show image
	imshow("binarize", imgproc);

	// write image 
	imwrite("binarize Lena.jpg", imgproc);
}

void histogram(Mat img)
{
	int hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = 0;
	}
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			hist[img.at<Vec3b>(i, j)[0]]++;
		}
	std::fstream file;
	file.open("histogram.csv", std::ios::out);
	file << "histogram";
	file << '\n';
	for (int i = 0; i < 256; i++)
	{
		file << hist[i];
		file << '\n';
	}
}

bool neighbor_comparison(int x, int y, int width, int height)
{
	int n1 = (x - 1) < 0 ? -1 : binary_image[(x - 1) * width + y];
	int n2 = (y - 1) < 0 ? -1 : binary_image[x * width + (y - 1)];
	int n3 = (y + 1) >= width ? -1 : binary_image[x * width + (y + 1)];
	int n4 = (x + 1) >= height ? -1 : binary_image[(x + 1) * width + y];
	int c1, c2, final_value;

	if (n1 < 0 && n2 < 0 && n3 < 0 && n4 < 0) return false;
	if (min(n1, n4) <= 0) c1 = max(n1, n4);
	else c1 = min(n1, n4);
	if (min(n2, n3) <= 0) c2 = max(n2, n3);
	else c2 = min(n2, n3);
	if (min(c1, c2) <= 0) final_value = max(c1, c2);
	else final_value = min(c1, c2);
	if (binary_image[x * width + y] > final_value)
	{
		binary_image[x * width + y] = final_value;
		return true;
	}
	else if (binary_image[x * width + y] == 0)
	{
		binary_image[x * width + y] = final_value > 0 ? final_value : label++;
		return true;
	}
	else return false;
}

void CCA(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC3);
	binary_image = new int[imgproc.rows * imgproc.cols];
	// binarize
	for (int i = 0; i < imgproc.rows; i++)
		for (int j = 0; j < imgproc.cols; j++)
		{
			if (img.at<Vec3b>(i, j)[0] >= 128)
			{
				imgproc.at<Vec3b>(i, j)[0] = 255;
				imgproc.at<Vec3b>(i, j)[1] = 255;
				imgproc.at<Vec3b>(i, j)[2] = 255;
				binary_image[i * imgproc.rows + j] = 0;
			}
			else
			{
				imgproc.at<Vec3b>(i, j)[0] = 0;
				imgproc.at<Vec3b>(i, j)[1] = 0;
				imgproc.at<Vec3b>(i, j)[2] = 0;
				binary_image[i * imgproc.rows + j] = -1;
			}
		}
	// Connected Components Algorithms

	// top-down
	for (int i = 0; i < imgproc.rows; i++)
		for (int j = 0; j < imgproc.cols; j++)
		{
			if (binary_image[i * imgproc.rows + j] == 0)
			{
				neighbor_comparison(i, j, imgproc.rows, imgproc.cols);
			}
		}
	// bottom-up
	bool change = true;
	while (change)
	{
		change = false;
		for (int i = imgproc.rows - 1; i >= 0; i--)
			for (int j = imgproc.cols - 1; j >= 0; j--)
			{
				if (binary_image[i * imgproc.rows + j] > 0)
				{
					change = change | neighbor_comparison(i, j, imgproc.rows, imgproc.cols);
				}
			}
	}
	// draw bounding box
	for (int k = 1; k <= label - 1; k++)
	{
		int count = 0;
		int row = 0, col = 0;
		int row_min = imgproc.rows - 1;
		int row_max = 0;
		int col_min = imgproc.cols - 1;
		int col_max = 0;
		for (int i = 0; i < imgproc.rows; i++)
			for (int j = 0; j < imgproc.cols; j++)
			{
				if (binary_image[i * imgproc.rows + j] == k)
				{
					row += i;
					col += j;
					if (row_min > i) row_min = i;
					if (row_max < i) row_max = i;
					if (col_min > j) col_min = j;
					if (col_max < j) col_max = j;
					count++;
				}
			}
		if (count > 500)
		{
			rectangle(imgproc, Point(col_min, row_min), Point(col_max, row_max), Scalar(255, 0, 0), 2);
			line(imgproc, Point(col / count - 7, row / count), Point(col / count + 7, row / count), Scalar(0, 0, 255), 2);
			line(imgproc, Point(col / count, row / count - 7), Point(col / count, row / count + 7), Scalar(0, 0, 255), 2);
		}
	}
	// show image
	imshow("Connected Components Algorithm", imgproc);

	// write image 
	imwrite("Connected Components Algorithm Lena.jpg", imgproc);
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp");

	binarize(img);
	histogram(img);
	CCA(img);

	waitKey(0);
	return 0;
}