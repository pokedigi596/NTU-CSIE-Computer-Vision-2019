#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;

int hist[256];

void pixel_divided_by_3(Mat img)
{
	// pixel divided
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			img.at<Vec3b>(i, j)[0] /= 3;
			img.at<Vec3b>(i, j)[1] /= 3;
			img.at<Vec3b>(i, j)[2] /= 3;
		}
	// show image
	imshow("pixel divided by 3", img);

	// write image 
	imwrite("pixel divided by 3 Lena.jpg", img);
}

void histogram(Mat img, std::string filename)
{
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
	file.open(filename, std::ios::out);
	file << "histogram";
	file << '\n';
	for (int i = 0; i < 256; i++)
	{
		file << hist[i];
		file << '\n';
	}
}

void histogram_equalization(Mat img)
{
	// histogram equalization
	for (int i = 255; i >= 0; i--)
	{
		for (int j = 0; j < i; j++)
		{
			hist[i] += hist[j];
		}
		hist[i] = 255 * hist[i] / (img.rows * img.cols);
	}
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			img.at<Vec3b>(i, j)[0] = hist[img.at<Vec3b>(i, j)[0]];
			img.at<Vec3b>(i, j)[1] = hist[img.at<Vec3b>(i, j)[1]];
			img.at<Vec3b>(i, j)[2] = hist[img.at<Vec3b>(i, j)[2]];
		}
	// show image
	imshow("histogram equalization", img);

	// write image 
	imwrite("histogram equalization Lena.jpg", img);
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp");
	histogram(img, "lena.csv");
	pixel_divided_by_3(img);
	histogram(img, "pixel_divided_by_3 lena.csv");
	histogram_equalization(img);
	histogram(img, "histogram equalization lena .csv");
	waitKey(0);
	return 0;
}