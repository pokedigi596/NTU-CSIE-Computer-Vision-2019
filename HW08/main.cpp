#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<string>
#include<vector>

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

Mat gaussian_noise(Mat img, int amplitude)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			double ran = 0;
			for (int i = 0; i < 12; i++)
			{
				ran += ((double)rand() / RAND_MAX);
			}
			ran = ran - 6.0;
			int pixel = img.at<uchar>(i, j) + (int)(amplitude * ran);
			imgproc.at<uchar>(i, j) = pixel;
		}
	}
	return imgproc;
}

Mat SAP_noise(Mat img, double threshold)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			double ran = ((double)rand() / RAND_MAX);
			if (ran < threshold) imgproc.at<uchar>(i, j) = 0;
			else if (ran > 1 - threshold) imgproc.at<uchar>(i, j) = 255;
			else imgproc.at<uchar>(i, j) = img.at<uchar>(i, j);
		}
	}
	return imgproc;
}

Mat box_filter(Mat img, int size)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int kernel_size = 0;
			int pixel = 0;
			for (int x = 0; x < size; x++)
			{
				for (int y = 0; y < size; y++)
				{
					if (i + x - size / 2 < 0 || i + x - size / 2 >= img.rows || j + y - size / 2 < 0 || j + y - size / 2 >= img.cols)
						continue;
					kernel_size++;
					pixel += img.at<uchar>(i + x - size / 2, j + y - size / 2);
				}
			}
			imgproc.at<uchar>(i, j) = pixel / kernel_size;
		}
	}
	return imgproc;
}

Mat median_filter(Mat img, int size)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			std::vector<uchar> pixel_arr;
			int kernel_size = 0;
			for (int x = 0; x < size; x++)
			{
				for (int y = 0; y < size; y++)
				{
					if (i + x - size / 2 < 0 || i + x - size / 2 >= img.rows || j + y - size / 2 < 0 || j + y - size / 2 >= img.cols)
						continue;
					kernel_size++;
					pixel_arr.push_back(img.at<uchar>(i + x - size / 2, j + y - size / 2));
				}
			}
			sort(pixel_arr.begin(), pixel_arr.end());
			imgproc.at<uchar>(i, j) = (kernel_size % 2 == 0) ? (pixel_arr.at(kernel_size / 2) + pixel_arr.at(kernel_size / 2 + 1)) / 2 : pixel_arr.at(kernel_size / 2 + 1);
		}
	}
	return imgproc;
}

Mat open_close(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);
	// Opening
	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(img, imgproc, i, j, 1);
		}
	}
	for (int i = 2; i < imgproc.rows - 2; i++)
	{
		for (int j = 2; j < imgproc.cols - 2; j++)
		{
			kernal(imgproc, imgproc1, i, j, 0);
		}
	}
	// Closing
	for (int i = 2; i < imgproc1.rows - 2; i++)
	{
		for (int j = 2; j < imgproc1.cols - 2; j++)
		{
			kernal(imgproc1, imgproc, i, j, 0);
		}
	}
	for (int i = 2; i < imgproc.rows - 2; i++)
	{
		for (int j = 2; j < imgproc.cols - 2; j++)
		{
			kernal(imgproc, imgproc1, i, j, 1);
		}
	}
	return imgproc1;
}

Mat close_open(Mat img)
{
	Mat imgproc = Mat(img.rows, img.cols, CV_8UC1);
	Mat imgproc1 = Mat(img.rows, img.cols, CV_8UC1);
	// Closing
	for (int i = 2; i < img.rows - 2; i++)
	{
		for (int j = 2; j < img.cols - 2; j++)
		{
			kernal(img, imgproc, i, j, 0);
		}
	}
	for (int i = 2; i < imgproc.rows - 2; i++)
	{
		for (int j = 2; j < imgproc.cols - 2; j++)
		{
			kernal(imgproc, imgproc1, i, j, 1);
		}
	}
	// Opening
	for (int i = 2; i < imgproc1.rows - 2; i++)
	{
		for (int j = 2; j < imgproc1.cols - 2; j++)
		{
			kernal(imgproc1, imgproc, i, j, 1);
		}
	}
	for (int i = 2; i < imgproc.rows - 2; i++)
	{
		for (int j = 2; j < imgproc.cols - 2; j++)
		{
			kernal(imgproc, imgproc1, i, j, 0);
		}
	}
	return imgproc1;
}

double SNR(Mat img, Mat noise)
{
	double result = 0.0;
	double avg_img = 0.0;
	double avg_noise = 0.0;
	double VS = 0.0;
	double VN = 0.0;
	int size = img.rows * img.cols;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			avg_img += (double)img.at<uchar>(i, j) / 255;
			avg_noise += ((double)noise.at<uchar>(i, j) / 255 - (double)img.at <uchar>(i, j) / 255);
		}
	}
	avg_img /= size;
	avg_noise /= size;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			VS += pow((double)img.at<uchar>(i, j) / 255 - avg_img, 2);
			VN += pow((double)noise.at<uchar>(i, j) / 255 - (double)img.at <uchar>(i, j) / 255 - avg_noise, 2);
		}
	}
	VS /= size;
	VN /= size;
	result = log10(pow(VS, 0.5) / pow(VN, 0.5)) * 20;
	return result;
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	Mat gaussian = Mat(img.rows, img.cols, CV_8UC1);
	Mat salt_and_pepper = Mat(img.rows, img.cols, CV_8UC1);
	Mat tmp = Mat(img.rows, img.cols, CV_8UC1);

	// Gaussian 10
	gaussian = gaussian_noise(img, 10);
	printf("%f\n", SNR(img, gaussian));
	imshow("Gaussian Noise10", gaussian);
	imwrite("Gaussian Noise10.jpg", gaussian);
	tmp = box_filter(gaussian, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise10 Box3", tmp);
	imwrite("Gaussian Noise10 Box3.jpg", tmp);
	tmp = box_filter(gaussian, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise10 Box5", tmp);
	imwrite("Gaussian Noise10 Box5.jpg", tmp);
	tmp = median_filter(gaussian, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise10 Median3", tmp);
	imwrite("Gaussian Noise10 Median3.jpg", tmp);
	tmp = median_filter(gaussian, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise10 Median5", tmp);
	imwrite("Gaussian Noise10 Median5.jpg", tmp);
	tmp = open_close(gaussian);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise10 OpenClose", tmp);
	imwrite("Gaussian Noise10 OpenClose.jpg", tmp);
	tmp = close_open(gaussian);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise10 CloseOpen", tmp);
	imwrite("Gaussian Noise10 CloseOpen.jpg", tmp);
	// Gaussian 30
	gaussian = gaussian_noise(img, 30);
	printf("%f\n", SNR(img, gaussian));
	imshow("Gaussian Noise30", gaussian);
	imwrite("Gaussian Noise30.jpg", gaussian);
	tmp = box_filter(gaussian, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise30 Box3", tmp);
	imwrite("Gaussian Noise30 Box3.jpg", tmp);
	tmp = box_filter(gaussian, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise30 Box5", tmp);
	imwrite("Gaussian Noise30 Box5.jpg", tmp);
	tmp = median_filter(gaussian, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise30 Median3", tmp);
	imwrite("Gaussian Noise30 Median3.jpg", tmp);
	tmp = median_filter(gaussian, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise30 Median5", tmp);
	imwrite("Gaussian Noise30 Median5.jpg", tmp);
	tmp = open_close(gaussian);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise30 OpenClose", tmp);
	imwrite("Gaussian Noise30 OpenClose.jpg", tmp);
	tmp = close_open(gaussian);
	printf("%f\n", SNR(img, tmp));
	imshow("Gaussian Noise30 CloseOpen", tmp);
	imwrite("Gaussian Noise30 CloseOpen.jpg", tmp);
	// Salt and Pepper 0.05
	salt_and_pepper = SAP_noise(img, 0.05);
	printf("%f\n", SNR(img, salt_and_pepper));
	imshow("SAP0.05", salt_and_pepper);
	imwrite("SAP0.05.jpg", salt_and_pepper);
	tmp = box_filter(salt_and_pepper, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.05 Box3", tmp);
	imwrite("SAP0.05 Box3.jpg", tmp);
	tmp = box_filter(salt_and_pepper, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.05 Box5", tmp);
	imwrite("SAP0.05 Box5.jpg", tmp);
	tmp = median_filter(salt_and_pepper, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.05 Median3", tmp);
	imwrite("SAP0.05 Median3.jpg", tmp);
	tmp = median_filter(salt_and_pepper, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.05 Median5", tmp);
	imwrite("SAP0.05 Median5.jpg", tmp);
	tmp = open_close(salt_and_pepper);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.05 OpenClose", tmp);
	imwrite("SAP0.05 OpenClose.jpg", tmp);
	tmp = close_open(salt_and_pepper);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.05 CloseOpen", tmp);
	imwrite("SAP0.05 CloseOpen.jpg", tmp);
	// Salt and Pepper 0.1
	salt_and_pepper = SAP_noise(img, 0.1);
	printf("%f\n", SNR(img, salt_and_pepper));
	imshow("SAP0.1", salt_and_pepper);
	imwrite("SAP0.1.jpg", salt_and_pepper);
	tmp = box_filter(salt_and_pepper, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.1 Box3", tmp);
	imwrite("SAP0.1 Box3.jpg", tmp);
	tmp = box_filter(salt_and_pepper, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.1 Box5", tmp);
	imwrite("SAP0.1 Box5.jpg", tmp);
	tmp = median_filter(salt_and_pepper, 3);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.1 Median3", tmp);
	imwrite("SAP0.1 Median3.jpg", tmp);
	tmp = median_filter(salt_and_pepper, 5);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.1 Median5", tmp);
	imwrite("SAP0.1 Median5.jpg", tmp);
	tmp = open_close(salt_and_pepper);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.1 OpenClose", tmp);
	imwrite("SAP0.1 OpenClose.jpg", tmp);
	tmp = close_open(salt_and_pepper);
	printf("%f\n", SNR(img, tmp));
	imshow("SAP0.1 CloseOpen", tmp);
	imwrite("SAP0.1 CloseOpen.jpg", tmp);

	waitKey(0);
	return 0;
}