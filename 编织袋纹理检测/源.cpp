#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/photo/photo.hpp>

using namespace cv;
using namespace std;

Mat WDT(const Mat &_src, const string _wname, const int _level);
void wavelet(const string _wname, Mat &_lowFilter, Mat &_highFilter);
Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
Mat GetRedComponet(Mat srcImg);
Mat Inpainting(Mat oriImg, Mat maskImg);
///  小波变换


Mat WDT(const Mat &_src, const string _wname, const int _level)
{
	///int reValue = THID_ERR_NONE;
	Mat src = Mat_<float>(_src);///Mat类型数据访问，CV_32F
	Mat dst = Mat::zeros(src.rows, src.cols, src.type());///返回指定的大小和类型的零数组
	int N = src.rows;
	int D = src.cols;

	/// 高通低通滤波器
	Mat lowFilter;
	Mat highFilter;
	wavelet(_wname, lowFilter, highFilter);

	/// 小波变换
	int t = 1;
	int row = N;
	int col = D;

	while (t <= _level)
	{
		///先进行行小波变换
		for (int i = 0; i<row; i++)
		{
			/// 取出src中要处理的数据的一行
			Mat oneRow = Mat::zeros(1, col, src.type());
			for (int j = 0; j<col; j++)
			{
				oneRow.at<float>(0, j) = src.at<float>(i, j);
			}
			oneRow = waveletDecompose(oneRow, lowFilter, highFilter);
			/// 将src这一行置为oneRow中的数据
			for (int j = 0; j<col; j++)
			{
				dst.at<float>(i, j) = oneRow.at<float>(0, j);
			}
		}
		/// 小波列变换
		for (int j = 0; j<col; j++)
		{
			/// 取出src数据的一行输入
			Mat oneCol = Mat::zeros(row, 1, src.type());
			for (int i = 0; i<row; i++)
			{
				oneCol.at<float>(i, 0) = dst.at<float>(i, j);
			}
			oneCol = (waveletDecompose(oneCol.t(), lowFilter, highFilter)).t();

			for (int i = 0; i<row; i++)
			{
				dst.at<float>(i, j) = oneCol.at<float>(i, 0);
			}
		}

		/// 更新
		row /= 2;
		col /= 2;
		t++;
		src = dst;
	}

	return dst;
}


///  小波逆变换
Mat IWDT(const Mat &_src, const string _wname, const int _level)
{
	//int reValue = THID_ERR_NONE;
	Mat src = Mat_<float>(_src);
	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	int N = src.rows;
	int D = src.cols;

	/// 高通低通滤波器
	Mat lowFilter;
	Mat highFilter;
	wavelet(_wname, lowFilter, highFilter);

	/// 小波变换
	int t = 1;
	int row = N / pow(2., _level - 1);
	int col = D / pow(2., _level - 1);

	while (row <= N && col <= D)
	{
		/// 小波列逆变换
		for (int j = 0; j<col; j++)
		{
			/// 取出src数据的一行输入
			Mat oneCol = Mat::zeros(row, 1, src.type());
			for (int i = 0; i<row; i++)
			{
				oneCol.at<float>(i, 0) = src.at<float>(i, j);
			}
			oneCol = waveletReconstruct(oneCol.t(), lowFilter, highFilter);


			for (int i = 0; i<row; i++)
			{
				dst.at<float>(i, j) = oneCol.at<float>(i, 0);
			}
		}

		///行小波逆变换
		for (int i = 0; i<row; i++)
		{
			/// 取出src中要处理的数据的一行
			Mat oneRow = Mat::zeros(1, col, src.type());
			for (int j = 0; j<col; j++)
			{
				oneRow.at<float>(0, j) = dst.at<float>(i, j);
			}
			oneRow = waveletReconstruct(oneRow, lowFilter, highFilter);
			/// 将src这一行置为oneRow中的数据
			for (int j = 0; j<col; j++)
			{
				dst.at<float>(i, j) = oneRow.at<float>(0, j);
			}
		}
		row *= 2;
		col *= 2;
		src = dst;
	}

	return dst;
}

/// 调用函数

/// 生成不同类型的小波，现在只有haar，sym2
void wavelet(const string _wname, Mat &_lowFilter, Mat &_highFilter)
{
	if (_wname == "sym2")
	{
		int N = 4;
		float h[] = { -0.495, 0.86, -0.224, -0.129 };
		float l[] = { -0.129, 0.224, 0.837, 0.483 };

		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		for (int i = 0; i<N; i++)
		{
			_lowFilter.at<float>(0, i) = l[i];
			_highFilter.at<float>(0, i) = h[i];
		}
	}
}


/// 小波分解
Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{

	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	Mat &src = Mat_<float>(_src);

	int D = src.cols;

	Mat &lowFilter = Mat_<float>(_lowFilter);
	Mat &highFilter = Mat_<float>(_highFilter);

	/// 频域滤波，或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());

	filter2D(src, dst1, -1, lowFilter);
	filter2D(src, dst2, -1, highFilter);

	/// 下采样
	Mat downDst1 = Mat::zeros(1, D / 2, src.type());
	Mat downDst2 = Mat::zeros(1, D / 2, src.type());

	resize(dst1, downDst1, downDst1.size());
	resize(dst2, downDst2, downDst2.size());

	/// 数据拼接
	for (int i = 0; i<D / 2; i++)
	{
		src.at<float>(0, i) = downDst1.at<float>(0, i);
		src.at<float>(0, i + D / 2) = downDst2.at<float>(0, i);
	}

	return src;
}


/// 小波重建
Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{
	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	Mat &src = Mat_<float>(_src);

	int D = src.cols;

	Mat &lowFilter = Mat_<float>(_lowFilter);
	Mat &highFilter = Mat_<float>(_highFilter);

	/// 插值;
	Mat Up1 = Mat::zeros(1, D, src.type());
	Mat Up2 = Mat::zeros(1, D, src.type());

	/// 插值为0
	//for ( int i=0, cnt=1; i<D/2; i++,cnt+=2 )
	//{
	//    Up1.at<float>( 0, cnt ) = src.at<float>( 0, i );     ///< 前一半
	//    Up2.at<float>( 0, cnt ) = src.at<float>( 0, i+D/2 ); ///< 后一半
	//}

	/// 线性插值
	Mat roi1(src, Rect(0, 0, D / 2, 1));
	Mat roi2(src, Rect(D / 2, 0, D / 2, 1));
	resize(roi1, Up1, Up1.size(), 0, 0, INTER_CUBIC);
	resize(roi2, Up2, Up2.size(), 0, 0, INTER_CUBIC);

	/// 前一半低通，后一半高通
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());
	filter2D(Up1, dst1, -1, lowFilter);
	filter2D(Up2, dst2, -1, highFilter);

	/// 结果相加
	dst1 = dst1 + dst2;

	return dst1;

}


///该方法可能产生误检点，但在可容忍的错范围内
Mat GetRedComponet(Mat srcImg)
{
	///如果直接对srcImg处理会改变main()函数中的实参
	Mat dstImg = srcImg.clone();
	MatIterator_<Vec3b> it = dstImg.begin<Vec3b>();///迭代器遍历像素
	Mat_<Vec3b>::iterator itend = dstImg.end<Vec3b>();///与上一行作用一样
	for (; it != itend; it++)
	{
		if ((*it)[0] < 50 && (*it)[1] < 50 && (*it)[3] < 50)///对红色分量做阈值处理
		//{
		//	(*it)[0] = 0;
		//	(*it)[1] = 0;
		//	//(*it)[2] = 255;//红色分量保持不变
		//}
		//else
		{
			(*it)[0] = 200;
			(*it)[1] = 200;
			(*it)[2] = 200;
		}
	}
	return dstImg;
}						

Mat Inpainting(Mat oriImg, Mat maskImg)
{
	Mat grayMaskImg;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(maskImg, maskImg, element);///膨胀后结果作为修复掩膜
									  ///将彩色图转换为单通道灰度图，最后一个参数为通道数
	cvtColor(maskImg, grayMaskImg, CV_BGR2GRAY, 1);
	///修复图像的掩膜必须为8位单通道图像
	Mat inpaintedImage;
	inpaint(oriImg, grayMaskImg, inpaintedImage, 1, INPAINT_TELEA);///图像修复
	waitKey(0);
	return inpaintedImage;
}


void main()
{
	Mat I;
	I = imread("纹理2.jpg");
	int height = I.rows;
	int width = I.cols;
	namedWindow("wddt", 0);
	//namedWindow("原始图像", 0);
	//imshow("原始图像", I);
	Mat img_wdt;
	Mat I_gray;
	Mat imgComponet = GetRedComponet(I);
	Mat img_Inpainting;
	img_Inpainting = Inpainting(I, imgComponet);
	//imshow("wddt",img_Inpainting);
	cvtColor(img_Inpainting, I_gray, CV_RGB2GRAY);///转换颜色空间
	img_wdt = WDT(img_Inpainting, "sym2", 2);
	imshow("wddt", img_wdt);
	//imwrite("2.jpg", img_wdt);
	waitKey(0);
}


