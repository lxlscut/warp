//如果要生成python库文件，需要在属性->常规->配置类型中将其改为dll，生成后在release文件夹下将文件名改为pyd即可

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<Eigen/Dense>
#include<Eigen/SVD>
#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
using namespace std;
using namespace cv;
using namespace Eigen;


py::array_t<double> add_array_3d(py::array_t<double>& input1, py::array_t<double>& input2) {
	//获取input1,input2的信息
	auto r1 = input1.unchecked<3>();
	auto r2 = input2.unchecked<3>();

	py::array_t<double> out = py::array_t<double>(input1.size());
	out.resize({ input1.shape()[0],input1.shape()[1],input1.shape()[2] });

	auto r3 = out.mutable_unchecked<3>();

	for (int i = 0; i < input1.shape()[0]; i++) {
		for (int j = 0; j < input1.shape()[1]; j++) {
			for (int k = 0; k < input1.shape()[2]; k++) {
				double value1 = r1(i, j, k);
				double value2 = r2(i, j, k);
				r3(i, j, k) = value1 + value2;
			}
		}
	}
	return out;
}


py::array_t<double> add_array_4d(py::array_t<double>& input1, py::array_t<double>& input2) {
	//获取input1,input2的信息
	auto r1 = input1.unchecked<4>();
	auto r2 = input2.unchecked<4>();

	py::array_t<double> out = py::array_t<double>(input1.size());
	out.resize({ input1.shape()[0],input1.shape()[1],input1.shape()[2],input1.shape()[3]});

	auto r3 = out.mutable_unchecked<4>();

	for (int i = 0; i < input1.shape()[0]; i++) {
		for (int j = 0; j < input1.shape()[1]; j++) {
			for (int k = 0; k < input1.shape()[2]; k++) {
				for (int l = 0; l < input1.shape()[3]; l++) {
					double value1 = r1(i, j, k,l);
					double value2 = r2(i, j, k,l);
					r3(i, j, k,l) = value1 + value2;
				}
			}
		}
	}
	return out;
}

py::array_t<uint> warp_local(py::array_t<double>& src_img,py::array_t<double> &dst_img, 
	py::array_t<float>& local_H, int& height, int& width, float &offset_x, float &offset_y,py::array_t<float> &point) {
	//源图
	auto r1 = src_img.unchecked<3>();
	//目标图
	auto r2 = dst_img.unchecked<3>();
	//局部矩阵
	auto r3 = local_H.unchecked<4>();
	//长宽锚点
	auto r4 = point.unchecked<2>();

	py::array_t<uint> result = py::array_t<uint>(height*width*3);

	result.resize({ height,width,3 });

	auto r5 = result.mutable_unchecked<3>();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int m = 0;
			int n = 0;
			while (i>=r4(1,m))
			{
				m++;
			}
			while (j>=r4(0,n))
			{
				n++;
			}
			MatrixXf current_h(3,3);
			current_h << r3(m-1, n-1, 0, 0), r3(m-1, n-1, 0, 1), r3(m-1, n-1, 0, 2), r3(m-1, n-1, 1, 0),
				r3(m-1, n-1, 1, 1), r3(m-1, n-1, 1, 2), r3(m-1, n-1, 2, 0), r3(m-1, n-1, 2, 1), r3(m-1, n-1, 2, 2);
			MatrixXf current_h_new = current_h.inverse();
			//cout << current_h_new << endl;
			MatrixXf current_point(3, 1);
			current_point <<  j-offset_x, i - offset_y, 1;
			//cout << current_point << endl;
			MatrixXf a = current_h_new*current_point;
			//cout << "origin a is" << a << endl;
			//cout << a << "a(0,0) is :" << a(0, 0) << "a(0,1) is :" << a(0, 1) << "a(0,2) is :" << a(0, 2) << endl;

			//x坐标
			int col = a(0, 0) / a(2, 0);
			//y坐标
			int row = a(1, 0) / a(2, 0);
			//cout << "row is" << row << "col is " << col << endl;
			if ( (row >= 0 && row < src_img.shape()[0]) && (col>= 0&& col < src_img.shape()[1])) {
				for (int channel = 0; channel < 3; channel++) {
					uint value = r1(row, col, channel);
					r5(i, j, channel) = value;
				}
				//cout << "i is :" << i << "j is :" << j << "row is :" << row << "col is" << col << endl;
			}
		}
	}

	return result;
}


PYBIND11_MODULE(NUMPY, m) {
	m.doc() = "3d and 4d canculate demo";
	m.def("add_arrays_3d", &add_array_3d);
	m.def("add_arrays_4d", &add_array_4d);
	m.def("warp_local", &warp_local);
}
