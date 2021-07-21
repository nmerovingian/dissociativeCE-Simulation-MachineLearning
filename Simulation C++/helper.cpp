#include<iostream>
#include <windows.h>
#include <assert.h>
#include <sstream>
#include <vector>
#include <thread>
#include <fstream>
#include <ctime>
#include <cassert>
#include <numeric>
#include <algorithm>
#include<Windows.h>
void Print2DArray(double** array, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << array[i][j] << "\t";
		}
		std::cout << "\n\n";
	}
}

void Save2DArray(double** array, int n) {
	std::ofstream file;
	file.open("./Data/MatrixJ.csv");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			file << array[i][j] << ",";
		}
		file << "\n";
	}
	file.close();
}
void Print1DArray(double* array, int n) {
	for (int i = 0; i < n; i++) {
		std::cout << array[i] << "\t";
	}
	std::cout << "\n";
}

void Save1DArray(double* array, int n) {
	std::ofstream file;
	file.open("./Data/MatrixF.csv");
	for (int i = 0; i < n; i++) {
		file << array[i] << "\n";
	}
	file.close();
}


std::string genAddress(std::string str) {

	return ("./Data/" + str);

} 
std::string to_sci(double num) {
	std::ostringstream streamObj;
	streamObj << num;
	std::string StrObj = streamObj.str();
	return StrObj;
}
std::string to_sci(int num) {
	std::ostringstream streamObj;
	streamObj << num;
	std::string StrObj = streamObj.str();
	return StrObj;
}
bool dcomp(double a, double b, double epsilon) {
	return (fabs(a - b) < epsilon);
}
void writeStartTimeToLog() {
	std::ofstream logFile("./Log/log.txt",std::ios::app);
	auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	char str[26];
	ctime_s(str, 26, &start_time);
	std::cout << "Start at" << str << "\n";
	logFile <<"Start at: "<< str <<"\n";
	logFile.close();
}
void writeEndTimeToLog() {
	std::ofstream logFile("./Log/log.txt", std::ios::app);
	auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	char str[26];
	ctime_s(str, 26, &start_time);
	std::cout << "End at:" << str << '\n';
	logFile << "End at: "<< str << "\n\n\n\n\n";
	logFile.close();
}

void startAlphaToLog() {
	std::ofstream alphaFile("./Log/alpha.txt", std::ios::app);
	auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	char str[26];
	ctime_s(str, 26, &start_time);
	alphaFile << "Start: " << str << "\n\n\n\n\n";
	alphaFile.close();
}

void endAlphaToLog() {
	std::ofstream alphaFile("./Log/alpha.txt", std::ios::app);
	auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	char str[26];
	ctime_s(str, 26, &start_time);
	alphaFile << "End: " << str << "\n\n\n\n\n";
	alphaFile << "\n";
	alphaFile.close();
}
double cal_slope(const std::vector<double>& x, const std::vector<double>& y) {
	assert(x.size() == y.size() && "input size mismatch in slope!");
	const auto n = x.size();
	const auto s_x = std::accumulate(x.begin(), x.end(), 0.0);
	const auto s_y = std::accumulate(y.begin(), y.end(), 0.0);
	const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
	const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
	const auto a = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
	return a;
}
double cal_alpha(const std::vector<double> &CV_theta, const std::vector<double> &CV_flux) {
	std::vector<double>  x_theta= {};
	std::vector<double>  y_flux = {};
	assert(CV_theta.size()==CV_flux.size() &&"Input size does not mathch in cal_alpha");
	double min = 0.0;
	int min_index = 0;
	for (int i = 0; i < static_cast<int>(CV_flux.size()); i++) {
		if (CV_flux[i] < min) {
			min = CV_flux[i];
			min_index = i;
		}
	}

	for (int i = 0; i < min_index;i++) {
		if (CV_flux[i]<min * 0.1 && CV_flux[i]>min * 0.3) {
			x_theta.push_back(CV_theta[i]);
			y_flux.push_back(CV_flux[i]);
		}
	}
	assert(x_theta.size() == y_flux.size() && "Output size does not mathch in cal_alpha");
	
	//get log of abs(y_flux)
	for (int i = 0; i < static_cast<int>(y_flux.size()); i++) {
		y_flux[i] = log(fabs(y_flux[i]));
	}

	//find slope between y_flux and x_theta;
	double slope = 0.0;

	slope = cal_slope(x_theta, y_flux);
	return -slope;


}


void colorP(std::string str, int color = 244) {
	HANDLE hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	SetConsoleTextAttribute(hConsole, color);
	std::cout << str;
	SetConsoleTextAttribute(hConsole, 2);
}


void estimateRunTime(int startTime) {
	int timeElapsed = static_cast<int>(GetTickCount64()) - startTime;
	double timeTotal = double(timeElapsed / 600.0);
	std::string str{ "Total Run Time is " };
	std::string unit{ " Minutes" };
	colorP(str+to_sci(timeTotal)+unit, 245);
	//std::cout << "total run time is " << timeTotal << "Minutes\n";
}