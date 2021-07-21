#include <iostream>
#include <cmath>
#include <fstream>
#include <windows.h>
#include <thread>
#include <vector>
#include <thread>
#include <iterator>
#include <mutex>
#include <assert.h>
#include "Grid.h"
#include "Trid.h"
#include "Coeff.h"
#include "convergence.h"
#include "ThomasXL.h"
#include "helper.h"

using namespace std;
std::mutex mlock;
void spawnThread(int n1, double variable1[],int n2, double variable2[], int n3, double variable3[]);
void foo(double variable1, double variable2, double variable3);
double DA = 1e-9;
double E0f = 0.01;
double dElectrode = 1e-6;
void foo(double variable1, double variable2, double variable3) {
	//variable 1 is scan rate, Variable 3 is K0. Now change Varible 2 
	//Input parameters
	double concA = 1.0;
	double concY = 0.0;

	double cTstar = 1e-3;
	double cAstarSI = cTstar * 1000;


	//double theta_i = 20.0;
	double theta_i = 20.0;
	double theta_v = -20.0;
	double sigma = 1.0;


	double deltaX = 0.01;
	double deltaTheta = 0.5;
	double K0 = variable1;
	//K0 /= cAstar;
	//double dimKeq = 10; //unit is M^(-1)
	//double Kf = 0.0;
	//double dimKf = Kf * DA / (cAstar * dElectrode * dElectrode);
	double dimKeq = variable2;
	double dimKf = variable3;
	double dimKb = dimKf / dimKeq;


	double Kf = dimKf * dElectrode * dElectrode / DA;


	double Kb = dimKb * cTstar * dElectrode * dElectrode / DA;
	std::cout << "Kf is " << Kf << "Kb is " << Kb << "\n";

	double Keq = dimKeq / cTstar;

	double concB = 0.0;
	double concZ = 0.0;


	//using newton's method to solve for concA,concB and concZ; 

	double cAstar = cTstar;

	double dc = 0.0;
	
	do {
		dc = (sqrt(cAstar*dimKeq)+cAstar-cTstar) / (0.5*sqrt(dimKeq)*pow(cAstar,-0.5)+1.0);
		cAstar = cAstar - dc;
		std::cout << "cAstar now is " << cAstar << "\n";

	} while (dc > 1e-10);


	// New concentration definition
	concA = cAstar / cTstar;
	concB = sqrt(cAstar * dimKeq) / cTstar;
	concZ = sqrt(cAstar * dimKeq) / cTstar;
	concY = 0.0;


	//concB = 1.0;



	std::cout << concA << "\t" << concB << "\t" << concY << "\t" << concZ << "\t";
  

	//	std::cout << "dimKf is " << dimKf << " dimKb is " << dimKb << " Kf is " << Kf << " Kb is " << Kb << "\n";

	double alpha = 0.5;
	double gamma = 0.1;
	//if (sigma > 38943) {
	//	gamma = 0.02;
	//}

	double dB = 1.0;
	double dY = 1.0;
	double dZ = 1.0;
	double epsilon = 1e-2;
	int numOfIteration = 5000;
	int maxNumOfIteration = 5000;

	std::vector<double> CV_theta = {};
	std::vector<double> CV_flux = {};
	double calculated_alpha = 0.0;

	bool outputConcProfile = false;
	bool Print = false; // if true, will not print concentration profiles. 
	bool printA = true;  //if print forward half peak max profile.
	bool printB = true;	 //if print forward  peak max profile.
	bool printC = true;	 //if print theta_v peak max profile.	
	bool printD = true;  //if print backward half peak max profile
	bool printE = true;  // if print backward peak max profile
	if (!Print) {
		printA = false;
		printB = false;
		printC = false;
		printD = false;
		printE = false;
	}
	//the point where needs to be printed
	double pointA = 10.0;
	double pointB = 5.0;
	double pointC =0.0;
	double pointD = -10.0;
	double pointE = theta_v;

	//Calcute the maximums of time and position 
	double deltaT = deltaTheta / sigma;
	double maxT = 2.0 * fabs(theta_v - theta_i) / deltaT;
	double maxX = 1.0;//6.0 * sqrt(maxT);
	Coeff coeff(maxX, K0, Kf, Kb, deltaT, alpha, concB, gamma, dB, dY, dZ);
	mlock.lock();
	coeff.calc_n(deltaX);
	mlock.unlock();
	int m = (int)2.0 * fabs(theta_v - theta_i) / deltaTheta;
	coeff.ini_jacob();
	coeff.ini_fx();
	coeff.ini_dx();
	Grid grid(coeff.n);
	grid.grid(deltaX, gamma);
	grid.init_c(concA, concB, concY, concZ);
	std::cout << grid.conc[0] << "\t" << grid.conc[1] << "\t" << grid.conc[2] << "\t" << grid.conc[3] << "\t" << grid.conc[4] << "\t" << grid.conc[5] << "\t" << grid.conc[6] << "\t" << grid.conc[7] << "\t";
	coeff.get_XX(grid.x);
	coeff.update(grid.conc, theta_i, deltaX, concA, concB, concY, concZ);
	coeff.Allcalc_abc(deltaT, theta_i, deltaX);
	coeff.calc_jacob(grid.conc, theta_i);
	coeff.calc_fx(grid.conc, theta_i);

	ThomasXL Trid(coeff.n * 4, 9);
	Trid.solve(coeff.J, coeff.fx, coeff.dx);
	coeff.xupdate(grid.conc);


	for (int i = 0; i < 10; i++) {

		coeff.calc_jacob(grid.conc, theta_i);
		coeff.calc_fx(grid.conc, theta_i);
		Trid.solve(coeff.J, coeff.fx, coeff.dx);
		coeff.xupdate(grid.conc);
	}
	//double logVar2 = log10(variable2);

	std::string BKineticsLocation = "Concentration B C var1=" + to_string(variable1) + "var2=" + to_string(variable2) + "var3=" + to_string(variable3) + ".csv";
	//std::string CVLocation = "Variable =" + to_sci(variable2) + " Sigma="; //+"K1=" + to_sci(K1) + "Sigma=" + to_sci(sigma) + "K0=" + to_sci(K0) + "dB=" + to_sci(dB) + "dY=" + to_sci(dY);
	std::string CVLocation = "var1=" + to_string(variable1) + "var2=" + to_string(variable2) + "var3=" + to_string(variable3) + ".csv"; //+"K1=" + to_sci(K1) + "Sigma=" + to_sci(sigma) + "K0=" + to_sci(K0) + "dB=" + to_sci(dB) + "dY=" + to_sci(dY);

	std::string ConcALocation = "concA=" + to_sci(concA) + "var1=" + to_string(variable1) + "var2=" + to_string(dimKf) + "var3=" + to_string(dimKeq) + ".csv";
	std::string ConcBLocation = "concB=" + to_sci(concB) + "var1=" + to_string(variable1) + "var2=" + to_string(dimKf) + "var3=" + to_string(dimKeq) + ".csv";
	std::string ConcYLocation = "concY=" + to_sci(concY) + "var1=" + to_string(variable1) + "var2=" + to_string(dimKf) + "var3=" + to_string(dimKeq) + ".csv";
	std::string concZLocation = "concZ=" + to_sci(concZ) + "var1=" + to_string(variable1) + "var2=" + to_string(dimKf) + "var3=" + to_string(dimKeq) + ".csv";



	//Calculate the first flux and write it in output file
	grid.grad();
	grid.gradB();

	//std::cout << "first flux is " << grid.g << "\n";
	std::ofstream myFile;
	myFile.open(genAddress(CVLocation));
	myFile << theta_i << "," << grid.g << "\n"; //Dimensionless Form 
	//myFile << (theta_i / (96385 / (8.314 * 298)) + E0f) << "," << (grid.g * 3.1415926 * dElectrode * 96485 * DA * cAstar * 1000.0) << "\n"; //Dimensional Form
	CV_theta.push_back(theta_i);
	CV_flux.push_back(grid.g);
	mlock.lock();
	cout << "Time step is " << m << "\n";
	mlock.unlock();
	std::ofstream bFile;
	bFile.open(genAddress(BKineticsLocation));
	bFile << theta_i << "," << concB <<"," <<concY << "\n";


	double fluxTotal = 0.0;
	int startTime = static_cast<int>(GetTickCount64());
	//Simulate through the given time
	double Theta = theta_i;
	double max_flux = 0.0;
	double min_flux = 0.0;
	double forwardPeak = 0.0;
	double backwardPeak = 0.0;
	std::thread::id this_id = std::this_thread::get_id();
	for (int i = 0; i < (int)m/2; i++) {
		//std::cout << "mass Conservation is: " << conservation(concA, concB, concY, concZ, grid.concA, grid.concB, grid.concY, grid.concZ, grid.x, coeff.n) << "%\n";
		if (i < m / 2) {
			Theta -= deltaTheta;
		}
		else {
			Theta += deltaTheta;
		}
		if (i % 10 == 0) {
			std::cout << "Thread id " << this_id<< "Time Step" << i << '\n';
		}
		if (i == int(m * .01)) {
			estimateRunTime(startTime);
		}
		grid.init_c(concA, concB, concY, concZ);
		/*coeff.Allcalc_abc(deltaT, Theta, deltaX);
		coeff.update(grid.conc, Theta, deltaX);
		Trid.solve(coeff.A, coeff.d, grid.conc);
		grid.grad();*/
		coeff.update(grid.conc, Theta, deltaX, concA,concB, concY,concZ); //get new d from grid.conc;
		coeff.Allcalc_abc(deltaT, Theta, deltaX);
		for (int ii = 0; ii < numOfIteration; ii++) {

			coeff.calc_jacob(grid.conc, Theta);
			coeff.calc_fx(grid.conc, Theta);
			Trid.solve(coeff.J, coeff.fx, coeff.dx);
			coeff.xupdate(grid.conc);
		}
		//std::cout << "max_dX is: " << coeff.max_dx() << "\n";
		if (numOfIteration < maxNumOfIteration && coeff.max_dx() > 1e-11) {
			
			if (numOfIteration < maxNumOfIteration) {
				numOfIteration++;
				std::cout << "/n/nNot converging!/n/n" << Kf << "\t" << "avg_dx()is" << coeff.avg_dx() << "\n" << "max_dX is" << coeff.max_dx() << "\n";
 				colorP(std::string("Increase the number of iterations!")+to_string(numOfIteration), 3);
				std::cout << "Num of Iteration Now is: " << numOfIteration << "\n";
			}
			
		}

		std::cout << "Max dx " <<coeff.max_dx() << "\n";
		grid.updateAll();
		grid.grad();
		grid.gradB();
		//std::cout << "\n\n\n\n\n\n" << grid.g << "\n\n\n";
		//std::cout << "grid" << grid.g << "\t";
		fluxTotal += grid.g * deltaT;
 		myFile << Theta << "," << grid.g << "\n";  //Dimensionless Form
		std::cout << Theta << "," << grid.g << "\n";  //Dimensionless Form
		//myFile << (theta_i / (96385 / (8.314 * 298)) + E0f) << "," << (grid.g * 3.1415926 * dElectrode * 96485 * DA * cAstar * 1000.0) << "\n"; //Dimensional Form
		CV_theta.push_back(Theta);
		CV_flux.push_back(grid.g);
		bFile << Theta << "," << grid.concB[0] <<"," <<grid.concY[0] << "\n";
		// find the maximum flux
		if (grid.g > max_flux) {
			max_flux = grid.g;
			forwardPeak = Theta;

		}
		//find the minimum flux
		if (grid.g < min_flux) {
			min_flux = grid.g;
			backwardPeak = Theta;
		}
		if (printA && (fabs(Theta - pointA) < epsilon && i < m / 2)) {
			std::cout << "Saving PointA at " << Theta << "\n";
			//Point A Forward scan Half Max
			std::string  str = "Point=A,Theta=" + to_string(Theta);
			grid.saveA(genAddress( str + ConcALocation));
			grid.saveB(genAddress( str + ConcBLocation));
			grid.saveY(genAddress(str +  ConcYLocation));
			grid.saveZ(genAddress(str + concZLocation));
			printA = false;
		}
		if (printB && fabs(Theta - pointB) < epsilon && i < m / 2) {
			std::cout << "Saving PointB at " << Theta << "\n";
			std::string  str = "Point=B,Theta=" + to_string(Theta);
			grid.saveA(genAddress(str + ConcALocation));
			grid.saveB(genAddress(str + ConcBLocation));
			grid.saveY(genAddress(str + ConcYLocation));
			grid.saveZ(genAddress(str + concZLocation));
			printB = false;
		}
		if (printC && fabs(Theta - pointC) < epsilon) {
			std::string  str = "Point=C,Theta=" + to_string(Theta);
			std::cout << "Saving PointC at " << Theta << "\n";
			grid.saveA(genAddress(str + ConcALocation));
			grid.saveB(genAddress(str + ConcBLocation));
			grid.saveY(genAddress(str + ConcYLocation));
			grid.saveZ(genAddress(str + concZLocation));
			printC = false;
		}
		if (printD && (fabs(Theta - pointD) < epsilon && i < m / 2)) {
			std::string  str = "Point=D,Theta=" + to_string(Theta);
			std::cout << "Saving PointD at " << Theta << "\n";
			grid.saveA(genAddress(str + ConcALocation));
			grid.saveB(genAddress(str + ConcBLocation));
			grid.saveY(genAddress(str + ConcYLocation));
			grid.saveZ(genAddress(str + concZLocation));
			printD = false;
		}
		if (printE && (fabs(Theta - pointE) < epsilon && i < m / 2)) {
			std::string  str = "Point=E,Theta=" + to_string(Theta);
			std::cout << "Saving PointE at " << Theta << "\n";
			grid.saveA(genAddress(str + ConcALocation));
			grid.saveB(genAddress(str + ConcBLocation));
			grid.saveY(genAddress(str + ConcYLocation));
			grid.saveZ(genAddress(str + concZLocation));
			printE = false;
		}
		//Debug 
		//std::cout.precision(3);
		//std::cout << Theta << " " << grid.conc[0] << " " << grid.conc[1] << " " << grid.conc[2] << " " << grid.conc[3] << " " << grid.conc[4] << " " << grid.conc[5] << " " << grid.conc[0] + grid.conc[1] * 2 + grid.conc[2] * 3 << " " << grid.conc[3] + 2 * grid.conc[4] + grid.conc[5] * 3 << "\n";  //<< grid.conc[6] <<" " << grid.conc[7] << " " << grid.conc[8] << " " << grid.conc[9] << " " << grid.conc[10] << " " << grid.conc[11] << " " << grid.conc[12] << " " << grid.conc[13] << " " << grid.conc[14] << " " << grid.conc[15] << " " << grid.conc[16] << " " << grid.conc[17] << " " << grid.conc[18] << " " << grid.conc[19] << " " << grid.conc[20] << " Gradient: " << -grid.g <<"\n";
		//assert(grid.conc[0] < 1.000000001 && "Error of mass conservation");
		//assert(grid.conc[3] < 1.000000001 && "Error of mass conservation");
		//assert(grid.conc[6] < 1.000000001 && "Error of mass conservation");

		//std::cout << grid.g << " "  <<grid.gB * (2.0) << "  " <<grid.conc[0] << " " <<  grid.conc[1] << "\n";

	}
	//std::cout.precision(9);
	double predIrrFlux = 0.496 * sqrt(alpha) * sqrt(sigma);
	std::cout<<"Kf= " << Kf << "\t" << "Kb= " << Kb << "\n";
	std::cout << "dB= " << dB << "\t" << "dY= " << dY << "\n";
	double diffFromPredIrrFlux = (max_flux - predIrrFlux) / predIrrFlux * 100;
	std::cout << "Difference from Predicted Irr Flux is " << diffFromPredIrrFlux << "% \n";
	double predRevFlux = 0.446 * sqrt(sigma);
	double diffFromPredRevFlux = (max_flux - predRevFlux) / predRevFlux * 100;
	std::cout << "Difference from Predicted Rev Flux is " << diffFromPredRevFlux << "% \n";
	std::cout << "Min Flux is " << min_flux << "\n" << "Forward Scan Peak at " << forwardPeak << "\n";
	std::cout << "Max Flux is " << max_flux << "\n" << "Backward Scan Peak at " << backwardPeak << "\n";
	std::cout << "Peak Separation is " << fabs(forwardPeak - backwardPeak) << " \n";
	std::cout << "flux_integrate is: " << fluxTotal << "\n";
	double massDifference = convergence(grid.concA, grid.x, coeff.n);
	std::cout << "mass difference is " << massDifference << "\n";
	std::cout << "percentage" << (1.0 + fluxTotal / massDifference) * 100.0 << "%" << "\n";
	double massConservation = conservation(concA, concB, concY,concZ, grid.concA, grid.concB, grid.concY,grid.concZ, grid.x, coeff.n);
	std::cout << "mass Conservation is: " << massConservation << "%\n";

	

	calculated_alpha = cal_alpha(CV_theta, CV_flux);
	std::cout << "calculated alpha is: " << calculated_alpha << "\t" << "alpha is: " << alpha << "\n";
	std::cout << "\n\n\n";

	myFile.close();
	bFile.close();


	if (outputConcProfile) {
		std::string  str = "Point=F,Theta=" + to_string(Theta);
		grid.saveA(genAddress(str + ConcALocation));
		grid.saveB(genAddress(str + ConcBLocation));
		grid.saveY(genAddress(str + ConcYLocation));
	}

	// Write log file in here 
	mlock.lock();
	//std::ofstream logFile("C:/Users/nmero/OneDrive - Nexus365/Log/log.txt", std::ios::app); //append to logfile
	std::ofstream logFile(genAddress("log.txt"), std::ios::app); //append to logfile
	logFile << "\n";
	logFile << "Thread ID=" << this_id << "\n";
	logFile << "Sigma = " << to_sci(sigma) << "\t" << "Kf = " << to_sci(Kf) << "\t" << "Kb = " << to_sci(Kb) << "\n";
	logFile << "dB= " << dB << "\t" << "dY= " << dY << "\n";
	logFile << "MaxX = " << maxX << "\t" <<"MaxT = " << maxT << "\n";
	logFile << "Min Flux is " << min_flux << "\t" << "Forward Scan Peak at " << forwardPeak << "\n";
	logFile << "Max Flux is " << max_flux << "\t" << "Backward Scan Peak at " << backwardPeak << "\n";
	logFile << "Difference from Predicted Irr Flux is " << diffFromPredIrrFlux << "% \n";
	logFile << "Difference from Predicted Rev Flux is " << diffFromPredRevFlux << "% \n";
	logFile << "Peak Separation is " << fabs(forwardPeak-backwardPeak) << " \n";
	logFile << "flux_integrate is: " << fluxTotal << "\n";
	logFile << "mass Conservation is: " << massConservation << "%\n";
	logFile << "calculated alpha is: " << calculated_alpha << "\t" << "alpha is: " << alpha << "\n";
	logFile << "deltaX is: " << deltaX << "\n";
	logFile << "deltaTheta is: " << deltaTheta << "\n";
	logFile << "gamma is: " << gamma << "\n";
	logFile << "\n";
	logFile.close();
	std::ofstream alphaFile("C:/Users/nmero/OneDrive - Nexus365/Log/alpha.txt", std::ios::app);
	alphaFile << sigma << "\t" << Kf << "\t" << calculated_alpha << "\n";
	alphaFile.close();
	mlock.unlock();
}
void spawnThread(int n1, double variable1[],int n2,double variable2[], int n3, double variable3[] ) {
	std::cout <<"Maximum Concurrency is: " << (int)std::thread::hardware_concurrency <<"\n";
	//assert((int)(std::thread::hardware_concurrency) >= static_cast<int>(n1 * n2 * n3) && "Threads exceeding Maximum Threads of Hardware!");
	std::vector<std::thread> threads(n1*n2*n3);
	//spawn n1*n2 threads
	int index = 0;

	for (int i = 0; i < n1; i++) {
		for (int j = 0; j < n2; j++) {
			for (int k = 0; k < n3; k++) {
				threads[index] = std::thread(foo, variable1[i], variable2[j], variable3[k]);
				index++;
			}
		}
	}
	for (auto& th : threads) {
		th.join();
	}
}
int main() {



	double variables1[] = {1e9}; //K0

	double variables2[] = {1e-5}; //dimensional keq

	double variables3[] = { 0.0 };  //dimensional Kf

	writeStartTimeToLog();
	spawnThread(std::size(variables1), variables1, std::size(variables2),variables2, size(variables3), variables3);
	writeEndTimeToLog();
	endAlphaToLog();
	


	std::cout << "\a";
	return 0;
}