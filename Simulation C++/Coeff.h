
class Coeff{
    double xi;
    double xm;
    double deltaX_m;
    double deltaX_p;
public:
    double beta;
    double gamma;

	double concB;
	double alpha;
	double K0;
	double Kf;
	double Kb;
	double dB;
	double dY;
	double dZ;
    int n;
	double* aA = nullptr;
	double* bA = nullptr;
	double* cA = nullptr;
	double* aB = nullptr;
	double* bB = nullptr;
	double* cB = nullptr;
	double* aY = nullptr;
	double* bY = nullptr;
	double* cY = nullptr;
	double* aZ = nullptr;
	double* bZ = nullptr;
	double* cZ = nullptr;
	double* d = nullptr;
	double* XX = nullptr;
	double* dx = nullptr;
	double* fx = nullptr;
	double** A = nullptr;
	double** J = nullptr;
	Coeff(double maxX, double K0_input, double Kf_input, double Kb_input, double deltaT, double alpha_input, double cAstar, double gamma, double dB_input, double dY_input,double dZ_input);
    ~Coeff();
    void calc_n(double dx);
	void ini_jacob();
	void ini_fx();
	void ini_dx();
	void calc_fx(double* x,double Theta);
	void negative_fx();
	void calc_jacob(double* x, double Theta);
    void Acalc_abc(double deltaT, double Theta, double deltaX);
	void Bcalc_abc(double deltaT, double Theta, double deltaX);
	void Ycalc_abc(double deltaT, double Theta, double deltaX);
	void Zcalc_abc(double deltaT, double Theta, double deltaX);
	void Allcalc_abc(double deltaT, double Theta, double deltaX);
    void get_XX(double *xx);
    void update(double* x, double Theta, double deltaX,double concA, double concB, double concC,double concZ);
	void xupdate(double* x);
	double avg_dx();
	double max_dx();
	double avg_abs_dx();
	double avg_abs_fx();
};