#include <iostream>
#include <cmath>

// Define your function here
// Modify this function according to your specific requirements
double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

int main() {
    double start = 0.0;
    double end = 4.0;
    double step = 0.1;

    for (double x = start; x < end; x += step) {
        double result = f(x);
        std::cout << "f(" << x << ") = " << result << std::endl;
    }

    return 0;
}

