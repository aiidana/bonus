#include <iostream>
#include <cmath>
#define n 51
int main() {
    int i, iter = 0;
    double dx, dt, a, b, c, d[n], f[n], alfa[n], betta[n], newP[n], oldP[n], eps, max;
    const double pi = 3.14159265359;
    dx = 1.0 / (n - 1);
    dt = 0.5 * dx * dx;
    eps = 0.0000000001;
    for (i = 0; i < n; i++)
        oldP[i] = sin(pi * i * dx);
    for (i = 0; i < n; i++)
        f[i] = 0.0;
    a = -1.0 / (dx * dx);
    b = 1.0 / dt + 2.0 / (dx * dx);
    c = -1.0 / (dx * dx);
    do {
        for (i = 0; i < n; i++)
            d[i] = oldP[i] / dt + f[i];
        // Прямой ход метода прогонки
        alfa[1] = 0.0;
        betta[1] = 0.0;
        for (i = 1; i < n - 1; i++) {
            alfa[i+1] = -a / (b + c * alfa[i]);
            betta[i+1] = (d[i] - c * betta[i]) / (b + c * alfa[i]);
        }
        // Обратный ход метода прогонки
        newP[n - 1] = 0.0;
        for (i = n - 2; i >= 0; i--) {
            newP[i] = alfa[i+1] * newP[i + 1] + betta[i+1];
        }
        // Проверка на сходимость
        max = 0.0;
        for (i = 0; i < n; i++) {
            if (max < fabs(newP[i] - oldP[i]))
                max = fabs(newP[i] - oldP[i]);
        }
        // Обновление значений для следующей итерации
        for (i = 0; i < n; i++)
            oldP[i] = newP[i];
        iter++;
    } while (max > eps);
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "X\tP\tTrue\n";
    for (i = 0; i < n; i++) {
        double x = i * dx;
        double true_value = exp(-pi * pi * dt * iter) * sin(pi * x);
        std::cout << x << "\t" << newP[i] << "\t" << true_value << std::endl;
    }
    return 0;}
// /2. Thomas
// #include <iostream>
// #include <cmath>
// #define n 101
// int main() {
//     int i, iter = 0;
//     double dx, dt, a, b, c, d[n], alfa[n], betta[n], oldP[n], newP[n], eps, max;
//     const double pi = 3.14159265359, C = 1.0;

//     dx = 1.0 / (n - 1);
//     dt = 0.01 * dx / (2.0 * C);
//     eps = 0,00001;
//     for (i = 0; i < n; i++) {
//         oldP[i] = 0.0;
//         if (i <= n / 2) {
//             oldP[i] = 1.0;
//         }
//     }

//     a = C / dx;
//     b = 1.0 / dt - C / dx;
//     c = 0.0;

//     do {
//         for (i = 0; i < n; i++) {
//             d[i] = oldP[i] / dt;
//         }
//         // Прямой ход метода прогонки
//         alfa[1] = 0.0;
//         betta[1] = 1.0;
//         for (i = 1; i < n - 1; i++) {
//             alfa[i + 1] = -a / (b + c * alfa[i]);
//             betta[i + 1] = (d[i] - c * betta[i]) / (b + c * alfa[i]);
//         }
//         // Обратный ход метода прогонки
//         newP[n - 1] = 0.0;
//         for (i = n - 2; i >= 0; i--) {
//             newP[i] = alfa[i + 1] * newP[i + 1] + betta[i + 1];
//         }
//         // Проверка на сходимость
//         max = 0.0;
//         for (i = 0; i < n; i++) {
//             if (max < fabs(newP[i] - oldP[i])) {
//                 max = fabs(newP[i] - oldP[i]);
//             }
//         }
//         // Обновление значений для следующей итерации
//         for (i = 0; i < n; i++) {
//             oldP[i] = newP[i];
//         }
//         iter++;
//     } while (iter < 100);
//     std::cout << "Iterations: " << iter << std::endl;
//     std::cout << "VARIABLES = \"X\", \"P\"\n";
//     std::cout << "ZONE I=" << n << ", F=POINT\n";
//     for (i = 0; i < n; i++) {
//         std::cout << i * dx << " " << newP[i] << std::endl;
//     }
//     return 0;
// }


// 1 exact
// #include <iostream>
// #include <cmath>
// #define n 51
// int main() {
//     int i, iter = 0;
//     double dx, dt, oldP[n], newP[n], eps, max;
//     const double pi = 3.14159265359;
//     dx = 1.0 / (n - 1);
//     dt = 0.5 * dx * dx;
//     eps = 0.00001;
//     for (i = 0; i < n; i++) {
//         oldP[i] = sin(pi * i * dx);
//     }
//     do {
//         oldP[0] = 0.0;
//         newP[0] = 0.0;
//         oldP[n - 1] = 0.0;
//         newP[n - 1] = 0.0;
//         for (i = 1; i < n - 1; i++) {
//             newP[i] = oldP[i] + dt * ((oldP[i + 1] - 2.0 * oldP[i] + oldP[i - 1]) / (dx * dx));
//         }
//         max = 0.0;
//         for (i = 0; i < n; i++) {
//             if (max < fabs(newP[i] - oldP[i])) {
//                 max = fabs(newP[i] - oldP[i]);
//             }
//         }
//         for (i = 0; i < n; i++) {
//             oldP[i] = newP[i];
//         }
//         iter++;
//     } while (max > eps);
//     std::cout << "VARIABLES = \"X\", \"P\", \"True\"\n";
//     std::cout << "ZONE I=" << n << ", F=POINT\n";
//     for (i = 0; i < n; i++) {
//         double true_value = exp(-pi * pi * dt * iter) * sin(pi * i * dx);
//         std::cout << i * dx << "\t" << newP[i] << "\t" << true_value << std::endl;
//     }
//     return 0;
// }


//2exact
// #include <iostream>
// #include <cmath>
// #define n 101
// int main() {
//     int i, iter = 0;
//     double dx, dt, newP[n], oldP[n], eps, max;
//     const double pi = 3.1415, C = 1.0;
//     dx = 1.0 / (n - 1);
//     dt = 0.01 * dx / (2.0 * C);
//     eps = 0.00001;
//     for (i = 0; i < n; i++) {
//         oldP[i] = 0.0;
//         if (i <= n / 2) {
//             oldP[i] = 1.0;
//         }
//     }
//     do {
//         oldP[0] = 1.0; 
//         newP[0] = 1.0;
//         oldP[n - 1] = 0.0;
//         newP[n - 1] = 0.0;

//         for (i = 1; i < n - 1; i++) {
//             newP[i] = oldP[i] - C * dt * ((oldP[i] - oldP[i - 1]) / dx);
//         }
//         oldP[0] = 1.0; 
//         newP[0] = 1.0;
//         oldP[n - 1] = 0.0;
//         newP[n - 1] = 0.0;
//         max = 0.0;
//         for (i = 0; i < n; i++) {
//             if (max < fabs(newP[i] - oldP[i])) {
//                 max = fabs(newP[i] - oldP[i]);
//             }
//         }
//         for (i = 0; i < n; i++) {
//             oldP[i] = newP[i];
//         }
//         iter++;
//     } while (iter < 10000);
//     std::cout << "Iterations: " << iter << std::endl;
//     // Вывод результатов в консоль
//     std::cout << "VARIABLES = \"X\", \"P\"\n";
//     std::cout << "ZONE I=" << n << ", F=POINT\n";
//     for (i = 0; i < n; i++) {
//         std::cout << i * dx << "\t" << newP[i] << std::endl;
//     }
//     return 0;
// }
