#include <iostream>

#include <vector>
#include <limits>
#include <iomanip>  // Для форматирования вывода

using namespace std;

// Симплекс-метод для максимизации
pair<double, vector<double>> simplex_method_maximize(const vector<double>& z, const vector<vector<double>>& c, const vector<double>& r) {
    // Количество переменных и ограничений
    int z_func = z.size();
    int n_constraints = r.size();

    // Создаем таблицу симплекс-метода (n_constraints + 1 строк, z_func + n_constraints + 1 столбцов)
    vector<vector<double>> table(n_constraints + 1, vector<double>(z_func + n_constraints + 1, 0));

    // Заполняем целевую функцию (последняя строка таблицы)
    for (int i = 0; i < z_func; ++i) {
        table[n_constraints][i] = z[i];  // Для максимизации используем положительный знак
    }

    // Заполняем матрицу ограничений
    for (int i = 0; i < n_constraints; ++i) {
        for (int j = 0; j < z_func; ++j) {
            table[i][j] = c[i][j];
        }
        table[i][z_func + i] = 1;  // Единичная матрица для дополнительных переменных
        table[i][z_func + n_constraints] = r[i];  // Права часть ограничений
    }

    int iteration = 0;
    cout << "Начальная таблица (Итерация " << iteration << "):\n";
    for (const auto& row : table) {
        for (double value : row) {
            cout << setw(10) << value << " ";
        }
        cout << endl;
    }
    cout << endl;

    // Симплекс-метод
    while (*max_element(table[n_constraints].begin(), table[n_constraints].end() - 1) > 0) {
        ++iteration;
        // Выбираем ведущий столбец (столбец с максимальным значением в целевой функции)
        int pivot_col = distance(table[n_constraints].begin(), max_element(table[n_constraints].begin(), table[n_constraints].end() - 1));

        // Проверяем условия неограниченности
        bool is_unbounded = true;
        for (int i = 0; i < n_constraints; ++i) {
            if (table[i][pivot_col] > 0) {
                is_unbounded = false;
                break;
            }
        }
        if (is_unbounded) {
            throw runtime_error("Задача не имеет решений (неограниченность).");
        }

        // Выбираем ведущую строку
        vector<double> ratios(n_constraints, numeric_limits<double>::infinity());
        for (int i = 0; i < n_constraints; ++i) {
            if (table[i][pivot_col] > 0) {
                ratios[i] = table[i][z_func + n_constraints] / table[i][pivot_col];
            }
        }
        int pivot_row = distance(ratios.begin(), min_element(ratios.begin(), ratios.end()));

        // Приводим ведущий элемент к единице
        double pivot_value = table[pivot_row][pivot_col];
        for (int j = 0; j < z_func + n_constraints + 1; ++j) {
            table[pivot_row][j] /= pivot_value;
        }

        // Обновляем остальные строки
        for (int i = 0; i <= n_constraints; ++i) {
            if (i != pivot_row) {
                double factor = table[i][pivot_col];
                for (int j = 0; j < z_func + n_constraints + 1; ++j) {
                    table[i][j] -= factor * table[pivot_row][j];
                }
            }
        }

        cout << "Таблица после итерации " << iteration << " (Ведущий элемент в строке " << pivot_row + 1 << ", столбце " << pivot_col + 1 << "):\n";
        for (const auto& row : table) {
            for (double value : row) {
                cout << setw(10) << value << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // Получаем оптимальное решение
    vector<double> solution(z_func, 0);
    for (int i = 0; i < n_constraints; ++i) {
        bool is_basic = true;
        int basic_var_index = -1;
        for (int j = 0; j < z_func; ++j) {
            if (table[i][j] == 1 && basic_var_index == -1) {
                basic_var_index = j;
            } else if (table[i][j] != 0) {
                is_basic = false;
                break;
            }
        }
        if (is_basic && basic_var_index != -1) {
            solution[basic_var_index] = table[i][z_func + n_constraints];
        }
    }

    // Возвращаем максимальное значение целевой функции и оптимальное решение
    return {-table[n_constraints][z_func + n_constraints], solution};
}

int main() {
    // Пример задачи
    vector<double> z = {3, 4};  // Коэффициенты целевой функции
    vector<vector<double>> c = {
        {2, 1},
        {1, 1},
        {0, 1},
        {1, 0}
    };  // Ограничения
    vector<double> r = {16, 10, 6, 7};  // Правая часть ограничений

    try {
        auto [max_value, optimal_solution] = simplex_method_maximize(z, c, r);
        cout << "Максимальное значение: " << max_value << endl;
        cout << "Оптимальные решения: ";
        for (double sol : optimal_solution) {
            cout << sol << " ";
        }
        cout << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }

    return 0;
}
