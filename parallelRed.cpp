#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
using namespace std;

// Parallel Min
void parallel_reduce_min(const vector<int>& data, int& min_val) {
    min_val = data[0];
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < data.size(); i++) {
        if (data[i] < min_val)
            min_val = data[i];
    }
}

// Parallel Max
void parallel_reduce_max(const vector<int>& data, int& max_val) {
    max_val = data[0];
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < data.size(); i++) {
        if (data[i] > max_val)
            max_val = data[i];
    }
}

// Parallel Sum
void parallel_reduce_sum(const vector<int>& data, int& sum_val) {
    sum_val = 0;
    #pragma omp parallel for reduction(+:sum_val)
    for (int i = 0; i < data.size(); i++) {
        sum_val += data[i];
    }
}

// Parallel Average
void parallel_reduce_avg(const vector<int>& data, float& avg_val) {
    int sum = 0;
    parallel_reduce_sum(data, sum);
    avg_val = static_cast<float>(sum) / data.size();
}

int main() {
    int size;
    cout << "Enter number of elements: ";
    cin >> size;

    if (size <= 0) {
        cout << "Size must be greater than 0." << endl;
        return 1;
    }

    vector<int> data(size);
    cout << "Enter " << size << " elements:\n";
    for (int i = 0; i < size; i++) {
        cin >> data[i];
    }

    int min_val, max_val, sum_val;
    float avg_val;

    parallel_reduce_min(data, min_val);
    parallel_reduce_max(data, max_val);
    parallel_reduce_sum(data, sum_val);
    parallel_reduce_avg(data, avg_val);

    cout << "\nResults using Parallel Reduction (OpenMP):\n";
    cout << "Minimum: " << min_val << endl;
    cout << "Maximum: " << max_val << endl;
    cout << "Sum: " << sum_val << endl;
    cout << "Average: " << avg_val << endl;

    return 0;
}
