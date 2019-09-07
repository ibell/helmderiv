// C++ includes
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
using namespace autodiff;

/*
Calculate the n-th derivative of the (one-dimensional) function
*/
template <std::size_t n> double deriv(double x)
{
    using vartype = HigherOrderDual<n>;
    vartype _x = x; // explicit copy construction (leaves the shared_ptr untouched, right?)
    auto f = [](vartype x) -> vartype { return sin(x)*cos(x)*sin(x); };
    if constexpr(n == 1){
        return derivative(f, wrt(_x), at(_x));
    }
    else if constexpr(n == 2){
        return derivative(f, wrt(_x, _x), at(_x));
    }
    else if constexpr(n == 3){
        return derivative(f, wrt(_x, _x, _x), at(_x));
    }
    else if constexpr(n == 4){
        return derivative(f, wrt(_x, _x, _x, _x), at(_x));
    }
    else if constexpr(n == 5){
        return derivative(f, wrt(_x, _x, _x, _x, _x), at(_x));
    }
    else if constexpr(n == 6){
        return derivative(f, wrt(_x, _x, _x, _x, _x, _x), at(_x));
    }
    else if constexpr(n == 7){
        return derivative(f, wrt(_x, _x, _x, _x, _x, _x, _x), at(_x));
    }
    else if constexpr(n == 8){
        return derivative(f, wrt(_x, _x, _x, _x, _x, _x, _x, _x), at(_x));
    }
    else if constexpr(n == 9){
        return derivative(f, wrt(_x, _x, _x, _x, _x, _x, _x, _x, _x), at(_x));
    }
    else if constexpr(n == 10){
        return derivative(f, wrt(_x, _x, _x, _x, _x, _x, _x, _x, _x, _x), at(_x));
    }
    else{
        return 1000000000000000;
        //static_assert(false, "Impossible number of derivatives"); // If I leave this static_assert, it gets hit, but there are no template instantiations out of range.  How does this happen?
    }
}

template <std::size_t n>  void time_one()
{
    auto N = 10000000;
    auto startTime = std::chrono::high_resolution_clock::now();
    double d = 0;
    for (auto rr = 0; rr < N; ++rr){
        d += deriv<n>(0+rr*1e-5);
    }
    auto toc = std::chrono::high_resolution_clock::now();
    auto elap = std::chrono::duration<double>(toc - startTime).count()/static_cast<double>(N)*1e6;
    std::cout << std::to_string(n) << " " << elap << " " << d/N << std::endl;
}

int main()
{
    std::cout << deriv<1>(0) << std::endl;
    std::cout << deriv<2>(0) << std::endl;
    std::cout << deriv<3>(0) << std::endl;
    std::cout << deriv<4>(0) << std::endl;

    time_one<1>();
    time_one<2>();
    time_one<3>();
    time_one<4>();
    time_one<5>();
    time_one<6>();
    time_one<7>();
    std::cout << "done\n";
}