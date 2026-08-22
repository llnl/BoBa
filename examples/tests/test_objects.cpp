// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <iostream>
#include <random>

/*
  Test BoBa's object classes
*/

template <typename T>
void test_complex(bool& check)
{
  constexpr boba::execution_space space = boba::default_execution_space;

  {
    boba::complex<T> imag{0.0, 1.0};

    boba_print("test real( )");
    pass_or_fail(check, boba::abs(boba::real(imag) - static_cast<T>(0.0)), boba::epsilon<T>());

    boba_print("test imag( )");
    pass_or_fail(check, boba::abs(boba::imag(imag) - static_cast<T>(1.0)), boba::epsilon<T>());

    boba_print("test conj( )");
    pass_or_fail(check, boba::abs(boba::real(boba::conj(imag)) - static_cast<T>(0.0)), boba::epsilon<T>());
    pass_or_fail(check, boba::abs(boba::imag(boba::conj(imag)) - static_cast<T>(-1.0)), boba::epsilon<T>());

    boba_print("test operator-");
    pass_or_fail(check, boba::abs(imag - imag), boba::epsilon<T>());

    boba_print("test operator-, operator+");
    pass_or_fail(check, boba::abs(imag + (-imag)), boba::epsilon<T>());

    boba_print("test operator*");
    auto imag_squared = imag * imag;
    pass_or_fail(check, boba::abs(boba::real(imag_squared) - static_cast<T>(-1.0)), boba::epsilon<T>());
    pass_or_fail(check, boba::abs(boba::imag(imag_squared) - static_cast<T>(0.0)), boba::epsilon<T>());
    pass_or_fail(check, boba::abs(imag_squared + static_cast<T>(1.0)), boba::epsilon<T>());

    boba_print("test operator* vs scalar");
    auto two = static_cast<T>(2.0);
    pass_or_fail(check, boba::abs(two * imag) - two * boba::abs(imag), boba::epsilon<T>());

    boba_print("test operator*=");
    imag_squared *= imag;
    pass_or_fail(check, boba::abs(boba::real(imag_squared) - static_cast<T>(0.0)), boba::epsilon<T>());
    pass_or_fail(check, boba::abs(boba::imag(imag_squared) + static_cast<T>(1.0)), boba::epsilon<T>());

    boba_print("test operator+=");
    imag_squared += imag;
    pass_or_fail(check, boba::abs(boba::real(imag_squared)), boba::epsilon<T>());
    pass_or_fail(check, boba::abs(boba::imag(imag_squared)), boba::epsilon<T>());

    boba_print("test pow()");
    auto imag_to_imag = boba::pow(imag, imag);
    pass_or_fail(check,
                 boba::abs(boba::real(imag_to_imag) - boba::pow(static_cast<T>(boba::e), -static_cast<T>(boba::pi) / static_cast<T>(2))),
                 boba::epsilon<T>());
    pass_or_fail(check, boba::abs(boba::imag(imag_to_imag)), boba::epsilon<T>());
  }

  {
    boba::complex<T> numerator{4.0, 2.0};
    boba::complex<T> denominator{1.0, 1.0};

    auto quotient = numerator / denominator;
    auto expected_quotient = (numerator * boba::conj(denominator)) / boba::pow(boba::abs(denominator), static_cast<T>(2.0));

    boba_print("test operator/");
    pass_or_fail(check, boba::abs(quotient - expected_quotient), 10 * boba::epsilon<T>());
  }

  {
    boba_print("test norm_frobenius()");

    boba::Tensor<3, space, boba::complex<T>> complex_tensor({10, 8, 12});
    auto complex_tensor_view = complex_tensor.view();

    boba::Tensor<3, space, T> real_tensor(complex_tensor.sizes());
    real_tensor.fill_with_random();
    auto real_tensor_view = real_tensor.const_view();

    boba::Tensor<3, space, T> imaginary_tensor(complex_tensor.sizes());
    imaginary_tensor.fill_with_random();
    auto imaginary_tensor_view = imaginary_tensor.const_view();

    boba::loop<space, 3>(complex_tensor.sizes(), [=] __boba_host_device__(boba::Array<size_t, 3> multiindex)
    {
      complex_tensor_view(multiindex) = boba::complex<T>{real_tensor_view(multiindex), imaginary_tensor_view(multiindex)};
    });

    auto complex_norm = ::boba::norm_frobenius(complex_tensor);
    auto real_norm = ::boba::norm_frobenius(real_tensor);
    auto imaginary_norm = ::boba::norm_frobenius(imaginary_tensor);

    pass_or_fail(check,
                 boba::abs(complex_norm * complex_norm - real_norm * real_norm - imaginary_norm * imaginary_norm),
                 100 * complex_tensor.size() * boba::epsilon<T>());
  }

  {
    boba_print("test norm_inf()");

    boba::Tensor<2, space, boba::complex<T>> complex_tensor({42, 86});
    auto complex_tensor_view = complex_tensor.view();

    boba::Tensor<2, space, T> real_tensor(complex_tensor.sizes());
    real_tensor.fill_with_random();
    auto real_tensor_view = real_tensor.const_view();

    boba::loop<space, 2>(complex_tensor.sizes(), [=] __boba_host_device__(boba::Array<size_t, 2> multiindex)
    {
      auto sign = (boba::sum(multiindex) % 2 == 0) ? T(1) : -T(1);
      complex_tensor_view(multiindex) = boba::complex<T>{real_tensor_view(multiindex), sign * real_tensor_view(multiindex)};
    });

    auto complex_norm = ::boba::norm_inf(complex_tensor);
    auto real_norm = ::boba::norm_inf(real_tensor);

    pass_or_fail(check, boba::abs(complex_norm - boba::sqrt(T(2)) * real_norm), 10 * boba::epsilon<T>());
  }

  {
    boba_print("test norm_l1()");

    boba::Tensor<2, space, boba::complex<T>> complex_tensor({42, 86});
    auto complex_tensor_view = complex_tensor.view();

    boba::Tensor<2, space, T> real_tensor(complex_tensor.sizes());
    real_tensor.fill_with_random();
    auto real_tensor_view = real_tensor.const_view();

    boba::loop<space, 2>(complex_tensor.sizes(), [=] __boba_host_device__(boba::Array<size_t, 2> multiindex)
    {
      auto sign = (boba::sum(multiindex) % 2 == 0) ? T(1) : -T(1);
      complex_tensor_view(multiindex) = boba::complex<T>{real_tensor_view(multiindex), sign * real_tensor_view(multiindex)};
    });

    auto complex_norm = ::boba::norm_l1(complex_tensor);
    auto real_norm = ::boba::norm_l1(real_tensor);

    pass_or_fail(check, boba::abs(complex_norm - boba::sqrt(T(2)) * real_norm), 100 * complex_tensor.size() * boba::epsilon<T>());
  }

  {
    boba_print("test norm_difference_frobenius()");

    boba::Tensor<3, space, boba::complex<T>> left_tensor({11, 17, 13});
    auto left_tensor_view = left_tensor.view();

    boba::Tensor<3, space, boba::complex<T>> right_tensor(left_tensor.sizes());
    auto right_tensor_view = right_tensor.view();

    boba::Tensor<3, space, T> real_tensor(left_tensor.sizes());
    real_tensor.fill_with_random();
    auto real_tensor_view = real_tensor.const_view();

    boba::Tensor<3, space, T> imaginary_tensor(left_tensor.sizes());
    imaginary_tensor.fill_with_random();
    auto imaginary_tensor_view = imaginary_tensor.const_view();

    boba::loop<space, 3>(left_tensor.sizes(), [=] __boba_host_device__(boba::Array<size_t, 3> multiindex)
    {
      auto sign = (boba::sum(multiindex) % 2 == 0) ? T(1) : -T(1);
      left_tensor_view(multiindex) = boba::complex<T>{real_tensor_view(multiindex), T(0)};
      right_tensor_view(multiindex) = boba::complex<T>{T(0), sign * imaginary_tensor_view(multiindex)};
    });

    auto real_norm = ::boba::norm_frobenius(real_tensor);
    auto imaginary_norm = ::boba::norm_frobenius(imaginary_tensor);
    auto difference_norm = ::boba::norm_difference_frobenius(left_tensor, right_tensor);

    pass_or_fail(check,
                 boba::abs(difference_norm * difference_norm - real_norm * real_norm - imaginary_norm * imaginary_norm),
                 100 * left_tensor.size() * boba::epsilon<T>());
  }

  {
    boba_print("test norm_difference_inf()");

    boba::Tensor<1, space, boba::complex<T>> left_tensor({100});
    auto left_tensor_view = left_tensor.view();

    boba::Tensor<1, space, boba::complex<T>> right_tensor({100});
    auto right_tensor_view = right_tensor.view();

    boba::loop<space, 1>(100ul, [=] __boba_host_device__(size_t i)
    {
      left_tensor_view(i) = boba::complex<T>{T(i), T(99 - i)};
      right_tensor_view(i) = boba::complex<T>{T(99 - i), T(i)};
    });

    auto difference_norm = ::boba::norm_difference_inf(left_tensor, right_tensor);

    pass_or_fail(check, boba::abs(difference_norm - T(99) * boba::sqrt(T(2))), 141 * boba::epsilon<T>());
  }
}

template <boba::execution_space space, typename type, size_t size>
void test_array_size()
{
  checkpoint();
  boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
  {
    boba::Array<type, size> arr;
    boba::Array<type, size> const& carr = arr;

    boba_always_assert_equal(carr.size(), size, "Unexpected value");
    if constexpr (size == 0_z)
    {
      boba_always_assert(carr.empty(), "Unexpected value");
      boba_always_assert_equal(arr.begin(), arr.end(), "Unexpected value");
      boba_always_assert_equal(carr.begin(), carr.end(), "Unexpected value");
      boba_always_assert_equal(carr.const_begin(), carr.const_end(), "Unexpected value");
    }
    else
    {
      boba_always_assert(!carr.empty(), "Unexpected value");
      boba_always_assert_lt(arr.begin(), arr.end(), "Unexpected value");
      boba_always_assert_lt(carr.begin(), carr.end(), "Unexpected value");
      boba_always_assert_lt(carr.const_begin(), carr.const_end(), "Unexpected value");
    }

    // check looping and access
    arr.fill(type(3));

    boba_always_assert_equal(boba::sum(carr), type(3) * type(size), "Unexpected value");

    type prod = (size > type(0)) ? type(1) : type(0);
    for (size_t i = 0_z; i < size; ++i)
    {
      prod *= type(3);
    }
    boba_always_assert_equal(boba::product(carr), prod, "Unexpected value");

    for (auto iter = arr.const_begin(); iter != arr.const_end(); ++iter)
    {
      boba_always_assert_equal(*iter, type(3), "Unexpected value");
    }

    for (type const& val : carr)
    {
      boba_always_assert_equal(val, type(3), "Unexpected value");
    }

    for (auto iter = arr.begin(); iter != arr.end(); ++iter)
    {
      boba_always_assert_equal(*iter, type(3), "Unexpected value");
      *iter = type(4);
    }

    for (type& val : arr)
    {
      boba_always_assert_equal(val, type(4), "Unexpected value");
      val = type(5);
    }

    for (size_t i = 0; i < size; ++i)
    {
      boba_always_assert_equal(arr[i], type(5), "Unexpected value");
      boba_always_assert_equal(carr[i], type(5), "Unexpected value");
    }
  });
  boba::detail::synchronize<space>();

  if constexpr (size > 0_z)
  {
    constexpr size_t safe_size = size;

    boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
    {
      type a[safe_size];
      for (size_t i = 0_z; i < safe_size; ++i)
      {
        a[i] = type(i);
      }

      boba::Array<type, safe_size> arr = boba::to_array(a);
      boba_always_assert_equal(arr.size(), safe_size, "Unexpected value");

      for (size_t i = 0; i < safe_size; ++i)
      {
        boba_always_assert_equal(arr[i], type(i), "Unexpected value");
      }
    });
    boba::detail::synchronize<space>();

    boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
    {
      long a[safe_size];
      for (size_t i = 0; i < safe_size; ++i)
      {
        a[i] = long(i + 1); // add 1 to make rocm compile properly
      }

      boba::Array<type, safe_size> arr = boba::typed_array<type>(a);
      boba_always_assert_equal(arr.size(), safe_size, "Unexpected value");

      for (size_t i = 0; i < safe_size; ++i)
      {
        boba_always_assert_equal(arr[i], type(i + 1), "Unexpected value");
      }
    });
    boba::detail::synchronize<space>();

    boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
    {
      boba::Array<long, size> a;
      for (size_t i = 0; i < size; ++i)
      {
        a[i] = long(i);
      }

      boba::Array<type, size> arr = boba::typed_array<type>(a);
      boba_always_assert_equal(arr.size(), size, "Unexpected value");

      for (size_t i = 0; i < size; ++i)
      {
        boba_always_assert_equal(arr[i], type(i), "Unexpected value");
      }
    });
    boba::detail::synchronize<space>();
  }

  boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
  {
    boba::Array<type, size> arr = boba::filled_array<size>(type(4));
    boba_always_assert_equal(arr.size(), size, "Unexpected value");

    for (size_t i = 0; i < size; ++i)
    {
      boba_always_assert_equal(arr[i], type(4), "Unexpected value");
    }
  });
  boba::detail::synchronize<space>();

  boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
  {
    boba::Array<type, size> arr1 = boba::filled_array<size>(type(7));
    boba::Array<type, size> arr2 = boba::filled_array<size>(type(7));
    boba_always_assert_equal(arr1, arr1, "Unexpected value");
    boba_always_assert_le(arr1, arr1, "Unexpected value");
    boba_always_assert_ge(arr1, arr1, "Unexpected value");
    boba_always_assert_equal(arr1, arr2, "Unexpected value");
    boba_always_assert_le(arr1, arr2, "Unexpected value");
    boba_always_assert_ge(arr1, arr2, "Unexpected value");
  });
  boba::detail::synchronize<space>();

  boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
  {
    boba::Array<type, size> arr1 = boba::filled_array<size>(type(4));
    boba::Array<type, size> arr2 = boba::filled_array<size>(type(7));
    if (size > 0_z)
    {
      boba_always_assert_ne(arr1, arr2, "Unexpected value");
      boba_always_assert_lt(arr1, arr2, "Unexpected value");
      boba_always_assert_le(arr1, arr2, "Unexpected value");
      boba_always_assert_ge(arr2, arr1, "Unexpected value");
      boba_always_assert_gt(arr2, arr1, "Unexpected value");
    }
    else
    {
      boba_always_assert_equal(arr1, arr2, "Unexpected value");
    }
  });
  boba::detail::synchronize<space>();

  for (size_t k = 0; k < size; ++k)
  {
    boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
    {
      boba::Array<type, size> arr1 = boba::filled_array<size>(type(4));
      boba::Array<type, size> arr2 = boba::filled_array<size>(type(4));
      arr1[k] = 3;
      boba_always_assert_ne(arr1, arr2, "Unexpected value");
      boba_always_assert_le(arr1, arr2, "Unexpected value");
      boba_always_assert_ge(arr2, arr1, "Unexpected value");
    });
    boba::detail::synchronize<space>();
  }
}

template <boba::execution_space space>
void test_array_structured_binding()
{
  {
    checkpoint();
    boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
    {
      boba::Array<size_t, 1> arr{3};
      auto [val0] = arr;
      boba_always_assert_equal(val0, 3_z, "Unexpected value");
    });
    boba::detail::synchronize<space>();
  }

  {
    checkpoint();
    boba::detail::loop<space>(0_z, 1_z, [=] __boba_host_device__(size_t)
    {
      boba::Array<size_t, 3> arr{3, 2, 1};
      auto& [val0, val1, val2] = arr;
      boba_always_assert_equal(val0, 3_z, "Unexpected value");
      boba_always_assert_equal(val1, 2_z, "Unexpected value");
      boba_always_assert_equal(val2, 1_z, "Unexpected value");
      arr[1] = 5;
      boba_always_assert_equal(val0, 3_z, "Unexpected value");
      boba_always_assert_equal(val1, 5_z, "Unexpected value");
      boba_always_assert_equal(val2, 1_z, "Unexpected value");
    });
    boba::detail::synchronize<space>();
  }
}

// Sanity check boba Array
template <boba::execution_space space>
void test_array()
{
  checkpoint();

  test_array_size<space, size_t, 0>();
  test_array_size<space, double, 0>();

  test_array_size<space, size_t, 1>();
  test_array_size<space, double, 1>();

  test_array_size<space, size_t, 11>();
  test_array_size<space, double, 11>();

  test_array_structured_binding<space>();
}

// Sanity check boba loops
template <boba::execution_space space>
void test_loops(const size_t size0, const size_t size1, const size_t size2, const size_t size3)
{
  checkpoint();

  const size_t total_size = size0 * size1 * size2 * size3;
  auto default_allocator = boba::detail::get_default_allocator<space>();
  auto default_host_allocator = boba::detail::get_default_allocator<boba::host_space>();
  boba::index_t* data = boba::detail::malloc<space, boba::index_t>(total_size, default_allocator);
  boba::index_t* check = boba::detail::malloc<boba::host_space, boba::index_t>(total_size, default_host_allocator);

  {
    checkpoint();
    boba::detail::memset<space>(data, 0, total_size);
    boba::detail::loop<space>(0_z, size0, [=] __boba_host_device__(size_t i)
    {
      boba_always_assert_nonnegative(i, "i less than 0");
      boba_always_assert_lt(i, size0, "i greater than or equal to size0");
      const size_t idx = i;
      data[idx] = idx;
    });
    boba::detail::memcpy<boba::host_space, space>(check, data, total_size);
    boba::detail::synchronize<space>();
    for (size_t i = 0; i < size0; ++i)
    {
      const size_t idx = i;
      boba_always_assert_equal(check[idx], idx, "Unexpected value");
    }
  }

  {
    checkpoint();
    boba::detail::memset<space>(data, 0_z, total_size);
    boba::loop<space, 1>({size0},
                         [=] __boba_host_device__(size_t i)
    {
      boba_always_assert_nonnegative(i, "i less than 0");
      boba_always_assert_lt(i, size0, "i greater than or equal to size0");
      const size_t idx = i;
      data[idx] = idx;
    });
    boba::detail::memcpy<boba::host_space, space>(check, data, total_size);
    boba::detail::synchronize<space>();
    for (size_t i = 0; i < size0; ++i)
    {
      const size_t idx = i;
      boba_always_assert_equal(check[idx], idx, "Unexpected value");
    }
  }

  {
    checkpoint();
    boba::detail::memset<space>(data, 0, total_size);
    // Sanity check for boba loops
    boba::detail::loop_2d<space>(0_z, size0, 0_z, size1, [=] __boba_host_device__(size_t i, size_t j)
    {
      boba_always_assert_nonnegative(i, "i less than 0");
      boba_always_assert_lt(i, size0, "i greater than or equal to size0");
      boba_always_assert_nonnegative(j, "j less than 0");
      boba_always_assert_lt(j, size1, "j greater than or equal to size1");
      const size_t idx = i * size1 + j;
      data[idx] = idx;
    });
    boba::detail::memcpy<boba::host_space, space>(check, data, total_size);
    boba::detail::synchronize<space>();
    for (size_t j = 0; j < size1; ++j)
    {
      for (size_t i = 0; i < size0; ++i)
      {
        const size_t idx = i * size1 + j;
        boba_always_assert_equal(check[idx], idx, "Unexpected value");
      }
    }
  }

  {
    checkpoint();
    boba::detail::memset<space>(data, 0, total_size);
    // Sanity check for boba loops
    boba::loop<space, 2>({size0, size1},
                         [=] __boba_host_device__(boba::Array<size_t, 2> idcs)
    {
      auto [i, j] = idcs;
      boba_always_assert_nonnegative(i, "i less than 0");
      boba_always_assert_lt(i, size0, "i greater than or equal to size0");
      boba_always_assert_nonnegative(j, "j less than 0");
      boba_always_assert_lt(j, size1, "j greater than or equal to size1");
      const size_t idx = i * size1 + j;
      data[idx] = idx;
    });
    boba::detail::memcpy<boba::host_space, space>(check, data, total_size);
    boba::detail::synchronize<space>();
    for (size_t j = 0; j < size1; ++j)
    {
      for (size_t i = 0; i < size0; ++i)
      {
        const size_t idx = i * size1 + j;
        boba_always_assert_equal(check[idx], idx, "Unexpected value");
      }
    }
  }

  {
    checkpoint();
    boba::detail::memset<space>(data, 0, total_size);
    boba::detail::loop_3d<space>(0, size0, 0, size1, 0, size2, [=] __boba_host_device__(size_t i, size_t j, size_t k)
    {
      boba_always_assert_nonnegative(i, "i less than 0");
      boba_always_assert_lt(i, size0, "i greater than or equal to size0");
      boba_always_assert_nonnegative(j, "j less than 0");
      boba_always_assert_lt(j, size1, "j greater than or equal to size1");
      boba_always_assert_nonnegative(k, "k less than 0");
      boba_always_assert_lt(k, size2, "k greater than or equal to size2");
      const size_t idx = (i * size1 + j) * size2 + k;
      data[idx] = idx;
    });
    boba::detail::memcpy<boba::host_space, space>(check, data, total_size);
    boba::detail::synchronize<space>();
    for (size_t k = 0; k < size2; ++k)
    {
      for (size_t j = 0; j < size1; ++j)
      {
        for (size_t i = 0; i < size0; ++i)
        {
          const size_t idx = (i * size1 + j) * size2 + k;
          boba_always_assert_equal(check[idx], idx, "Unexpected value");
        }
      }
    }
  }

  {
    checkpoint();
    boba::detail::memset<space>(data, 0, total_size);
    boba::loop<space, 3>({size0, size1, size2},
                         [=] __boba_host_device__(boba::Array<size_t, 3> idcs)
    {
      auto [i, j, k] = idcs;
      boba_always_assert_nonnegative(i, "i less than 0");
      boba_always_assert_lt(i, size0, "i greater than or equal to size0");
      boba_always_assert_nonnegative(j, "j less than 0");
      boba_always_assert_lt(j, size1, "j greater than or equal to size1");
      boba_always_assert_nonnegative(k, "k less than 0");
      boba_always_assert_lt(k, size2, "k greater than or equal to size2");
      const size_t idx = (i * size1 + j) * size2 + k;
      data[idx] = idx;
    });
    boba::detail::memcpy<boba::host_space, space>(check, data, total_size);
    boba::detail::synchronize<space>();
    for (size_t k = 0; k < size2; ++k)
    {
      for (size_t j = 0; j < size1; ++j)
      {
        for (size_t i = 0; i < size0; ++i)
        {
          const size_t idx = (i * size1 + j) * size2 + k;
          boba_always_assert_equal(check[idx], idx, "Unexpected value");
        }
      }
    }
  }

  checkpoint();

  boba::detail::free<boba::host_space>(check, default_host_allocator);
  boba::detail::free<space>(data, default_allocator);
}

// test empty matrix
template <boba::execution_space space>
void test_matrix_0()
{
  // Initialize a boba matrix
  boba::Matrix<space, double> test0;

  boba_always_assert(test0.empty(), "matrix must start empty");
  boba_always_assert_equal(test0.size(), 0_z, "matrix must start with 0 size");
  boba_always_assert_equal(test0.rows(), 0_z, "matrix must start with 0 rows");
  boba_always_assert_equal(test0.cols(), 0_z, "matrix must start with 0 cols");
  boba_always_assert_equal(test0.capacity(), 0_z, "matrix must start with 0 capacity");
  boba_always_assert_not(test0.data(), "matrix must start with nullptr data");
  boba_always_assert_equal(test0.get_space(), space, "matrix must have same space");
  boba_always_assert_equal(test0.name(), std::string("Matrix"), "Matrix must be named \"Matrix\"");

  // test rename
  {
    test0.rename("test0");
    boba_always_assert_equal(test0.name(), std::string("test0"), "matrix must be named \"test0\"");
  }

  // test ways to construct views
  {
    /* TODO<bugfix> Matrix<space> doesn't have a View (it is derived from tensorbase)
    typename boba::Matrix<space>::View view0 = test0.view();
    boba_always_assert_equal(view0.rows(), test0, "view must have same rows as matrix");
    boba_always_assert_equal(view0.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view0.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::View view1 = test0;
    boba_always_assert_equal(view1.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view1.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view1.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::View view2 = view1;
    boba_always_assert_equal(view2.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view2.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view2.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view3 = test0.view();
    boba_always_assert_equal(view3.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view3.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view3.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view4 = test0.const_view();
    boba_always_assert_equal(view4.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view4.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view4.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view5 = test0;
    boba_always_assert_equal(view5.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view5.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view5.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view6 = view0;
    boba_always_assert_equal(view6.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view6.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view6.data(), test0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view7 = view3;
    boba_always_assert_equal(view7.rows(), test0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view7.cols(), test0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view7.data(), test0.data(), "view must have same data as matrix");
    */
  }

  boba::Matrix<space, double> const& ctest0 = test0;
  boba_always_assert(ctest0.empty(), "matrix must start empty");
  boba_always_assert_equal(ctest0.size(), 0_z, "matrix must start with 0 size");
  boba_always_assert_equal(ctest0.rows(), 0_z, "matrix must start with 0 rows");
  boba_always_assert_equal(ctest0.cols(), 0_z, "matrix must start with 0 cols");
  boba_always_assert_equal(ctest0.capacity(), 0_z, "matrix must start with 0 capacity");
  boba_always_assert_not(ctest0.const_data(), "matrix must start with nullptr data");
  boba_always_assert_equal(ctest0.get_space(), space, "matrix must have same space");
  boba_always_assert_equal(ctest0.name(), std::string("test0"), "matrix must have name \"test0\"");

  // test ways to construct views
  {
    /* TODO<bugfix> matrix doesn't have a ConstView (it is derived from tensorbase)
    typename boba::Matrix<space>::ConstView view3 = ctest0.view();
    boba_always_assert_equal(view3.rows(), ctest0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view3.cols(), ctest0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view3.data(), ctest0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view4 = ctest0.const_view();
    boba_always_assert_equal(view4.rows(), ctest0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view4.cols(), ctest0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view4.data(), ctest0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view5 = ctest0;
    boba_always_assert_equal(view5.rows(), ctest0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view5.cols(), ctest0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view5.data(), ctest0.data(), "view must have same data as matrix");

    typename boba::Matrix<space>::ConstView view7 = view3;
    boba_always_assert_equal(view7.rows(), ctest0.rows(), "view must have same rows as matrix");
    boba_always_assert_equal(view7.cols(), ctest0.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view7.data(), ctest0.data(), "view must have same data as matrix");
    */
  }

  // test ways to copy/move construct matrix
  {
    boba::Matrix<space, double> test1 = test0;
    boba_always_assert_equal(test1.empty(), test0.empty(), "matrix must have same emptiness");
    boba_always_assert_equal(test1.size(), test0.size(), "matrix must have same size");
    boba_always_assert_equal(test1.rows(), test0.rows(), "matrix must have same rows");
    boba_always_assert_equal(test1.cols(), test0.cols(), "matrix must have same cols");
    boba_always_assert_equal(test1.capacity(), test0.size(), "matrix must have capacity same as src size");
    boba_always_assert_not(test1.data(), "matrix must have nullptr data");
    boba_always_assert_equal(test1.get_space(), space, "matrix must have same space");
    boba_always_assert_equal(test1.name(), test0.name(), "matrix must have same name");

    boba::Matrix<space, double> test2 = ctest0;
    boba_always_assert_equal(test2.empty(), ctest0.empty(), "matrix must have same emptiness");
    boba_always_assert_equal(test2.size(), ctest0.size(), "matrix must have same size");
    boba_always_assert_equal(test2.rows(), ctest0.rows(), "matrix must have same rows");
    boba_always_assert_equal(test2.cols(), ctest0.cols(), "matrix must have same cols");
    boba_always_assert_equal(test2.capacity(), ctest0.size(), "matrix must have capacity same as src size");
    boba_always_assert_not(test2.data(), "matrix must have nullptr data");
    boba_always_assert_equal(test2.get_space(), space, "matrix must have same space");
    boba_always_assert_equal(test2.name(), ctest0.name(), "matrix must have same name");

    boba::Matrix<space, double> test3 = std::move(test1);
    boba_always_assert_equal(test3.empty(), test0.empty(), "matrix must have same emptiness");
    boba_always_assert_equal(test3.size(), test0.size(), "matrix must have same size");
    boba_always_assert_equal(test3.rows(), test0.rows(), "matrix must have same rows");
    boba_always_assert_equal(test3.cols(), test0.cols(), "matrix must have same cols");
    boba_always_assert_equal(test3.capacity(), test0.size(), "matrix must have capacity same as src size");
    boba_always_assert_not(test3.data(), "matrix must have nullptr data");
    boba_always_assert_equal(test3.get_space(), space, "matrix must have same space");
    boba_always_assert_equal(test3.name(), test0.name(), "matrix must have same name");
  }

  // test ways to copy/move assign matrix
  {
    boba::Matrix<space, double> test1;
    test1 = test0;
    boba_always_assert_equal(test1.empty(), test0.empty(), "matrix must have same emptiness");
    boba_always_assert_equal(test1.size(), test0.size(), "matrix must have same size");
    boba_always_assert_equal(test1.rows(), test0.rows(), "matrix must have same rows");
    boba_always_assert_equal(test1.cols(), test0.cols(), "matrix must have same cols");
    boba_always_assert_ge(test1.capacity(), test0.size(), "matrix must have capacity at least src size");
    boba_always_assert_not(test1.data(), "matrix must have nullptr data");
    boba_always_assert_equal(test1.get_space(), space, "matrix must have same space");
    boba_always_assert_equal(test1.name(), test0.name(), "matrix must have same name");

    boba::Matrix<space, double> test2;
    test2 = ctest0;
    boba_always_assert_equal(test2.empty(), ctest0.empty(), "matrix must have same emptiness");
    boba_always_assert_equal(test2.size(), ctest0.size(), "matrix must have same size");
    boba_always_assert_equal(test2.rows(), ctest0.rows(), "matrix must have same rows");
    boba_always_assert_equal(test2.cols(), ctest0.cols(), "matrix must have same cols");
    boba_always_assert_ge(test2.capacity(), ctest0.size(), "matrix must have capacity at least src size");
    boba_always_assert_not(test2.data(), "matrix must have nullptr data");
    boba_always_assert_equal(test2.get_space(), space, "matrix must have same space");
    boba_always_assert_equal(test2.name(), ctest0.name(), "matrix must have same name");

    boba::Matrix<space, double> test3;
    test3 = std::move(test1);
    boba_always_assert_equal(test3.empty(), test0.empty(), "matrix must have same emptiness");
    boba_always_assert_equal(test3.size(), test0.size(), "matrix must have same size");
    boba_always_assert_equal(test3.rows(), test0.rows(), "matrix must have same rows");
    boba_always_assert_equal(test3.cols(), test0.cols(), "matrix must have same cols");
    boba_always_assert_ge(test3.capacity(), test0.size(), "matrix must have capacity at least src size");
    boba_always_assert_not(test3.data(), "matrix must have nullptr data");
    boba_always_assert_equal(test3.get_space(), space, "matrix must have same space");
    boba_always_assert_equal(test3.name(), test0.name(), "matrix must have same name");
  }
}

// test sized vector
template <boba::execution_space space>
void test_vector(const size_t rows)
{
  checkpoint();

  //
  // Test boba::Vector constructor with std::vector
  //
  if constexpr (boba::is_host(space))
  {
    checkpoint();
    std::cout << "Construct boba::Vector from std::vector \n";
    std::vector<double> test_vector = {1., 2., 3., 4., 5.};
    ::boba::Vector<space, double> boba_vector(test_vector);

    double diff = 0.;

    auto boba_vector_view = boba_vector.view();

    for (size_t i = 0; i < test_vector.size(); i++)
    {
      diff += ::boba::abs(boba_vector_view(i) - test_vector[i]);
    }
  }

  using vector_type = boba::Vector<space, double>;
  using cpu_vector_type = boba::Vector<boba::host_space, double>;

  // Initialize a boba matrix
  // vector_type test(rows, cols, "test");
  vector_type test({rows});
  test.rename("test");

  if (rows == 0_z)
  {
    boba_always_assert(test.empty(), "vector must have requested emptiness");
    boba_always_assert_not(test.data(), "vector must have requested memoryness");
  }
  else
  {
    boba_always_assert(!test.empty(), "vector must have requested emptiness");
    boba_always_assert(test.data(), "vector must have requested memoryness");
  }
  boba_always_assert_equal(test.size(), rows, "vector must have requested size");
  boba_always_assert_equal(test.get_space(), space, "vector must have same space");
  boba_always_assert_equal(test.name(), std::string("test"), "vector must be named \"test\"");

  checkpoint();

  // Test host/device
  {
    cpu_vector_type cpu_vector = test;
    vector_type test_again = cpu_vector;

    auto diff_test = test_again - test;

    boba_always_assert_lt(::boba::norm_inf(diff_test), 1.0e-09, "Host/device transfers OR operators are wrong.");
  }

  // TODO<feature> additional tests like in test_matrix or test_tensor
}

template <boba::execution_space space>
void test_concatenate()
{
  checkpoint();

  using matrix_type = boba::Matrix<space, double>;
  using cpu_matrix_type = boba::Matrix<boba::host_space, double>;
  using vector_type = boba::Vector<space, double>;
  using cpu_vector_type = boba::Vector<boba::host_space, double>;

  matrix_type left({2, 2});
  matrix_type right({2, 1});
  auto left_view = left.view();
  auto right_view = right.view();

  boba::detail::loop_2d<space>(0_z, left.rows(), 0_z, left.cols(), [=] __boba_host_device__(size_t row, size_t col)
  {
    left_view({row, col}) = double(10 * row + col + 1);
  });

  boba::detail::loop_2d<space>(0_z, right.rows(), 0_z, right.cols(), [=] __boba_host_device__(size_t row, size_t col)
  {
    right_view({row, col}) = double(100 + 10 * row + col + 1);
  });
  boba::detail::synchronize<space>();

  cpu_matrix_type columns = boba::concatenate_columns(left, right);
  boba_always_assert_equal(columns.rows(), 2_z, "Column concatenation row count is wrong.");
  boba_always_assert_equal(columns.cols(), 3_z, "Column concatenation column count is wrong.");
  boba_always_assert_equal(columns({0, 0}), 1.0, "Column concatenation element is wrong.");
  boba_always_assert_equal(columns({0, 1}), 2.0, "Column concatenation element is wrong.");
  boba_always_assert_equal(columns({0, 2}), 101.0, "Column concatenation element is wrong.");
  boba_always_assert_equal(columns({1, 0}), 11.0, "Column concatenation element is wrong.");
  boba_always_assert_equal(columns({1, 1}), 12.0, "Column concatenation element is wrong.");
  boba_always_assert_equal(columns({1, 2}), 111.0, "Column concatenation element is wrong.");

  matrix_type top({1, 3});
  matrix_type bottom({2, 3});
  auto top_view = top.view();
  auto bottom_view = bottom.view();

  boba::detail::loop_2d<space>(0_z, top.rows(), 0_z, top.cols(), [=] __boba_host_device__(size_t row, size_t col)
  {
    top_view({row, col}) = double(col + 1);
  });

  boba::detail::loop_2d<space>(0_z, bottom.rows(), 0_z, bottom.cols(), [=] __boba_host_device__(size_t row, size_t col)
  {
    bottom_view({row, col}) = double(10 * (row + 1) + col + 1);
  });
  boba::detail::synchronize<space>();

  cpu_matrix_type rows = boba::concatenate_rows(top, bottom);
  boba_always_assert_equal(rows.rows(), 3_z, "Row concatenation row count is wrong.");
  boba_always_assert_equal(rows.cols(), 3_z, "Row concatenation column count is wrong.");
  boba_always_assert_equal(rows({0, 0}), 1.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({0, 1}), 2.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({0, 2}), 3.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({1, 0}), 11.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({1, 1}), 12.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({1, 2}), 13.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({2, 0}), 21.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({2, 1}), 22.0, "Row concatenation element is wrong.");
  boba_always_assert_equal(rows({2, 2}), 23.0, "Row concatenation element is wrong.");

  vector_type front({3});
  vector_type back({2});
  auto front_view = front.view();
  auto back_view = back.view();

  boba::detail::loop<space>(0_z, front.size(), [=] __boba_host_device__(size_t i)
  {
    front_view(i) = double(i + 1);
  });

  boba::detail::loop<space>(0_z, back.size(), [=] __boba_host_device__(size_t i)
  {
    back_view(i) = double(10 + i + 1);
  });
  boba::detail::synchronize<space>();

  cpu_vector_type joined = boba::concatenate_vectors(front, back);
  boba_always_assert_equal(joined.size(), 5_z, "Vector concatenation size is wrong.");
  boba_always_assert_equal(joined({0}), 1.0, "Vector concatenation element is wrong.");
  boba_always_assert_equal(joined({1}), 2.0, "Vector concatenation element is wrong.");
  boba_always_assert_equal(joined({2}), 3.0, "Vector concatenation element is wrong.");
  boba_always_assert_equal(joined({3}), 11.0, "Vector concatenation element is wrong.");
  boba_always_assert_equal(joined({4}), 12.0, "Vector concatenation element is wrong.");
}

// test sized matrix
template <boba::execution_space space>
void test_matrix(const size_t rows, const size_t cols)
{
  checkpoint();

  using matrix_type = boba::Matrix<space, double>;
  using cpu_matrix_type = boba::Matrix<boba::host_space, double>;

  // constexpr auto matrix_stride_1_dim = matrix_type::stride_1_dim;
  // constexpr auto matrix_compact_representation = matrix_type::compact_representation;

  // Initialize a boba matrix
  // matrix_type test(rows, cols, "test");
  matrix_type test({rows, cols});
  test.rename("test");

  if (rows * cols == 0_z)
  {
    boba_always_assert(test.empty(), "matrix must have requested emptiness");
    boba_always_assert_not(test.data(), "matrix must have requested memoryness");
  }
  else
  {
    boba_always_assert(!test.empty(), "matrix must have requested emptiness");
    boba_always_assert(test.data(), "matrix must have requested memoryness");
  }
  boba_always_assert_equal(test.size(), rows * cols, "matrix must have requested size");
  boba_always_assert_equal(test.rows(), rows, "matrix must have requested rows");
  boba_always_assert_equal(test.cols(), cols, "matrix must have requested cols");
  boba_always_assert_equal(test.capacity(), rows * cols, "matrix must have requested capacity");
  boba_always_assert_equal(test.get_space(), space, "matrix must have same space");
  boba_always_assert_equal(test.name(), std::string("test"), "matrix must be named \"test\"");

  checkpoint();

  // Test host/device
  {
    cpu_matrix_type cpu_matrix = test;
    matrix_type test_again = cpu_matrix;

    auto diff_test = test_again - test;

    boba_always_assert_lt(::boba::norm_inf(diff_test), 1.0e-09, "Host/device transfers OR operators are wrong.");
  }

  // Test views
  {
    auto view = test.view();
    // boba_always_assert_equal(view.rows(), test.rows(), "view must have same rows as matrix");
    // boba_always_assert_equal(view.cols(), test.cols(), "view must have same cols as matrix");
    boba_always_assert_equal(view.data(), test.data(), "view must have same data as matrix");

    boba::detail::loop_2d<space>(0_z, rows, 0_z, cols, [=] __boba_host_device__(size_t i, size_t j)
    {
      view({i, j}) = i * cols + j;
    });
    boba::detail::synchronize<space>();

    cpu_matrix_type check_matrix = test;
    boba::detail::loop_2d<boba::host_space>(0_z, rows, 0_z, cols, [&](size_t i, size_t j)
    {
      boba_always_assert_equal(check_matrix({i, j}), double(i * cols + j), "matrix must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  // Test norms
  if ((rows >= 1_z) and (cols >= 1_z))
  {
    auto S_rows = ::boba::sum_of_i(rows - 1);
    auto S_cols = ::boba::sum_of_i(cols - 1);
    auto S2_rows = ::boba::sum_of_i2(rows - 1);
    auto S2_cols = ::boba::sum_of_i2(cols - 1);

    double norm_1 = test.matrix_one_norm();
    double expected_norm_1 = S_rows * cols + rows * (cols - 1);
    boba_always_assert_equal(norm_1, expected_norm_1, "Norm evaluation is wrong");

    double norm_inf = test.matrix_inf_norm();
    double expected_norm_inf = (rows - 1) * cols * cols + S_cols;
    boba_always_assert_equal(norm_inf, expected_norm_inf, "Norm evaluation is wrong");

    double norm_frobenius = ::boba::norm_frobenius(test);
    double expected_norm_frobenius = ::boba::sqrt(static_cast<double>(S2_rows * cols * cols * cols + 2 * cols * S_rows * S_cols + rows * S2_cols));
    boba_always_assert_equal(norm_frobenius, expected_norm_frobenius, "Norm evaluation is wrong");
  }

  checkpoint();

  // test resize up
  {
    auto new_rows = rows + 5;
    auto new_cols = cols + 5;

    test.resize({new_rows, new_cols});
    boba_always_assert_equal(test.rows(), new_rows, "matrix must have requested rows");
    boba_always_assert_equal(test.cols(), new_cols, "matrix must have requested cols");
    boba_always_assert_equal(test.size(), new_rows * new_cols, "matrix must have requested size");
    boba_always_assert_equal(test.capacity(), new_rows * new_cols, "matrix must have requested capacity");

    cpu_matrix_type check_matrix = test;
    boba::detail::loop_2d<boba::host_space>(0_z, rows, 0_z, cols, [&](size_t i, size_t j)
    {
      boba_always_assert_equal(check_matrix({i, j}), double(i * cols + j), "matrix must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }
  checkpoint();

  // test resize down cols
  {
    // static_assert(matrix_stride_1_dim == 0, "Stride 1 dimension assumed");
    // static_assert(matrix_compact_representation, "Compact representation assumed");
    auto old_rows = test.rows();
    auto new_cols = cols;
    // auto old_capacity = test.capacity();
    // auto old_data = test.data();

    test.resize({old_rows, new_cols});
    boba_always_assert_equal(test.rows(), old_rows, "matrix must have same rows");
    boba_always_assert_equal(test.cols(), new_cols, "matrix must have requested cols");
    boba_always_assert_equal(test.size(), old_rows * new_cols, "matrix must have requested size");
    // boba_always_assert_equal(test.capacity(), old_capacity, "matrix must have same capacity");
    // boba_always_assert_equal(test.data(), old_data, "matrix must have same memory");

    cpu_matrix_type check_matrix = test;
    boba::detail::loop_2d<boba::host_space>(0_z, rows, 0_z, cols, [&](size_t i, size_t j)
    {
      boba_always_assert_equal(check_matrix({i, j}), double(i * cols + j), "matrix must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test shrink_to_fit
  {
    auto old_rows = test.rows();
    auto old_cols = test.cols();
    auto old_size = test.size();

    // test.shrink_to_fit();
    boba_always_assert_equal(test.rows(), old_rows, "matrix must have same rows");
    boba_always_assert_equal(test.cols(), old_cols, "matrix must have same cols");
    boba_always_assert_equal(test.size(), old_size, "matrix must have same size");
    // boba_always_assert_equal(test.capacity(), old_size, "matrix must have capacity equal to size");

    cpu_matrix_type check_matrix = test;
    boba::detail::loop_2d<boba::host_space>(0_z, rows, 0_z, cols, [&](size_t i, size_t j)
    {
      boba_always_assert_equal(check_matrix({i, j}), double(i * cols + j), "matrix must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test resize down rows
  {
    auto new_rows = rows;
    auto old_cols = test.cols();

    test.resize({rows, old_cols});
    boba_always_assert_equal(test.rows(), new_rows, "matrix must have requested rows");
    boba_always_assert_equal(test.cols(), old_cols, "matrix must have requested cols");
    boba_always_assert_equal(test.size(), new_rows * old_cols, "matrix must have requested size");
    boba_always_assert_ge(test.capacity(), new_rows * old_cols, "matrix must have same capacity");

    cpu_matrix_type check_matrix = test;
    boba::detail::loop_2d<boba::host_space>(0_z, rows, 0_z, cols, [&](size_t i, size_t j)
    {
      boba_always_assert_equal(check_matrix({i, j}), double(i * cols + j), "matrix must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test reset to within capacity
  // {
  //   auto new_rows = (test.rows()+1) / 2;
  //   auto new_cols = (test.cols()+1) / 2;
  //   auto old_capacity = test.capacity();
  //   auto old_data = test.data();

  //   test.reset(new_rows, new_cols);
  //   boba_always_assert_equal(test.rows(), new_rows, "matrix must have expected rows");
  //   boba_always_assert_equal(test.cols(), new_cols, "matrix must have expected cols");
  //   boba_always_assert_equal(test.size(), new_rows*new_cols, "matrix must have expected size");
  //   boba_always_assert_equal(test.capacity(), old_capacity, "matrix must have same capacity");
  //   boba_always_assert_equal(test.data(), old_data, "matrix must have same memory");
  // }

  checkpoint();

  // test reset to above capacity
  // {
  //   auto new_rows = rows + 5;
  //   auto new_cols = cols + 5;

  //   test.reset(new_rows, new_cols);
  //   boba_always_assert_equal(test.rows(), new_rows, "matrix must have expected rows");
  //   boba_always_assert_equal(test.cols(), new_cols, "matrix must have expected cols");
  //   boba_always_assert_equal(test.size(), new_rows*new_cols, "matrix must have expected size");
  //   boba_always_assert_equal(test.capacity(), new_rows*new_cols, "matrix must have same capacity");
  // }

  checkpoint();

  // test set_to_zero
  {
    auto old_rows = test.rows();
    auto old_cols = test.cols();
    auto old_size = test.size();
    auto old_capacity = test.capacity();
    auto old_data = test.data();

    test.fill_with_zeros();
    boba_always_assert_equal(test.rows(), old_rows, "matrix must have same rows");
    boba_always_assert_equal(test.cols(), old_cols, "matrix must have same cols");
    boba_always_assert_equal(test.size(), old_size, "matrix must have same size");
    boba_always_assert_equal(test.capacity(), old_capacity, "matrix must have same capacity");
    boba_always_assert_equal(test.data(), old_data, "matrix must have same memory");

    cpu_matrix_type check_matrix = test;
    boba::detail::loop_2d<boba::host_space>(0_z, test.rows(), 0_z, test.cols(), [&](size_t i, size_t j)
    {
      boba_always_assert_equal(check_matrix({i, j}), 0.0, "matrix must be set to 0");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test reserve to below capacity
  // {
  //   auto old_rows = test.rows();
  //   auto old_cols = test.cols();
  //   auto old_size = test.size();
  //   auto old_capacity = test.capacity();
  //   auto old_data = test.data();

  //   test.reserve((old_rows+1) / 2, (old_cols+1) / 2);
  //   boba_always_assert_equal(test.rows(), old_rows, "matrix must have same rows");
  //   boba_always_assert_equal(test.cols(), old_cols, "matrix must have same cols");
  //   boba_always_assert_equal(test.size(), old_size, "matrix must have same size");
  //   boba_always_assert_equal(test.capacity(), old_capacity, "matrix must have same capacity");
  //   boba_always_assert_equal(test.data(), old_data, "matrix must have same memory");

  //   cpu_matrix_type check_matrix = test;
  //   boba::detail::loop_2d<boba::host_space>(0, test.rows(), 0, test.cols(),
  //       [&](size_t i, size_t j) {
  //     boba_always_assert_equal(check_matrix({i, j}), 0, "matrix must have same data");
  //   });
  //   boba::detail::synchronize<boba::host_space>();
  // }

  checkpoint();

  // test reserve to above capacity
  // {
  //   // static_assert(matrix_compact_representation, "Compact representation assumed");
  //   auto old_rows = test.rows();
  //   auto old_cols = test.cols();
  //   auto old_size = test.size();
  //   auto new_rows_capacity = 2*test.rows();
  //   auto new_cols_capacity = 2*test.cols();

  //   test.reserve(new_rows_capacity, new_cols_capacity);
  //   boba_always_assert_equal(test.rows(), old_rows, "matrix must have same rows");
  //   boba_always_assert_equal(test.cols(), old_cols, "matrix must have same cols");
  //   boba_always_assert_equal(test.size(), old_size, "matrix must have same size");
  //   boba_always_assert_equal(test.capacity(), new_rows_capacity*new_cols_capacity, "matrix must have expected capacity");

  //   cpu_matrix_type check_matrix = test;
  //   boba::detail::loop_2d<boba::host_space>(0, test.rows(), 0, test.cols(),
  //       [&](size_t i, size_t j) {
  //     boba_always_assert_equal(check_matrix({i, j}), 0, "matrix must have same data");
  //   });
  //   boba::detail::synchronize<boba::host_space>();
  // }

  checkpoint();

  // test clear
  {
    auto old_capacity = test.capacity();
    auto old_data = test.data();

    test.clear();
    boba_always_assert_equal(test.rows(), 0_z, "matrix must have 0 rows");
    boba_always_assert_equal(test.cols(), 0_z, "matrix must have 0 cols");
    boba_always_assert_equal(test.size(), 0_z, "matrix must have 0 size");
    boba_always_assert_equal(test.capacity(), old_capacity, "matrix must have same capacity");
    boba_always_assert_equal(test.data(), old_data, "matrix must have same memory");
  }

  checkpoint();

  // test shrink_to_fit
  // {
  //   test.shrink_to_fit();
  //   boba_always_assert_equal(test.rows(), 0, "matrix must have 0 rows");
  //   boba_always_assert_equal(test.cols(), 0, "matrix must have 0 cols");
  //   boba_always_assert_equal(test.size(), 0, "matrix must have 0 size");
  //   boba_always_assert_equal(test.capacity(), 0, "matrix must have 0 capacity");
  //   boba_always_assert_not(test.data(), "matrix must have no memory");
  // }

  checkpoint();
}

// test sized tensor
template <boba::execution_space space, size_t N>
void test_tensor(boba::Array<size_t, N> sizes)
{
  checkpoint();

  using indices_type = boba::Array<size_t, N>;
  using tensor_type = boba::Tensor<N, space, double>;
  using cpu_tensor_type = boba::Tensor<N, boba::host_space, double>;

  // constexpr auto tensor_stride_1_dim = tensor_type::stride_1_dim;
  // constexpr auto tensor_compact_representation = tensor_type::compact_representation;

  auto zeros = boba::filled_array<N>(0_z);
  auto total_size = boba::product(sizes);

  // Initialize a boba tensor
  tensor_type test(sizes, "test");

  if (total_size == 0_z)
  {
    boba_always_assert(test.empty(), "tensor must have requested emptiness");
    boba_always_assert_not(test.data(), "tensor must have requested memoryness");
  }
  else
  {
    boba_always_assert(!test.empty(), "tensor must have requested emptiness");
    boba_always_assert(test.data(), "tensor must have requested memoryness");
  }
  for (size_t i = 0; i < N; ++i)
  {
    boba_always_assert_equal(test.sizes(i), sizes[i], "tensor must have requested sizes");
  }
  boba_always_assert_equal(test.capacity(), total_size, "tensor must have requested capacity");
  boba_always_assert_equal(test.get_space(), space, "tensor must have same space");
  boba_always_assert_equal(test.name(), std::string("test"), "tensor must be named \"test\"");

  checkpoint();

  // Test views
  {
    auto view = test.view();
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(view.sizes(i), test.sizes(i), "view must have same sizes as tensor");
    }
    boba_always_assert_equal(view.data(), test.data(), "view must have same data as tensor");

    boba::loop<space, N>(zeros, sizes, [=] __boba_host_device__(indices_type idcs)
    {
      size_t idx = 0;
      for (size_t i = 0; i < N; ++i)
      {
        idx = idx * sizes[i] + idcs[i];
      }
      view(idcs) = idx;
    });
    boba::detail::synchronize<space>();

    cpu_tensor_type check_tensor = test;
    boba::loop<boba::host_space, N>(zeros, sizes, [&](indices_type idcs)
    {
      size_t idx = 0;
      for (size_t i = 0; i < N; ++i)
      {
        idx = idx * sizes[i] + idcs[i];
      }
      boba_always_assert_equal(check_tensor(idcs), double(idx), "tensor must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }
  checkpoint();

  // test resize up
  {
    auto new_sizes = sizes;
    for (size_t i = 0; i < N; ++i)
    {
      new_sizes[i] += 5;
    }
    auto new_total_size = boba::product(new_sizes);

    test.resize(new_sizes);
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(test.sizes(i), new_sizes[i], "tensor must have requested sizes");
    }
    boba_always_assert_equal(test.size(), new_total_size, "tensor must have requested size");
    boba_always_assert_equal(test.capacity(), new_total_size, "tensor must have requested capacity");

    cpu_tensor_type check_tensor = test;
    boba::loop<boba::host_space, N>(zeros, sizes, [&](indices_type idcs)
    {
      size_t idx = 0;
      for (size_t i = 0; i < N; ++i)
      {
        idx = idx * sizes[i] + idcs[i];
      }
      boba_always_assert_equal(check_tensor(idcs), double(idx), "tensor must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test resize down last dimension
  {
    // static_assert(tensor_stride_1_dim == 0, "Stride 1 dimension assumed");
    // static_assert(tensor_compact_representation, "Compact representation assumed");
    auto new_sizes = sizes;
    for (size_t i = 0; i < ((N > 0_z) ? N - 1_z : N); ++i)
    {
      new_sizes[i] += 5_z;
    }
    auto new_total_size = boba::product(new_sizes);

    test.resize(new_sizes);
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(test.sizes(i), new_sizes[i], "tensor must have requested sizes");
    }
    boba_always_assert_equal(test.size(), new_total_size, "tensor must have requested size");
    // boba_always_assert_equal(test.capacity(), old_capacity, "tensor must have same capacity");
    // boba_always_assert_equal(test.data(, old_data, "tensor must have same memory");

    cpu_tensor_type check_tensor = test;
    boba::loop<boba::host_space, N>(zeros, sizes, [&](indices_type idcs)
    {
      size_t idx = 0;
      for (size_t i = 0; i < N; ++i)
      {
        idx = idx * sizes[i] + idcs[i];
      }
      boba_always_assert_equal(check_tensor(idcs), double(idx), "tensor must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test shrink_to_fit
  {
    auto old_sizes = test.sizes();
    auto old_size = test.size();

    // test.shrink_to_fit();
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(test.sizes(i), old_sizes[i], "tensor must have same sizes");
    }
    boba_always_assert_equal(test.size(), old_size, "tensor must have same size");
    // boba_always_assert_equal(test.capacity(), old_size, "tensor must have capacity equal to size");

    cpu_tensor_type check_tensor = test;
    boba::loop<boba::host_space, N>(zeros, sizes, [&](indices_type idcs)
    {
      size_t idx = 0;
      for (size_t i = 0; i < N; ++i)
      {
        idx = idx * sizes[i] + idcs[i];
      }
      boba_always_assert_equal(check_tensor(idcs), double(idx), "tensor must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test resize down rest of dimensions
  {
    // auto old_capacity = test.capacity();

    test.resize(sizes);
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(test.sizes(i), sizes[i], "tensor must have same sizes");
    }
    boba_always_assert_equal(test.size(), total_size, "tensor must have requested size");
    // boba_always_assert_ge(test.capacity(), old_capacity, "tensor must have same capacity");

    cpu_tensor_type check_tensor = test;
    boba::loop<boba::host_space, N>(zeros, sizes, [&](indices_type idcs)
    {
      size_t idx = 0;
      for (size_t i = 0; i < N; ++i)
      {
        idx = idx * sizes[i] + idcs[i];
      }
      boba_always_assert_equal(check_tensor(idcs), double(idx), "tensor must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test reset to within capacity
  // {
  //   auto new_sizes = test.sizes();
  //   for (size_t i = 0; i < N; ++i) {
  //     new_sizes[i] = (new_sizes[i]+1) / 2;
  //   }
  //   auto old_capacity = test.capacity();
  //   auto old_data = test.data();

  //   test.reset(new_sizes);
  //   for (size_t i = 0; i < N; ++i) {
  //     boba_always_assert_equal(test.sizes(i), new_sizes[i], "tensor must have expected sizes");
  //   }
  //   boba_always_assert_equal(test.size(), boba::product(new_sizes), "tensor must have expected size");
  //   boba_always_assert_equal(test.capacity(), old_capacity, "tensor must have same capacity");
  //   boba_always_assert_equal(test.data(), old_data, "tensor must have same memory");
  // }

  checkpoint();

  // test reset to above capacity
  // {
  //   auto new_sizes = test.sizes();
  //   for (size_t i = 0; i < N; ++i) {
  //     new_sizes[i] += 5;
  //   }
  //
  //   test.reset(new_sizes);
  //   for (size_t i = 0; i < N; ++i) {
  //     boba_always_assert_equal(test.sizes(i), new_sizes[i], "tensor must have expected sizes");
  //   }
  //   boba_always_assert_equal(test.size(), boba::product(new_sizes), "tensor must have expected size");
  //   boba_always_assert_equal(test.capacity(), boba::product(new_sizes), "tensor must have same capacity");
  // }

  checkpoint();

  // test set_to_zero
  {
    auto old_sizes = test.sizes();
    auto old_size = test.size();
    auto old_capacity = test.capacity();
    auto old_data = test.data();

    test.fill_with_zeros();
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(test.sizes(i), old_sizes[i], "tensor must have same sizes");
    }
    boba_always_assert_equal(test.size(), old_size, "tensor must have same size");
    boba_always_assert_equal(test.capacity(), old_capacity, "tensor must have same capacity");
    boba_always_assert_equal(test.data(), old_data, "tensor must have same memory");

    cpu_tensor_type check_tensor = test;
    boba::loop<boba::host_space, N>(zeros, test.sizes(), [&](indices_type idcs)
    {
      boba_always_assert_equal(check_tensor(idcs), 0.0, "tensor must be set correctly");
    });
    boba::detail::synchronize<boba::host_space>();
  }

  checkpoint();

  // test clear
  {
    auto old_capacity = test.capacity();
    auto old_data = test.data();

    test.clear();
    for (size_t i = 0; i < N; ++i)
    {
      boba_always_assert_equal(test.sizes(i), 0_z, "tensor must have same sizes");
    }
    boba_always_assert_equal(test.size(), 0_z, "tensor must have 0 size");
    boba_always_assert_equal(test.capacity(), old_capacity, "tensor must have same capacity");
    boba_always_assert_equal(test.data(), old_data, "tensor must have same memory");
  }

  checkpoint();

  // test shrink_to_fit
  // {
  //   test.shrink_to_fit();
  //   for (size_t i = 0; i < N; ++i) {
  //     boba_always_assert_equal(test.sizes(i), 0, "tensor must have same sizes");
  //   }
  //   boba_always_assert_equal(test.size(), 0, "tensor must have 0 size");
  //   boba_always_assert_equal(test.capacity(), 0, "tensor must have 0 capacity");
  //   boba_always_assert_not(test.data(), "tensor must have no memory");
  // }

  checkpoint();
}

// test matrix print capability
template <boba::execution_space space>
void test_matrix_print()
{
  // Initialize a boba matrix
  boba::Matrix<space, double> test({4, 4});

  checkpoint();
  // Test views
  auto test_view = test.view();
  ::boba::detail::loop_2d<space>(0_z, 4_z, 0_z, 4_z, [=] __boba_host_device__(size_t i, size_t j)
  {
    test_view({i, j}) = i + j;
  });
  boba::detail::synchronize<space>();

  checkpoint();
  test.print();

  checkpoint();
  // Test data()
  double* data = test.data();
  ::boba::detail::loop<space>(0_z, 16, [=] __boba_host_device__(size_t i)
  {
    data[i] = i;
  });
  boba::detail::synchronize<space>();

  checkpoint();
  test.print();

  checkpoint();
  test.resize({5, 5});

  checkpoint();
  test.fill_with_zeros();

  checkpoint();
  test.print();
}

template <boba::execution_space space>
void test_multiindex()
{
  using tensor3_type = ::boba::Tensor<3, space, double>;
  using view3_type = typename tensor3_type::view_type;
  checkpoint();
  ::boba::Array<size_t, 3> sizes = {2, 4, 3};
  size_t length = ::boba::product(sizes);
  ::boba::detail::loop<space>(
    0_z, length, [=] __boba_host_device__(size_t new_index)
  {
    auto indices = view3_type::multiindex(sizes, new_index);
    auto strides = view3_type::precompute_strides(sizes);

    size_t new_id = view3_type::index(strides, indices);

    boba_always_assert_equal(new_id, new_index, "multindex error");
  });
  boba::detail::synchronize<space>();
}

template <boba::execution_space space>
void test_tensor_shaping()
{
  //
  // Test tensor shaping
  //
  {
    // https://www.mathworks.com/help/matlab/ref/reshape.html
    checkpoint();
    ::boba::Tensor<2, space, double> tensor_A({10, 1});
    tensor_A.rename("tensor_A");

    auto tensor_A_view = tensor_A.view();
    checkpoint();
    ::boba::detail::loop<space>(
      0, 10, [=] __boba_host_device__(size_t i)
    {
      tensor_A_view({i, 0_z}) = 1_z + i;
    });

    ::boba::Tensor<2, space, double> tensor_B({5, 2});
    tensor_B.rename("tensor_B");

    auto tensor_B_view = tensor_B.view();
    checkpoint();
    ::boba::detail::loop_2d<space>(
      0_z, 5_z, 0_z, 2_z, [=] __boba_host_device__(size_t i, size_t j)
    {
      tensor_B_view({i, j}) = 1_z + i + 5 * j;
    });

    ::boba::Tensor<2, space, double> tensor_B_from_A(tensor_B.sizes());
    tensor_B_from_A.rename("tensor_B_from_A");
    tensor_B_from_A.reshape(tensor_A);

    bool check = true;
    pass_or_fail(check, boba::norm_difference_inf(tensor_B, tensor_B_from_A), 1.0e-13);
    boba_always_assert(check, "Unexpected value");
  }

  //
  // Test tensor move semantics
  //
  {
    checkpoint();
    ::boba::Array<size_t, 3> sizes{3_z, 4_z, 2_z};

    ::boba::Tensor<3, space, double> tensor_A(sizes);
    tensor_A.fill_with(1.0);
    tensor_A.rename("tensor_A");
    ::boba::Tensor<3, ::boba::host_space, double> tensor_A_old_host = tensor_A;
    ::boba::Tensor<3, space, double> tensor_A_old = tensor_A;

    ::boba::Tensor<3, ::boba::host_space, double> tensor_B = std::move(tensor_A);

    tensor_B.rename("tensor_B");
    tensor_A_old_host.rename("tensor_A_old_host");

    boba_always_assert_equal(tensor_A.sizes(0_z), 0_z, "Unexpected value");
    boba_always_assert_equal(tensor_A.sizes(1_z), 0_z, "Unexpected value");
    boba_always_assert_equal(tensor_A.sizes(2_z), 0_z, "Unexpected value");
    boba_always_assert_equal(tensor_B.sizes(0_z), 3_z, "Unexpected value");
    boba_always_assert_equal(tensor_B.sizes(1_z), 4_z, "Unexpected value");
    boba_always_assert_equal(tensor_B.sizes(2_z), 2_z, "Unexpected value");
    boba_always_assert_lt(::boba::norm_difference_inf(tensor_A_old_host, tensor_B), 1.0e-11, "Move failed");

    auto move_test = [&]()
    {
      return std::move(tensor_B);
    };

    ::boba::Tensor<3, space, double> tensor_C = move_test();

    boba_always_assert_equal(tensor_B.sizes(0_z), 0_z, "Unexpected value");
    boba_always_assert_equal(tensor_B.sizes(1_z), 0_z, "Unexpected value");
    boba_always_assert_equal(tensor_B.sizes(2_z), 0_z, "Unexpected value");
    boba_always_assert_equal(tensor_C.sizes(0_z), 3_z, "Unexpected value");
    boba_always_assert_equal(tensor_C.sizes(1_z), 4_z, "Unexpected value");
    boba_always_assert_equal(tensor_C.sizes(2_z), 2_z, "Unexpected value");
    boba_always_assert_lt(::boba::norm_difference_inf(tensor_A_old, tensor_C), 1.0e-11, "Move failed");
  }
}

int main(int argc, char* argv[])
{
  boba::detail::ignore(argc);
  boba::detail::ignore(argv);

  boba::splash();
  boba::init();

  bool check = true;

  checkpoint();
  test_array<boba::default_execution_space>();

  checkpoint();
  test_matrix_0<boba::default_execution_space>();

  checkpoint();
  test_matrix_print<boba::default_execution_space>();

  checkpoint();
  test_concatenate<boba::default_execution_space>();

  checkpoint();
  test_multiindex<boba::default_execution_space>();

  checkpoint();
  test_tensor_shaping<boba::default_execution_space>();

  checkpoint();
  std::mt19937 rng(std::random_device{}());
  size_t num_tests = 10;
  size_t max_size = 20;
  for (size_t t = 0; t < num_tests; ++t)
  {
    std::uniform_int_distribution<size_t> distribution((t == 0_z) ? 0_z : 1_z, (t == 0_z) ? 0_z : max_size);

    size_t size0 = distribution(rng);
    size_t size1 = distribution(rng);
    size_t size2 = distribution(rng);
    size_t size3 = distribution(rng);

    std::cout << "Testing with sizes " << size0 << ", " << size1
              << ", " << size2 << ", " << size3 << std::endl;

    checkpoint();
    test_loops<boba::default_execution_space>(size0, size1, size2, size3);

    checkpoint();
    test_vector<boba::default_execution_space>(size0);

    checkpoint();
    test_matrix<boba::default_execution_space>(size0, size1);

    checkpoint();
    // Needs some help
    // test_tensor<boba::default_execution_space>(boba::Array<size_t, 1>{size0});

    checkpoint();
    test_tensor<boba::default_execution_space>(boba::Array<size_t, 2>{size0, size1});

    checkpoint();
    test_tensor<boba::default_execution_space>(boba::Array<size_t, 3>{size0, size1, size2});

    checkpoint();
    // Needs some help
    // test_tensor<boba::default_execution_space>(boba::Array<size_t, 4>{size0,size1,size2,size3});

    checkpoint();
  }

  checkpoint();
  std::cout << "Running complex<float> tests." << std::endl;
  test_complex<float>(check);

  checkpoint();
  std::cout << "Running complex<double> tests." << std::endl;
  test_complex<double>(check);

  checkpoint();

  boba::finalize();

  return final_check(check);
}
