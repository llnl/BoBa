// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
template <typename T>
void expect_value(bool& check, T value, T expected)
{
  if constexpr (std::is_floating_point_v<T>)
  {
    pass_or_fail(check, value - expected, 1.0e-13);
  }
  else
  {
    pass_or_fail_bool(check, value == expected);
  }
}

template <size_t K>
void expect_array_value(
  bool& check,
  const ::boba::Array<size_t, K>& value,
  const ::boba::Array<size_t, K>& expected)
{
  bool arrays_match = true;
  for (size_t i = 0; i < K; ++i)
  {
    arrays_match = arrays_match && value[i] == expected[i];
  }
  pass_or_fail_bool(check, arrays_match);
}

void expect_host_vector_value(
  bool& check,
  const ::boba::Vector<::boba::host_space, size_t>& value,
  const std::vector<size_t>& expected)
{
  bool vectors_match = value.size() == expected.size();
  if (vectors_match)
  {
    auto view = value.const_view();
    for (size_t i = 0; i < expected.size(); ++i)
    {
      vectors_match = vectors_match && view(static_cast<::boba::index_t>(i)) == expected[i];
    }
  }
  pass_or_fail_bool(check, vectors_match);
}

void expect_success(bool& check, const ::boba::argparser::ParseResult& result)
{
  pass_or_fail_bool(check, result.ok());
}

void expect_error_contains(
  bool& check,
  const ::boba::argparser& args,
  const ::boba::argparser::ParseResult& result,
  std::string_view expected)
{
  std::ostringstream os;
  args.print_error(os);
  pass_or_fail_bool(check, !result.ok());
  pass_or_fail_bool(check, os.str().find(expected) != std::string::npos);
}
} // namespace

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();
  std::cout << "Tests for argparser " << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();
  bool check = true;

  std::string scenario;
  int int_value = 0;
  double double_value = 0.0;
  size_t size_value = 0;
  bool verbose = false;
  bool required_flag = false;
  std::string label;
  std::vector<int> dims;
  std::vector<size_t> extents;
  ::boba::Array<size_t, 3> fixed_extents{};
  ::boba::Vector<::boba::host_space, size_t> shape_vector;

  ::boba::argparser args(argc, argv);
  args.add_required_argument(scenario, "-c", "--case", "Test case name.");
  args.add_optional_argument(int_value, "-n", "--number", "Integer value.");
  args.add_optional_argument(double_value, "-d", "--double", "Double value.");
  args.add_optional_argument(size_value, "-s", "--size", "size_t value.");
  args.add_optional_argument(verbose, "-v", "--verbose", "Verbose output.");
  args.add_required_argument(required_flag, "-r", "--required", "Required flag.");
  args.add_optional_argument(label, "-l", "--label", "String value.");
  args.add_optional_argument(dims, "", "--dims", "Integer vector value.");
  args.add_optional_argument(extents, "", "--extents", "size_t vector value.");
  args.add_optional_argument(fixed_extents, "", "--shape-array", "Fixed-rank shape.");
  args.add_optional_argument(shape_vector, "", "--shape-vector", "Dynamic shape.");
  const auto result = args.parse();

  if (scenario == "int_0010")
  {
    expect_success(check, result);
    expect_value(check, int_value, 10);
  }
  else if (scenario == "int_100")
  {
    expect_success(check, result);
    expect_value(check, int_value, 100);
  }
  else if (scenario == "int_neg7")
  {
    expect_success(check, result);
    expect_value(check, int_value, -7);
  }
  else if (scenario == "int_1e3_invalid")
  {
    expect_error_contains(check, args, result, "Wrong option format: -n 1e3");
  }
  else if (scenario == "double_0p011")
  {
    expect_success(check, result);
    expect_value(check, double_value, 0.011);
  }
  else if (scenario == "double_1p00")
  {
    expect_success(check, result);
    expect_value(check, double_value, 1.0);
  }
  else if (scenario == "double_1")
  {
    expect_success(check, result);
    expect_value(check, double_value, 1.0);
  }
  else if (scenario == "double_1e_minus_2_lower")
  {
    expect_success(check, result);
    expect_value(check, double_value, 1.0e-2);
  }
  else if (scenario == "double_1e_minus_2_upper")
  {
    expect_success(check, result);
    expect_value(check, double_value, 1.0e-2);
  }
  else if (scenario == "double_neg2p5")
  {
    expect_success(check, result);
    expect_value(check, double_value, -2.5);
  }
  else if (scenario == "size_0010")
  {
    expect_success(check, result);
    expect_value(check, size_value, static_cast<size_t>(10));
  }
  else if (scenario == "size_100")
  {
    expect_success(check, result);
    expect_value(check, size_value, static_cast<size_t>(100));
  }
  else if (scenario == "size_0")
  {
    expect_success(check, result);
    expect_value(check, size_value, static_cast<size_t>(0));
  }
  else if (scenario == "size_neg1_invalid")
  {
    expect_error_contains(check, args, result, "Wrong option format: -s -1");
  }
  else if (scenario == "bool_string")
  {
    expect_success(check, result);
    pass_or_fail_bool(check, verbose);
    pass_or_fail_bool(check, label == "alpha-beta");
  }
  else if (scenario == "bool_default")
  {
    expect_success(check, result);
    pass_or_fail_bool(check, !verbose);
  }
  else if (scenario == "vector_ints")
  {
    expect_success(check, result);
    pass_or_fail_bool(check, dims == std::vector<int>({2, 4, 6}));
  }
  else if (scenario == "vector_negative_ints")
  {
    expect_success(check, result);
    pass_or_fail_bool(check, dims == std::vector<int>({3, -1, 7}));
  }
  else if (scenario == "vector_size_t")
  {
    expect_success(check, result);
    pass_or_fail_bool(check, extents == std::vector<size_t>({5, 8, 13}));
  }
  else if (scenario == "array_size_t_tokens")
  {
    expect_success(check, result);
    expect_array_value(check, fixed_extents, ::boba::Array<size_t, 3>{2, 4, 6});
  }
  else if (scenario == "array_size_t_literal")
  {
    expect_success(check, result);
    expect_array_value(check, fixed_extents, ::boba::Array<size_t, 3>{3, 5, 8});
  }
  else if (scenario == "array_size_t_wrong_length")
  {
    expect_error_contains(check, args, result, "Wrong option format: --shape-array 1");
  }
  else if (scenario == "vector_boba_tokens")
  {
    expect_success(check, result);
    expect_host_vector_value(check, shape_vector, std::vector<size_t>({7, 9, 11}));
  }
  else if (scenario == "vector_boba_literal")
  {
    expect_success(check, result);
    expect_host_vector_value(check, shape_vector, std::vector<size_t>({1, 3, 5, 7}));
  }
  else if (scenario == "vector_stop_at_option")
  {
    expect_success(check, result);
    pass_or_fail_bool(check, verbose);
    pass_or_fail_bool(check, dims == std::vector<int>({9, 10}));
  }
  else if (scenario == "vector_invalid")
  {
    expect_error_contains(check, args, result, "Wrong option format: --dims nope");
  }
  else if (scenario == "vector_missing_argument")
  {
    expect_error_contains(check, args, result, "Missing argument for the last option: --dims");
  }
  else if (scenario == "bool_duplicate")
  {
    expect_error_contains(check, args, result, "provided multiple times");
  }
  else if (scenario == "unknown_option")
  {
    expect_error_contains(check, args, result, "Unrecognized option: --unknown");
  }
  else if (scenario == "missing_argument")
  {
    expect_error_contains(check, args, result, "Missing argument for the last option: -n");
  }
  else if (scenario == "missing_required")
  {
    expect_error_contains(check, args, result, "Missing required option: --required");
  }
  else
  {
    pass_or_fail_bool(check, false);
  }

  return final_check(check);
}
