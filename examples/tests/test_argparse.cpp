// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "common.hpp"

#include <sstream>
#include <string>
#include <string_view>

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

  ::boba::argparser args(argc, argv);
  args.add_required_argument(scenario, "-c", "--case", "Test case name.");
  args.add_optional_argument(int_value, "-n", "--number", "Integer value.");
  args.add_optional_argument(double_value, "-d", "--double", "Double value.");
  args.add_optional_argument(size_value, "-s", "--size", "size_t value.");
  args.add_optional_argument(verbose, "-v", "--verbose", "Verbose output.");
  args.add_required_argument(required_flag, "-r", "--required", "Required flag.");
  args.add_optional_argument(label, "-l", "--label", "String value.");
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
