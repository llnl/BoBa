// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "../tests/common.hpp"

/*
  This tutorial explains some the code's debugging mechanisms, as well as common ways to
  take inputs and outputs.
*/

int main(int argc, char *argv[]) {

  /*
    Every time the code runs this splash text + art is displayed
  */
  boba::splash();

  /*
    Init initializes libraries like cusolve and Umpire
  */
  boba::init();

  /*
    Every example, test, and tutorial is also a code test for CI. This works by setting a boolean
    check as true and at the end of the program determining if check is still true.
    Then, we can return 0 if everything is okay (check = true)
    or return 1 (interpreted as a error code) if a check fails (check = false)
  */
  bool check = true;

  std::cout << "BoBa Tutorial " << std::endl;

  /*
    BOBA_CALI_MARK automatically inserts a Caliper profiling region
    into the function it is placed (in this case, main).
    Caliper is used to profile code performance.
    e.g.
      make example_tutorial BOBA_ENABLE_CALIPER=1
    or to profile Caliper externally in examples, exercises, and tests:
      make example_tutorial BOBA_ENABLE_CALIPER_EXTERNAL=1
  */
  BOBA_CALI_MARK

  /*
    Throughout the code you will see a function "checkpoint()".
    Normally, checkpoints turn into nothing at compile time.
    If BOBA_CHECKPOINTS=1 is specified at compile time, then every checkpoint turns into
    a print statement showing the line and file. This is useful for determining
    where a bug is, as you can see what checkpoints you reach before a failure.
  */
  checkpoint();
  /*
    During development, it may be useful to insert checkpoints that are active while keeping all others inactive.
    This is achieved with always_checkpoint(), which will print out line and file information independent of
    whether BOBA_CHECKPOINTS=1 was on or not. (always means ALWAYS!)
    Try this:
      1. Compile with BOBA_CHECKPOINTS=1 (you may need to have a clean build) and see which checkpoints are printed.
      2. Compile without BOBA_CHECKPOINTS=1, note how this following checkpoint is still printed.
  */
  always_checkpoint();

  //
  // Options Parser
  // We have an options parser that looks like python's argparse (though not nearly as feature complete)
  // This is used in tutorials, examples, and tests to easily vary the options for a given test
  //
  size_t some_integer = 5;
  checkpoint()
  {
    /*
      To take input, we use this argparser class that converts the strings of inputs
      into useable ints, doubles, etc.
      These are specified after the executable
      e.g.
        ./example_tutorial.out -n 10
      or
        ./example_tutorial.out --resolution 100
    */
    ::boba::argparser args(argc, argv);

    args.add_optional_argument(some_integer,              // Takes a pointer to the variable some_integer
      "-n",                       // Short single-dash input variable description
      "--resolution",             // Long double-dash input variable description
      "Example of taking in .");  // Helper text related to the variable

    /*
      parse_check checks the inputs and compares it to options added by add_optional_argument and modifies the variables
      that we took references to. In this case, some_integer.

      TRY THIS:
        1. Try running this test with different resolutions (note we print some_integer below)
        2. try running this test with an incorrect input flag like "--peanut_butter", note that parse_check errors out
    */
    args.parse_check();

    // boba_print is a useful macro for printing out objects for which there is
    // a "<<" operator.
    boba_print(some_integer);
  }

  // Asserts
  // The code has a built-in assertion capability. You will see
  // Example of runtime assertion:
  // Enforce that some_integer is positive by ensuring that it is equal to |some_integer|
  /*
    TRY THIS:
    1. Try running this test with some_integer set as a negative value, and note that this assert catches it
  */
  boba_always_assert_equal(some_integer, ::boba::abs(some_integer), "Must be positive.");

  //
  // Environment variables
  // Another way to take input from a user is to check the environment variables
  //
  checkpoint();
  {
    // Check if the environment contains a variable called "MAKE_ME_FAIL"
    bool failure = boba::is_env_nonempty("another_way_to_fail");

    if(failure)
    {
      // Try this after compiling:
      //    another_way_to_fail=whatever ./example_tutorial.out
      // Returning a non-zero from main in C++ means failure
      return 1;
    }

    // Check if the environment variable "MAKE_ME_FAIL" is set to "yes"
    if(boba::env_match("MAKE_ME_FAIL", "yes"))
    {
      // Try this after compiling:
      //    MAKE_ME_FAIL=yes ./example_tutorial.out
      // Returning a non-zero from main in C++ means failure
      return 1;
    }

    // Checks if the evironment contains "print"
    // If yes, gets the value as a string
    bool print_something = boba::is_env_nonempty("print");
    if(print_something)
    {
      // Try this after compiling:
      //    MAKE_ME_FAIL=yes ./example_tutorial.out
      auto print = boba::get_env("print");
      std::cout << "print = " << print << std::endl;
    }
  }

  /*
    Finalize libraries
  */
  checkpoint();
  boba::finalize();

  /*
    If check remains true at the end of the execution, then all the tests passed!
  */
  return final_check(check);
}
// clang-format on
