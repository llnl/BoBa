// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using wallclock = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace boba
{

/**
 * \brief
 * Timer units. See TicToc.
 */

enum struct tictoc_units : int
{
  microseconds,
  milliseconds,
  seconds
};

/**
 * \brief
 * Nifty struct for coarse-grained timings.
 */

template <tictoc_units units = tictoc_units::microseconds>
struct TicToc
{
  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  wallclock _begin;
  wallclock _end;
  size_t duration;
  bool has_ended = false;
  std::string units_string = "err";
  size_t end_and_print_length = 60;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct and start the clock
   * Begins timing immediately.
   */

  TicToc()
  {
    ::boba::detail::device_sync();
    _begin = high_resolution_clock::now();
    if (units == tictoc_units::microseconds)
    {
      units_string = "(us)";
    }
    else if (units == tictoc_units::milliseconds)
    {
      units_string = "(ms)";
    }
    else if (units == tictoc_units::seconds)
    {
      units_string = "(s)";
    }
  }

  // -------------------------------------------------------------------------------------
  // Units
  // -------------------------------------------------------------------------------------

  /**
   * @brief Returns the timer units for this instance.
   * @return Template-selected timer units.
   */
  constexpr tictoc_units this_units() const noexcept
  {
    return units;
  }

  /**
   * \brief
   * Convert a time described in units of units_in to the units of this (default) or units_out (if specified).
   * @tparam units_in Input duration units.
   * @tparam units_out Output duration units.
   * @param duration Duration to convert.
   * @return `duration` converted from `units_in` to `units_out`.
   */

  template <tictoc_units units_in, tictoc_units units_out = units>
  static constexpr double convert(double duration) noexcept
  {
    double duration_temp = double(duration);

    // convert to seconds
    if (units_in == tictoc_units::milliseconds)
    {
      duration_temp = duration_temp / 1.0e3;
    }
    else if (units_in == tictoc_units::microseconds)
    {
      duration_temp = duration_temp / 1.0e6;
    }

    // convert to output from seconds
    if (units_out == tictoc_units::seconds)
    {
      return double(duration_temp);
    }
    if (units_out == tictoc_units::milliseconds)
    {
      return double(duration_temp * 1e3);
    }
    if (units_out == tictoc_units::microseconds)
    {
      return double(duration_temp * 1e6);
    }
    return -1;
  }

  // -------------------------------------------------------------------------------------
  // Printing/diagnostic utilities
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * End the timer and return the duration
   * @return Elapsed time in the timer's units.
   */

  size_t timing()
  {
    this->end();
    return duration;
  }

  // -------------------------------------------------------------------------------------
  // Section: End
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Stops the timer if it has not ended.
   */

  void end()
  {
    if (!has_ended)
    {
      ::boba::detail::device_sync();
      _end = high_resolution_clock::now();
      if (units == tictoc_units::microseconds)
      {
        duration = static_cast<size_t>(duration_cast<microseconds>(_end - _begin).count());
      }
      else if (units == tictoc_units::milliseconds)
      {
        duration = static_cast<size_t>(duration_cast<milliseconds>(_end - _begin).count());
      }
      else if (units == tictoc_units::seconds)
      {
        duration = static_cast<size_t>(duration_cast<seconds>(_end - _begin).count());
      }
      has_ended = true;
    }
  }

  /**
   * \brief
   * End the timer and print the result of the timing.
   * @param name Label printed before the duration.
   * @return Elapsed time in the timer's units.
   */

  size_t end_and_print(std::string_view name)
  {
    this->end();
    const size_t repeat = name.length() < end_and_print_length ? end_and_print_length - name.length() : 0;
    std::string full_name = std::string(name) + std::string(repeat, ' ');

    std::cout << full_name << units_string << ": " << std::setw(8) << duration << std::endl;
    return duration;
  }

  // -------------------------------------------------------------------------------------
  // Section: Matlab-like calls
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * See restart.
   * Restarts the timer.
   */

  void tic()
  {
    this->restart();
  }

  /**
   * \brief
   * See end.
   * @return Elapsed time in the timer's units.
   */

  size_t toc()
  {
    this->end();
    return duration;
  }

  // -------------------------------------------------------------------------------------
  // Section: Restart
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * End the current timer start the duration over again
   * Discards any previous completed interval.
   */

  void restart()
  {
    if (has_ended)
    {
      has_ended = false;
    }
    _begin = high_resolution_clock::now();
  }
};

} // namespace boba
