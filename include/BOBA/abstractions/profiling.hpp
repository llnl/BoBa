// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#if defined(BOBA_ENABLE_CALIPER)
#include <caliper/cali.h>
#endif

#ifdef BOBA_HIP
#include <roctracer/roctx.h>
#endif

namespace boba
{

namespace detail
{

/**
 * \brief
 * Profiling class. \n
 * Compile with BOBA_ENABLE_CALIPER to activate Caliper markers throughout the code.
 * By default, data from boba objects like matrix and tensor functions are not included in this
 * unless you compile with BOBA_ENABLE_CALIPER_OBJECTS. \n
 * Ranges can be started, stopped, and restarted.
 * Ranges automatically stop when out of scope.
 */

struct BobaProfilingRange
{

  /**
   * @brief Starts a profiling range.
   * @param _name Range name.
   */
  BobaProfilingRange(const char* _name)
  {
    this->start(_name);
  }

  /** @brief Active profiling range name. */
  const char* name = nullptr;

  /**
   * \brief destructor
   */
  ~BobaProfilingRange()
  {
    this->stop();
  }

  /**
   * @brief Starts a profiling range.
   * @param _name Range name.
   */
  void start(const char* _name)
  {
    name = _name;
#if defined(BOBA_ENABLE_CALIPER)
    ::boba::detail::device_sync();
    CALI_MARK_BEGIN(name);
#endif
#if defined(BOBA_PROFILING)
#if defined(BOBA_CUDA)
    nvtxRangePush(name);
#elif defined(BOBA_HIP)
    roctxRangePush(name);
#endif
#endif
  }

  /**
   * @brief Stops the active profiling range.
   */
  void stop()
  {
    if (name == nullptr)
    {
      return;
    }
#if defined(BOBA_ENABLE_CALIPER)
    ::boba::detail::device_sync();
    CALI_MARK_END(name);
#endif
#if defined(BOBA_PROFILING)
#if defined(BOBA_CUDA)
    nvtxRangePop();
#elif defined(BOBA_HIP)
    roctxRangePop();
#endif
#endif
    name = nullptr;
  }

  /**
   * @brief Replaces the active range with a new one.
   * @param _name New range name.
   */
  void restart(const char* _name)
  {
    this->stop();
    this->start(_name);
  }
};

// TODO<update>
//   change BOBA_CALI_SWITCH(a, b) -> BOBA_CALI_SWITCH(b)

#if defined(BOBA_ENABLE_CALIPER_OBJECTS) || defined(BOBA_PROFILING)
#define BOBA_CALI_OBJECT_MARK ::boba::detail::BobaProfilingRange boba_objects_range_function(__func__);
#define BOBA_CALI_OBJECT_BEGIN(a) ::boba::detail::BobaProfilingRange boba_objects_range(a);
#define BOBA_CALI_OBJECT_SWITCH(a, b) boba_objects_range.restart(b);
#define BOBA_CALI_OBJECT_END(a) boba_objects_range.stop();
#else
#define BOBA_CALI_OBJECT_MARK ::boba::detail::ignore();
#define BOBA_CALI_OBJECT_END(a) ::boba::detail::ignore(a);
#define BOBA_CALI_OBJECT_SWITCH(a, b) ::boba::detail::ignore(a, b);
#define BOBA_CALI_OBJECT_BEGIN(a) ::boba::detail::ignore(a);
#endif

#if defined(BOBA_ENABLE_CALIPER_EXTERNAL) || defined(BOBA_PROFILING) || defined(BOBA_ENABLE_CALIPER)
#define BOBA_CALI_EXTERNAL_MARK ::boba::detail::BobaProfilingRange boba_range_function(__func__);
#define BOBA_CALI_EXTERNAL_BEGIN(a) ::boba::detail::BobaProfilingRange boba_range(a);
#define BOBA_CALI_EXTERNAL_SWITCH(a, b) boba_range.restart(b);
#define BOBA_CALI_EXTERNAL_END(a) boba_range.stop();
#else
#define BOBA_CALI_EXTERNAL_MARK ::boba::detail::ignore();
#define BOBA_CALI_EXTERNAL_END(a) ::boba::detail::ignore(a);
#define BOBA_CALI_EXTERNAL_SWITCH(a, b) ::boba::detail::ignore(a, b);
#define BOBA_CALI_EXTERNAL_BEGIN(a) ::boba::detail::ignore(a);
#endif

#if defined(BOBA_ENABLE_CALIPER) || defined(BOBA_PROFILING)
#define BOBA_CALI_MARK ::boba::detail::BobaProfilingRange boba_range_function(__func__);
#define BOBA_CALI_BEGIN(a) ::boba::detail::BobaProfilingRange boba_range(a);
#define BOBA_CALI_SWITCH(a, b) boba_range.restart(b);
#define BOBA_CALI_END(a) boba_range.stop();
#else
#define BOBA_CALI_MARK ::boba::detail::ignore();
#define BOBA_CALI_END(a) ::boba::detail::ignore(a);
#define BOBA_CALI_SWITCH(a, b) ::boba::detail::ignore(a, b);
#define BOBA_CALI_BEGIN(a) ::boba::detail::ignore(a);
#endif

} // namespace detail

} // namespace boba
