// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace boba
{

/**
 * \brief
 * Returns true if key is undefined as an environment variable or if its value is empty.
 * @param key Environment variable name.
 * @return `true` when `key` is unset or empty.
 */

inline bool is_env_empty(std::string const& key)
{
  const char* val = getenv(key.c_str());
  if (val == nullptr)
  {
    return true;
  }
  std::string_view env_value(val);
  if (env_value.compare("") == 0)
  {
    return true;
  }
  return false;
}

/**
 * \brief
 * Returns false if if key is undefined as an environment variable or if its value is empty.
 * @param key Environment variable name.
 * @return `true` when `key` is set to a non-empty value.
 */

inline bool is_env_nonempty(std::string const& key)
{
  return not(is_env_empty(key));
}

/**
 * \brief
 * Gets the value of an environment variable key. If it does not exist, return an empty string.
 * @param key Environment variable name.
 * @return The environment value, or an empty string when unset.
 */

inline std::string get_env(std::string const& key)
{
  const char* val = getenv(key.c_str());
  return (val == nullptr) ? std::string("") : std::string(val);
}

/**
 * \brief
 * Returns true if key is defined as an environment variable and matches a given value
 * @param key Environment variable name.
 * @param value Expected environment value.
 * @return `true` when `key` is set to `value`.
 */

inline bool env_match(std::string const& key, std::string const& value)
{
  // https://www.geeksforgeeks.org/stdstringcompare-in-c/
  std::string env_value = get_env(key);
  if (env_value.empty())
  {
    return false;
  }
  return (env_value == value);
}

/**
 * \brief
 * Returns true if key is defined as an environment variable and matches a given value or if it is empty
 * Useful for when you want to run something unless another option is specified.
 * @param key Environment variable name.
 * @param value Expected environment value.
 * @return `true` when `key` is empty or matches `value`.
 */

inline bool env_match_or_empty(std::string const& key, std::string const& value)
{
  return (env_match(key, value) or is_env_empty(key));
}

} // namespace boba
