// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace boba
{

/**
 * \brief
 * Command-line options argument parser. Inspired by some of the concept's mfem's OptionsParser \n
 * \details
 * The class is initialized with a list of argument strings, and new options are added with
 * the add_optional_argument and add_required_argument methods. Currently supports bool, int,
 * size_t, double, and string options.
 */

class argparser
{
public:
  enum class ErrorType
  {
    none,
    help,
    unrecognized_option,
    missing_argument,
    duplicate_option,
    invalid_argument,
    missing_required
  };

private:
  using option_value_type = std::variant<
    std::reference_wrapper<bool>,
    std::reference_wrapper<int>,
    std::reference_wrapper<double>,
    std::reference_wrapper<size_t>,
    std::reference_wrapper<std::string>>;

public:
  struct ParseResult
  {
    ErrorType error_type = ErrorType::none;
    size_t error_idx = 0;

    [[nodiscard]]
    bool ok() const noexcept
    {
      return error_type == ErrorType::none;
    }

    [[nodiscard]]
    bool help_requested() const noexcept
    {
      return error_type == ErrorType::help;
    }

    [[nodiscard]]
    explicit operator bool() const noexcept
    {
      return ok();
    }
  };

private:
  template <typename CharT>
  static std::vector<std::string> make_arguments(int argc, CharT* const argv[])
  {
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i)
    {
      arguments.emplace_back(argv[i] == nullptr ? "" : argv[i]);
    }
    return arguments;
  }

  struct Option
  {
    option_value_type value;
    std::string short_name;
    std::string long_name;
    std::string description;
    bool required;

    Option() = delete;

    /**
     * @brief Constructs an option descriptor.
     * @param value_ Reference to storage for the parsed value.
     * @param short_name_ Short option spelling.
     * @param long_name_ Long option spelling.
     * @param description_ Help text.
     * @param req Whether the option is required.
     */
    Option(
      option_value_type value_,
      std::string_view short_name_,
      std::string_view long_name_,
      std::string_view description_,
      bool req)
        : value(value_),
          short_name(short_name_),
          long_name(long_name_),
          description(description_),
          required(req)
    {
    }

    [[nodiscard]]
    bool matches(std::string_view arg) const noexcept
    {
      return arg == short_name || arg == long_name;
    }

    [[nodiscard]]
    std::string_view current_bool_name() const noexcept
    {
      return std::get<std::reference_wrapper<bool>>(value).get() ? std::string_view(long_name)
                                                                 : std::string_view("disabled");
    }

    [[nodiscard]]
    bool takes_argument() const noexcept
    {
      return !std::holds_alternative<std::reference_wrapper<bool>>(value);
    }
  };

  void add_argument(
    option_value_type value,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description,
    bool required)
  {
    options.push_back(
      Option(
        value,
        short_name,
        long_name,
        description,
        required));
  }

  static constexpr std::string_view error_code_name(ErrorType error_type) noexcept
  {
    switch (error_type)
    {
    case ErrorType::none:
      return "none";
    case ErrorType::help:
      return "help";
    case ErrorType::unrecognized_option:
      return "unrecognized_option";
    case ErrorType::missing_argument:
      return "missing_argument";
    case ErrorType::duplicate_option:
      return "duplicate_option";
    case ErrorType::invalid_argument:
      return "invalid_argument";
    case ErrorType::missing_required:
      return "missing_required";
    default:
      return "unknown";
    }
  }

  ParseResult make_result(ErrorType new_error_type, size_t new_error_idx = 0) noexcept
  {
    last_result.error_type = new_error_type;
    last_result.error_idx = new_error_idx;
    return last_result;
  }

  std::vector<std::string> arguments;
  std::vector<Option> options;
  std::vector<char> option_check;
  ParseResult last_result;

public:
  /**
   * \brief
   * Construct a command line option parser from argc/argv.
   * The arguments are immediately copied into owned strings.
   * @param argc Argument count.
   * @param argv Argument vector.
   */

  template <typename CharT>
  argparser(int argc, CharT* const argv[])
      : argparser(make_arguments(argc, argv))
  {
  }

  /**
   * \brief
   * Construct a command line option parser with the provided argument list.
   * @param arguments_ Argument vector.
   */

  explicit argparser(std::vector<std::string> arguments_)
      : arguments(std::move(arguments_))
  {
    last_result = {};
  }

  /**
   * \brief
   * Add an optional boolean option and set `var` to receive the value.
   * Boolean options default to `false` and are enabled when the flag is passed.
   */
  void add_optional_argument(
    bool& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    var = false;
    add_argument(std::ref(var), short_name, long_name, description, false);
  }

  /**
   * \brief
   * Add a required boolean option and set `var` to receive the value.
   * Boolean options default to `false` and are enabled when the flag is passed.
   */
  void add_required_argument(
    bool& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    var = false;
    add_argument(std::ref(var), short_name, long_name, description, true);
  }

  /**
   * \brief
   * Add an optional integer option and set `var` to receive the value.
   */
  void add_optional_argument(
    int& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, false);
  }

  /**
   * \brief
   * Add a required integer option and set `var` to receive the value.
   */
  void add_required_argument(
    int& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, true);
  }

  /**
   * \brief
   * Add an optional double option and set `var` to receive the value.
   */
  void add_optional_argument(
    double& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, false);
  }

  /**
   * \brief
   * Add a required double option and set `var` to receive the value.
   */
  void add_required_argument(
    double& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, true);
  }

  /**
   * \brief
   * Add an optional size_t option and set `var` to receive the value.
   */
  void add_optional_argument(
    size_t& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, false);
  }

  /**
   * \brief
   * Add a required size_t option and set `var` to receive the value.
   */
  void add_required_argument(
    size_t& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, true);
  }

  /**
   * \brief
   * Add an optional string option and set `var` to receive the value.
   */
  void add_optional_argument(
    std::string& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, false);
  }

  /**
   * \brief
   * Add a required string option and set `var` to receive the value.
   */
  void add_required_argument(
    std::string& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_argument(std::ref(var), short_name, long_name, description, true);
  }

  /**
   * \brief
   * Return true if the command line options were parsed successfully.
   * @return `true` when no parsing error was recorded.
   */

  bool no_parameter_errors() const
  {
    return last_result.ok();
  }

  /**
   * \brief
   * Return true if we are flagged to print the help message.
   * @return `true` when help output was requested.
   */

  bool help() const
  {
    return last_result.help_requested();
  }

  /**
   * @brief Checks whether a string is a valid integer.
   * @param s String to validate.
   * @return `1` when `s` is a valid integer, otherwise `0`.
   */
  bool parse_int(std::string_view s, int& value) const
  {
    if (s.empty())
    {
      return false;
    }

    const auto begin = s.data();
    const auto end = begin + s.size();
    const auto [ptr, error_code] = std::from_chars(begin, end, value);
    return error_code == std::errc{} && ptr == end;
  }

  /**
   * @brief Checks whether a string is a valid nonnegative integer.
   * @param s String to validate.
   * @return `1` when `s` is a valid `size_t`, otherwise `0`.
   */
  bool parse_size_t(std::string_view s, size_t& value) const
  {
    if (s.empty())
    {
      return false;
    }

    const auto begin = s.data();
    const auto end = begin + s.size();
    const auto [ptr, error_code] = std::from_chars(begin, end, value);
    return error_code == std::errc{} && ptr == end;
  }

  /**
   * \brief
   * A valid floating point number is formed by \n
   * - an optional sign character (+ or -), \n
   * - followed by a sequence of digits, optionally containing a decimal-point character (.), \n
   * - optionally followed by an exponent part (an e or E character followed by an optional sign and a sequence of digits). \n
   * Return true if we are flagged to print the help message.
   * @param s String to validate.
   * @return `1` when `s` is a valid floating-point literal, otherwise `0`.
   */

  bool parse_double(std::string_view s, double& value) const
  {
    if (s.empty())
    {
      return false;
    }

    const auto begin = s.data();
    const auto end = begin + s.size();
    const auto [ptr, error_code] = std::from_chars(begin, end, value);
    return error_code == std::errc{} && ptr == end;
  }

  /**
   * @brief Parses the configured command-line options.
   */
  ParseResult parse()
  {
    for (const Option& option : options)
    {
      if (std::holds_alternative<std::reference_wrapper<bool>>(option.value))
      {
        std::get<std::reference_wrapper<bool>>(option.value).get() = false;
      }
    }

    option_check.assign(options.size(), 0);

    for (size_t i = 1; i < arguments.size();)
    {
      const std::string_view arg = arguments[i];

      if (arg == "-h" || arg == "--help")
      {
        // print help message
        return make_result(ErrorType::help);
      }

      size_t j = 0;
      for (; j < options.size(); j++)
      {
        if (options[j].matches(arg))
        {
          break;
        }
      }

      if (j >= options.size())
      {
        // unrecognized option
        return make_result(ErrorType::unrecognized_option, i);
      }

      const Option& option = options[j];

      if (option_check[j])
      {
        return make_result(ErrorType::duplicate_option, j);
      }
      option_check[j] = 1;

      if (!option.takes_argument())
      {
        std::get<std::reference_wrapper<bool>>(option.value).get() = true;
        i++;
        continue;
      }

      i++;
      if (i >= arguments.size())
      {
        // missing argument
        return make_result(ErrorType::missing_argument, j);
      }

      bool is_valid = true;
      const std::string_view value = arguments[i];
      if (auto* bool_ref = std::get_if<std::reference_wrapper<bool>>(&option.value))
      {
        bool_ref->get() = true;
      }
      else if (auto* int_ref = std::get_if<std::reference_wrapper<int>>(&option.value))
      {
        is_valid = parse_int(value, int_ref->get());
      }
      else if (auto* double_ref = std::get_if<std::reference_wrapper<double>>(&option.value))
      {
        is_valid = parse_double(value, double_ref->get());
      }
      else if (auto* size_ref = std::get_if<std::reference_wrapper<size_t>>(&option.value))
      {
        is_valid = parse_size_t(value, size_ref->get());
      }
      else if (auto* string_ref = std::get_if<std::reference_wrapper<std::string>>(&option.value))
      {
        string_ref->get() = std::string(value);
      }

      if (!is_valid)
      {
        return make_result(ErrorType::invalid_argument, i);
      }

      i++;
    }

    // check for missing required options
    for (size_t i = 0; i < options.size(); i++)
    {
      if (options[i].required && option_check[i] == 0)
      {
        return make_result(ErrorType::missing_required, i);
      }
    }

    return make_result(ErrorType::none);
  }

  /**
   * @brief Parses options, asserts success, and prints the final option values.
   * @param os Output stream used for reporting.
   */
  ParseResult parse_check(std::ostream& os = std::cout)
  {
    const ParseResult result = parse();
    boba_always_assert(result.ok(), "One or more parameters failed to be read.")
      print_options(os);
    return result;
  }

  /**
   * @brief Writes an option value to a stream.
   * @param opt Option descriptor.
   * @param os Output stream.
   */
  void write_value(const Option& opt, std::ostream& os) const
  {
    if (const auto* bool_ref = std::get_if<std::reference_wrapper<bool>>(&opt.value))
    {
      os << (bool_ref->get() ? std::string_view(opt.long_name) : std::string_view("disabled"));
    }
    else if (const auto* int_ref = std::get_if<std::reference_wrapper<int>>(&opt.value))
    {
      os << int_ref->get();
    }
    else if (const auto* double_ref = std::get_if<std::reference_wrapper<double>>(&opt.value))
    {
      os << double_ref->get();
    }
    else if (const auto* size_ref = std::get_if<std::reference_wrapper<size_t>>(&opt.value))
    {
      os << size_ref->get();
    }
    else if (const auto* string_ref = std::get_if<std::reference_wrapper<std::string>>(&opt.value))
    {
      os << string_ref->get();
    }
  }

  /**
   * @brief Prints the parsed option values.
   * @param os Output stream.
   */
  void print_options(std::ostream& os) const
  {
    static constexpr std::string_view indent = "   ";

    os << "Options used:\n";
    for (size_t j = 0; j < options.size(); j++)
    {
      os << indent;
      if (options[j].takes_argument())
      {
        os << options[j].long_name << " ";
        write_value(options[j], os);
      }
      else
      {
        os << options[j].current_bool_name();
      }
      os << '\n';
    }
  }

  /**
   * @brief Prints the current parse error.
   * @param os Output stream.
   */
  void print_error(std::ostream& os) const
  {
    static constexpr std::string_view line_sep = "";
    const ParseResult& result = last_result;

    os << line_sep;
    switch (result.error_type)
    {
    case ErrorType::unrecognized_option:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Unrecognized option: " << arguments[result.error_idx] << '\n'
         << line_sep;
      break;

    case ErrorType::missing_argument:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Missing argument for the last option: " << arguments.back()
         << '\n'
         << line_sep;
      break;

    case ErrorType::duplicate_option:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Option " << options[result.error_idx].long_name
         << " provided multiple times\n"
         << line_sep;
      break;

    case ErrorType::invalid_argument:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Wrong option format: " << arguments[result.error_idx - 1] << " "
         << arguments[result.error_idx] << '\n'
         << line_sep;
      break;

    case ErrorType::missing_required:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Missing required option: " << options[result.error_idx].long_name
         << '\n'
         << line_sep;
      break;
    case ErrorType::none:
    case ErrorType::help:
    default:
      boba_error("Unknown error");
    }
    os << std::endl;
  }
};

} // namespace boba
