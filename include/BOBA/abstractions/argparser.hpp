// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace boba
{

template <execution_space space, typename _data_t>
struct Vector;

/**
 * \brief
 * Command-line options argument parser. Inspired by some of the concept's mfem's OptionsParser \n
 * \details
 * The class is initialized with a list of argument strings, and new options are added with
 * the add_optional_argument and add_required_argument methods. Currently supports bool, int,
 * size_t, double, string, `std::vector<int>`, `std::vector<size_t>`,
 * `::boba::Array<size_t, K>`, and `::boba::Vector<::boba::host_space, size_t>` options.
 * List options accept either repeated tokens or comma-delimited literals such as `[4,8,16]`.
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
    std::monostate,
    std::reference_wrapper<bool>,
    std::reference_wrapper<int>,
    std::reference_wrapper<double>,
    std::reference_wrapper<size_t>,
    std::reference_wrapper<std::string>>;

public:
  struct ParseResult
  {
    static constexpr size_t npos = static_cast<size_t>(-1);

    ErrorType error_type = ErrorType::none;
    size_t error_idx = 0;
    size_t option_idx = npos;
    size_t arg_idx = npos;
    size_t arg_idx2 = npos;

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
    std::function<ParseResult(argparser&, size_t, size_t&)> custom_parse;
    std::function<void(std::ostream&)> custom_write;
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

    Option(
      std::string_view short_name_,
      std::string_view long_name_,
      std::string_view description_,
      std::function<ParseResult(argparser&, size_t, size_t&)> custom_parse_,
      std::function<void(std::ostream&)> custom_write_,
      bool req)
        : value(std::monostate{}),
          short_name(short_name_),
          long_name(long_name_),
          description(description_),
          custom_parse(std::move(custom_parse_)),
          custom_write(std::move(custom_write_)),
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
      return static_cast<bool>(custom_parse) || !std::holds_alternative<std::reference_wrapper<bool>>(value);
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

  void add_custom_argument(
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description,
    std::function<ParseResult(argparser&, size_t, size_t&)> custom_parse,
    std::function<void(std::ostream&)> custom_write,
    bool required)
  {
    options.push_back(
      Option(
        short_name,
        long_name,
        description,
        std::move(custom_parse),
        std::move(custom_write),
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

  ParseResult make_result(
    ErrorType new_error_type,
    size_t new_error_idx = 0,
    size_t new_option_idx = ParseResult::npos,
    size_t new_arg_idx = ParseResult::npos,
    size_t new_arg_idx2 = ParseResult::npos) noexcept
  {
    last_result.error_type = new_error_type;
    last_result.error_idx = new_error_idx;
    last_result.option_idx = new_option_idx;
    last_result.arg_idx = new_arg_idx;
    last_result.arg_idx2 = new_arg_idx2;
    return last_result;
  }

  [[nodiscard]]
  std::string option_value_placeholder(const Option& opt) const
  {
    if (!opt.takes_argument())
    {
      return {};
    }

    if (opt.custom_parse)
    {
      return " <value>";
    }

    if (std::holds_alternative<std::reference_wrapper<int>>(opt.value))
    {
      return " <int>";
    }
    if (std::holds_alternative<std::reference_wrapper<size_t>>(opt.value))
    {
      return " <size_t>";
    }
    if (std::holds_alternative<std::reference_wrapper<double>>(opt.value))
    {
      return " <double>";
    }
    if (std::holds_alternative<std::reference_wrapper<std::string>>(opt.value))
    {
      return " <string>";
    }

    return " <value>";
  }

  [[nodiscard]]
  std::string option_spelling(const Option& opt) const
  {
    std::string spelling;
    if (!opt.short_name.empty() && !opt.long_name.empty())
    {
      spelling = opt.short_name + ", " + opt.long_name;
    }
    else if (!opt.long_name.empty())
    {
      spelling = opt.long_name;
    }
    else
    {
      spelling = opt.short_name;
    }

    spelling += option_value_placeholder(opt);
    return spelling;
  }

  [[nodiscard]]
  bool is_help_option(std::string_view arg) const noexcept
  {
    return arg == "-h" || arg == "--help";
  }

  [[nodiscard]]
  size_t find_option_index(std::string_view arg) const noexcept
  {
    for (size_t i = 0; i < options.size(); ++i)
    {
      if (options[i].matches(arg))
      {
        return i;
      }
    }
    return options.size();
  }

  [[nodiscard]]
  bool is_registered_option(std::string_view arg) const noexcept
  {
    return is_help_option(arg) || find_option_index(arg) != options.size();
  }

  [[nodiscard]]
  static std::string_view trim_whitespace(std::string_view value) noexcept
  {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
    {
      ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
      --end;
    }

    return value.substr(begin, end - begin);
  }

  template <typename T, typename ParseElement>
  ParseResult parse_list_token(
    std::string_view token,
    size_t token_idx,
    std::vector<T>& values,
    ParseElement parse_element)
  {
    token = trim_whitespace(token);
    if (token.empty())
    {
      return make_result(ErrorType::invalid_argument, token_idx, ParseResult::npos, token_idx);
    }

    const bool starts_with_bracket = token.front() == '[';
    const bool ends_with_bracket = token.back() == ']';
    if (starts_with_bracket != ends_with_bracket)
    {
      return make_result(ErrorType::invalid_argument, token_idx, ParseResult::npos, token_idx);
    }

    if (starts_with_bracket)
    {
      token = trim_whitespace(token.substr(1, token.size() - 2));
      if (token.empty())
      {
        return make_result(ErrorType::invalid_argument, token_idx, ParseResult::npos, token_idx);
      }
    }

    size_t start = 0;
    while (start <= token.size())
    {
      const size_t comma = token.find(',', start);
      const size_t stop = (comma == std::string_view::npos) ? token.size() : comma;
      const std::string_view piece = trim_whitespace(token.substr(start, stop - start));
      if (piece.empty())
      {
        return make_result(ErrorType::invalid_argument, token_idx, ParseResult::npos, token_idx);
      }

      T parsed_value{};
      if (!parse_element(piece, parsed_value))
      {
        return make_result(ErrorType::invalid_argument, token_idx, ParseResult::npos, token_idx);
      }
      values.push_back(parsed_value);

      if (comma == std::string_view::npos)
      {
        break;
      }
      start = comma + 1;
    }

    return {};
  }

  template <typename T, typename ParseElement, typename AssignValues>
  ParseResult parse_list_argument(
    size_t option_idx,
    size_t& i,
    ParseElement parse_element,
    AssignValues assign_values,
    size_t expected_length = static_cast<size_t>(-1))
  {
    const size_t option_arg_idx = (i > 0) ? (i - 1) : ParseResult::npos;

    if (i >= arguments.size())
    {
      return make_result(ErrorType::missing_argument, option_idx, option_idx, option_arg_idx);
    }

    const size_t first_value_idx = i;
    if (is_registered_option(arguments[i]))
    {
      return make_result(ErrorType::missing_argument, option_idx, option_idx, option_arg_idx, i);
    }

    std::vector<T> parsed_values;
    for (; i < arguments.size(); ++i)
    {
      const std::string_view token = arguments[i];
      if (is_registered_option(token))
      {
        break;
      }

      const ParseResult token_result = parse_list_token(token, i, parsed_values, parse_element);
      if (!token_result.ok())
      {
        return make_result(token_result.error_type, token_result.error_idx, option_idx, token_result.error_idx, option_arg_idx);
      }
    }

    if (parsed_values.empty())
    {
      const size_t next_token_idx = (i < arguments.size()) ? i : ParseResult::npos;
      return make_result(ErrorType::missing_argument, option_idx, option_idx, option_arg_idx, next_token_idx);
    }

    if (expected_length != static_cast<size_t>(-1) && parsed_values.size() != expected_length)
    {
      return make_result(ErrorType::invalid_argument, first_value_idx, option_idx, first_value_idx, option_arg_idx);
    }

    assign_values(parsed_values);
    return {};
  }

  template <typename T>
  void write_vector_value(const std::vector<T>& values, std::ostream& os) const
  {
    os << '[';
    for (size_t i = 0; i < values.size(); ++i)
    {
      if (i > 0)
      {
        os << ", ";
      }
      os << values[i];
    }
    os << ']';
  }

  std::vector<std::string> arguments;
  std::vector<Option> options;
  std::vector<size_t> option_check;
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
   * Add an optional integer vector option and set `var` to receive the parsed values.
   * When present on the command line, the option consumes one or more integer tokens until
   * the next registered option is encountered.
   */
  void add_optional_argument(
    std::vector<int>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<int>(
        option_idx,
        i,
        [&parser](std::string_view value, int& parsed_value)
      {
        return parser.parse_int(value, parsed_value);
      },
        [&var](const std::vector<int>& values)
      {
        var = values;
      });
    },
      [&var](std::ostream& os)
    {
      os << '[';
      for (size_t i = 0; i < var.size(); ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << var[i];
      }
      os << ']';
    },
      false);
  }

  /**
   * \brief
   * Add a required integer vector option and set `var` to receive the parsed values.
   * When present on the command line, the option consumes one or more integer tokens until
   * the next registered option is encountered.
   */
  void add_required_argument(
    std::vector<int>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<int>(
        option_idx,
        i,
        [&parser](std::string_view value, int& parsed_value)
      {
        return parser.parse_int(value, parsed_value);
      },
        [&var](const std::vector<int>& values)
      {
        var = values;
      });
    },
      [&var](std::ostream& os)
    {
      os << '[';
      for (size_t i = 0; i < var.size(); ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << var[i];
      }
      os << ']';
    },
      true);
  }

  /**
   * \brief
   * Add an optional size_t vector option and set `var` to receive the parsed values.
   * When present on the command line, the option consumes one or more size_t tokens until
   * the next registered option is encountered.
   */
  void add_optional_argument(
    std::vector<size_t>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<size_t>(
        option_idx,
        i,
        [&parser](std::string_view value, size_t& parsed_value)
      {
        return parser.parse_size_t(value, parsed_value);
      },
        [&var](const std::vector<size_t>& values)
      {
        var = values;
      });
    },
      [&var](std::ostream& os)
    {
      os << '[';
      for (size_t i = 0; i < var.size(); ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << var[i];
      }
      os << ']';
    },
      false);
  }

  /**
   * \brief
   * Add a required size_t vector option and set `var` to receive the parsed values.
   * When present on the command line, the option consumes one or more size_t tokens until
   * the next registered option is encountered.
   */
  void add_required_argument(
    std::vector<size_t>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<size_t>(
        option_idx,
        i,
        [&parser](std::string_view value, size_t& parsed_value)
      {
        return parser.parse_size_t(value, parsed_value);
      },
        [&var](const std::vector<size_t>& values)
      {
        var = values;
      });
    },
      [&var](std::ostream& os)
    {
      os << '[';
      for (size_t i = 0; i < var.size(); ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << var[i];
      }
      os << ']';
    },
      true);
  }

  /**
   * \brief
   * Add an optional fixed-size size_t array option and set `var` to receive the parsed values.
   * The option accepts exactly `K` entries, either as repeated tokens or a list literal.
   */
  template <size_t K>
  void add_optional_argument(
    ::boba::Array<size_t, K>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<size_t>(
        option_idx,
        i,
        [&parser](std::string_view value, size_t& parsed_value)
      {
        return parser.parse_size_t(value, parsed_value);
      },
        [&var](const std::vector<size_t>& values)
      {
        for (size_t idx = 0; idx < K; ++idx)
        {
          var[idx] = values[idx];
        }
      },
        K);
    },
      [&var](std::ostream& os)
    {
      os << '[';
      for (size_t i = 0; i < K; ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << var[i];
      }
      os << ']';
    },
      false);
  }

  /**
   * \brief
   * Add a required fixed-size size_t array option and set `var` to receive the parsed values.
   * The option accepts exactly `K` entries, either as repeated tokens or a list literal.
   */
  template <size_t K>
  void add_required_argument(
    ::boba::Array<size_t, K>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<size_t>(
        option_idx,
        i,
        [&parser](std::string_view value, size_t& parsed_value)
      {
        return parser.parse_size_t(value, parsed_value);
      },
        [&var](const std::vector<size_t>& values)
      {
        for (size_t idx = 0; idx < K; ++idx)
        {
          var[idx] = values[idx];
        }
      },
        K);
    },
      [&var](std::ostream& os)
    {
      os << '[';
      for (size_t i = 0; i < K; ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << var[i];
      }
      os << ']';
    },
      true);
  }

  /**
   * \brief
   * Add an optional host Vector option and set `var` to receive the parsed values.
   * The option accepts one or more entries, either as repeated tokens or a list literal.
   */
  void add_optional_argument(
    ::boba::Vector<::boba::host_space, size_t>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<size_t>(
        option_idx,
        i,
        [&parser](std::string_view value, size_t& parsed_value)
      {
        return parser.parse_size_t(value, parsed_value);
      },
        [&var](const std::vector<size_t>& values)
      {
        var.resize(static_cast<::boba::index_t>(values.size()));
        auto view = var.view();
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
          view(static_cast<::boba::index_t>(idx)) = values[idx];
        }
      });
    },
      [&var](std::ostream& os)
    {
      auto view = var.const_view();
      os << '[';
      for (size_t i = 0; i < var.size(); ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << view(static_cast<::boba::index_t>(i));
      }
      os << ']';
    },
      false);
  }

  /**
   * \brief
   * Add a required host Vector option and set `var` to receive the parsed values.
   * The option accepts one or more entries, either as repeated tokens or a list literal.
   */
  void add_required_argument(
    ::boba::Vector<::boba::host_space, size_t>& var,
    std::string_view short_name,
    std::string_view long_name,
    std::string_view description)
  {
    add_custom_argument(
      short_name,
      long_name,
      description,
      [&var](argparser& parser, size_t option_idx, size_t& i)
    {
      return parser.parse_list_argument<size_t>(
        option_idx,
        i,
        [&parser](std::string_view value, size_t& parsed_value)
      {
        return parser.parse_size_t(value, parsed_value);
      },
        [&var](const std::vector<size_t>& values)
      {
        var.resize(static_cast<::boba::index_t>(values.size()));
        auto view = var.view();
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
          view(static_cast<::boba::index_t>(idx)) = values[idx];
        }
      });
    },
      [&var](std::ostream& os)
    {
      auto view = var.const_view();
      os << '[';
      for (size_t i = 0; i < var.size(); ++i)
      {
        if (i > 0)
        {
          os << ", ";
        }
        os << view(static_cast<::boba::index_t>(i));
      }
      os << ']';
    },
      true);
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

    option_check.assign(options.size(), ParseResult::npos);

    for (size_t i = 1; i < arguments.size();)
    {
      const std::string_view arg = arguments[i];

      if (is_help_option(arg))
      {
        // print help message
        return make_result(ErrorType::help);
      }

      const size_t j = find_option_index(arg);
      if (j >= options.size())
      {
        // unrecognized option
        return make_result(ErrorType::unrecognized_option, i, ParseResult::npos, i);
      }

      const Option& option = options[j];

      if (option_check[j] != ParseResult::npos)
      {
        return make_result(ErrorType::duplicate_option, j, j, i, option_check[j]);
      }
      option_check[j] = i;

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
        return make_result(ErrorType::missing_argument, j, j, i - 1);
      }

      if (option.custom_parse)
      {
        const ParseResult result = option.custom_parse(*this, j, i);
        if (!result.ok())
        {
          return result;
        }
        continue;
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
        return make_result(ErrorType::invalid_argument, i, j, i, i - 1);
      }

      i++;
    }

    // check for missing required options
    for (size_t i = 0; i < options.size(); i++)
    {
      if (options[i].required && option_check[i] == ParseResult::npos)
      {
        return make_result(ErrorType::missing_required, i, i);
      }
    }

    return make_result(ErrorType::none);
  }

  /**
   * @brief Parses options and prints the final option values.
   *
   * If `-h` / `--help` is provided, this prints a help message and exits with code 0.
   * If parsing fails, this prints a parse error and exits with code 1.
   * @param os Output stream used for reporting.
   */
  ParseResult parse_check(std::ostream& os = std::cout)
  {
    const ParseResult result = parse();

    if (result.help_requested())
    {
      print_help(os);
      std::exit(0);
    }

    if (!result.ok())
    {
      print_error(std::cerr);
      std::exit(1);
    }

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
    if (opt.custom_write)
    {
      opt.custom_write(os);
    }
    else if (const auto* bool_ref = std::get_if<std::reference_wrapper<bool>>(&opt.value))
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
   * @brief Prints a help message describing available options.
   * The help option itself (-h/--help) is always supported.
   * @param os Output stream.
   */
  void print_help(std::ostream& os) const
  {
    const std::string_view program_name =
      (!arguments.empty() && !arguments[0].empty()) ? std::string_view(arguments[0]) : std::string_view("program");

    os << "Usage:\n"
       << "  " << program_name << " [options]\n\n"
       << "Options:\n";

    // Compute alignment width for option spellings.
    size_t max_width = std::string_view("-h, --help").size();
    for (const auto& opt : options)
    {
      const size_t w = option_spelling(opt).size();
      if (w > max_width)
      {
        max_width = w;
      }
    }
    max_width += 2;

    os << "  " << std::left << std::setw(static_cast<int>(max_width)) << "-h, --help"
       << "Print this help message.\n";

    for (const auto& opt : options)
    {
      const std::string spelling = option_spelling(opt);
      os << "  " << std::left << std::setw(static_cast<int>(max_width)) << spelling
         << opt.description;

      if (opt.required)
      {
        os << " (required)";
      }

      const bool can_write_default =
        opt.custom_write || !std::holds_alternative<std::monostate>(opt.value);
      if (can_write_default)
      {
        os << " (default: ";
        write_value(opt, os);
        os << ")";
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
    const auto arg_at = [this](size_t idx) -> std::string_view
    {
      if (idx == ParseResult::npos || idx >= arguments.size())
      {
        return "<unknown>";
      }
      return arguments[idx];
    };

    os << line_sep;
    switch (result.error_type)
    {
    case ErrorType::unrecognized_option:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Unrecognized option: " << arg_at(result.error_idx);
      if (result.arg_idx != ParseResult::npos)
      {
        os << " (argv[" << result.arg_idx << "])";
      }
      os << '\n'
         << line_sep;
      break;

    case ErrorType::missing_argument:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Missing argument for the last option: " << arg_at(result.arg_idx);
      if (result.arg_idx != ParseResult::npos)
      {
        os << " (argv[" << result.arg_idx << "])";
      }
      if (result.arg_idx2 != ParseResult::npos)
      {
        os << " (next token: argv[" << result.arg_idx2 << "] " << arg_at(result.arg_idx2) << ")";
      }
      os << '\n'
         << line_sep;
      break;

    case ErrorType::duplicate_option:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Option " << options[result.error_idx].long_name
         << " provided multiple times";
      if (result.arg_idx2 != ParseResult::npos && result.arg_idx != ParseResult::npos)
      {
        os << " (first at argv[" << result.arg_idx2 << "], again at argv[" << result.arg_idx << "])";
      }
      os << '\n'
         << line_sep;
      break;

    case ErrorType::invalid_argument:
      os << '[' << error_code_name(result.error_type) << "] "
         << "Wrong option format: ";
      if (result.arg_idx2 != ParseResult::npos && result.arg_idx != ParseResult::npos)
      {
        os << arg_at(result.arg_idx2) << " " << arg_at(result.arg_idx);
        os << " (argv[" << result.arg_idx2 << ".." << result.arg_idx << "])";
      }
      else if (result.error_idx > 0 && result.error_idx < arguments.size())
      {
        os << arguments[result.error_idx - 1] << " " << arguments[result.error_idx];
        os << " (argv[" << (result.error_idx - 1) << ".." << result.error_idx << "])";
      }
      else if (result.error_idx < arguments.size())
      {
        os << arguments[result.error_idx] << " (argv[" << result.error_idx << "])";
      }
      else
      {
        os << "<unknown>";
      }

      if (result.option_idx != ParseResult::npos && result.option_idx < options.size())
      {
        os << " (expected" << option_value_placeholder(options[result.option_idx]) << ")";
      }

      os << '\n'
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
