// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <sstream>

namespace boba
{

/**
 * \brief Prints a tensor in the format selected by `BOBA_PRINT_STYLE`.
 * \param input Tensor to print.
 * \param indent Indentation level applied to each printed line.
 * \param stream Stream receiving the output.
 * \return \p stream after writing the tensor contents.
 */
template <size_t dimension, execution_space space, typename data_t>
std::ostream& print(const Tensor<dimension, space, data_t>& input, size_t indent = 1_z, std::ostream& stream = std::cout)
{
  using real_data_t = real_type_t<data_t>;

  if (input.empty())
  {
    stream << write_indent(indent) << input.name() << " is empty: " << input.sizes() << std::endl;
    return stream;
  }

  if constexpr (space != execution_space::CPU)
  {
    Tensor<dimension, execution_space::CPU, data_t> host_data = input;
    return print(host_data, indent, stream);
  }
  auto tensor_view = input.const_view();

  // default: boba::env_match("BOBA_PRINT_STYLE", "MATLAB")
  std::string lb = "(";
  std::string rb = ")";
  std::string post = ";";
  size_t offset = 1;
  bool print_multiindex = true;
  bool is_matlab = true;

  if (boba::env_match("BOBA_PRINT_STYLE", "CPP"))
  {
    lb = "[";
    rb = "]";
    post = ";";
    offset = 0;
    print_multiindex = false;
    is_matlab = false;
  }
  if (boba::env_match("BOBA_PRINT_STYLE", "BOBA"))
  {
    lb = "({";
    rb = "})";
    post = ";";
    offset = 0;
    print_multiindex = true;
    is_matlab = false;
  }
  if (boba::env_match("BOBA_PRINT_STYLE", "PYTHON"))
  {
    lb = "[";
    rb = "]";
    post = "";
    offset = 0;
    print_multiindex = true;
    is_matlab = false;
  }

  auto indentation = write_indent(indent);
  auto local_name = input.name();
  auto loop_size = tensor_view.size();
  auto local_dimension = tensor_view.get_dimension();
  const auto default_precision{stream.precision()};
  stream << std::setprecision(15);

  auto write = [&](size_t I)
  {
    auto value = tensor_view(I);
    bool is_last_element = I == loop_size - 1;
    bool is_not_small = not(::boba::abs(value) < real_data_t(1000) * boba::epsilon<real_data_t>());
    if (is_not_small or is_last_element)
    {
      stream << indentation << local_name << lb;
      if (print_multiindex)
      {
        //
        // Multiindex
        //
        auto indices = tensor_view.multiindex(I);
        stream << offset + indices[0];
        for (size_t d = 1; d < local_dimension; d++)
        {
          stream << ", " << offset + indices[d];
        }
      }
      else
      {
        //
        // Long index
        //
        stream << offset + I;
      }
      stream << rb + " = " << value << post << "\n";
    }
  };

  if ((loop_size > 100) and is_matlab)
  {
    // If tensor is "large", print last element first is using matlab
    // in MATLAB this will allocate the correct sized object upfront
    write(loop_size - 1);
    for (size_t I = 0; I < loop_size - 1; I++)
    {
      write(I);
    }
  }
  else
  {
    // Just print all elements in order
    for (size_t I = 0; I < loop_size; I++)
    {
      write(I);
    }
  }

  stream << std::setprecision(default_precision); // restore defaults
  return stream;
}

/**
 * \brief Dumps a tensor to a text file together with its indices.
 * \param input Tensor to write.
 * \param filename Output filename. Uses the tensor name when empty.
 * \param has_header Whether to write a header line containing the tensor extents.
 * \param is_zero_based Whether indices in the file should be zero-based.
 */
template <size_t dimension, execution_space space, typename data_t>
void write_to_file(
  const Tensor<dimension, space, data_t>& input,
  std::string_view filename = "",
  bool has_header = true,
  bool is_zero_based = true)
{
  BOBA_CALI_OBJECT_MARK

  auto output_filename = filename;
  if (filename == "")
  {
    output_filename = input.name();
  }
  if constexpr (space != execution_space::CPU)
  {
    Tensor<dimension, execution_space::CPU, data_t> host_data = input;
    boba::write_to_file(host_data, output_filename, has_header, is_zero_based);
  }
  else
  {

    std::ofstream file;
    std::string print_filename = "";

    index_t offset = is_zero_based ? 0 : 1;

    if (output_filename.empty())
    {
      print_filename = input.m_name;
    }
    else
    {
      print_filename = std::string(output_filename);
    }
    file.open(print_filename.c_str());

    boba_always_assert(file.good(),
                       "Failed to open file " + print_filename +
                         "\n If the filename includes a path, make sure that path exists!");

    auto tensor_view = input.const_view();

    //
    // Header is the sizes of the tensor
    // sizes[0] sizes[1] sizes[2] ... sizes[dimension-1] 0.0
    //
    if (has_header)
    {
      for (index_t d = 0; d < index_t(dimension); d++)
      {
        file << input.sizes(d) << " ";
      }
      file << 0.0;
    }

    //
    // All other lines are the multiindex and the value
    // mid[0] mid[1] mid[2] ... mid[dimension-1] value
    //
    for (index_t i = 0; i < input.size(); i++)
    {
      auto mi = tensor_view.multiindex(i);
      file << "\n";
      for (index_t d = 0; d < index_t(dimension); d++)
      {
        file << mi[d] + offset << " ";
      }
      // From https://stackoverflow.com/questions/4643641/best-way-to-output-a-full-precision-double-into-a-text-file
      auto precision = std::numeric_limits<real_type_t<data_t>>::max_digits10;

      file << std::setprecision(precision) << input.m_data[i];
    }
    file << "\n";
  }
}

/**
 * \brief Reads one tensor entry from a text line.
 * \param line Input line containing indices followed by a value.
 * \param is_zero_based Whether the stored indices are zero-based.
 * \return Pair of tensor multi-index and value parsed from \p line.
 *
 * \warning Data not overwritten by file contents should already be initialized before calling this
 * helper, for example when the serialized tensor is sparse.
 */
template <size_t mid_length, typename data_t>
[[nodiscard]]
std::pair<::boba::Array<index_t, mid_length>, data_t> read_line_to_array(
  std::string& line,
  bool is_zero_based)
{
  std::replace(line.begin(), line.end(), '\t', ' ');
  std::istringstream linestream(line);

  auto mid = filled_array<mid_length>(0_z);
  for (size_t d = 0; d < mid_length; ++d)
  {
    index_t id;
    linestream >> id;
    mid[d] = is_zero_based ? id : id - 1_z;
  }

  data_t value;
  linestream >> value;

  return std::make_pair(mid, value);
}

/**
 * \brief Reads tensor extents from the first line of a BoBa text file.
 * \param filename Input filename.
 * \return Tensor extents parsed from the header line.
 */
template <size_t dimension, typename data_t>
[[nodiscard]]
::boba::Array<index_t, dimension> read_sizes_from_file(std::string_view filename)
{
  BOBA_CALI_OBJECT_MARK

  auto filename_ext = std::string(filename);

  std::filebuf file_buffer;
  checkpoint_objects();
  bool file_buffer_check = file_buffer.open(filename_ext, std::ios::in);
  boba_always_assert(file_buffer_check, "File buffer failure, file = " + filename_ext);

  checkpoint_objects();
  std::istream file(&file_buffer);
  boba_always_assert(file.good(), "Failed to read file into stream.");
  std::string line;
  std::getline(file, line);
  auto [mid, value] = read_line_to_array<dimension, data_t>(line, true);
  return mid;
}

/**
 * \brief Reads a tensor from a text file written by write_to_file.
 * \param input Tensor receiving the file contents.
 * \param filename Input filename. Uses the tensor name when empty.
 * \param has_header Whether the file begins with a header line containing tensor extents.
 * \param is_zero_based Whether indices stored in the file are zero-based.
 */
template <size_t dimension, execution_space space, typename data_t>
void read_from_file(
  Tensor<dimension, space, data_t>& input,
  std::string_view filename = "",
  bool has_header = true,
  bool is_zero_based = true)
{
  BOBA_CALI_OBJECT_MARK

  auto input_filename = filename;
  if (filename == "")
  {
    input_filename = input.name();
  }
  if constexpr (space != execution_space::CPU)
  {
    Tensor<dimension, execution_space::CPU, data_t> host_data;
    if (not(has_header))
    {
      host_data.resize(input.sizes());
    }
    boba::read_from_file(host_data, input_filename, has_header, is_zero_based);
    input = host_data;
  }
  else
  {
    checkpoint_objects();
    auto filename_ext = std::string(input_filename);

    if (has_header)
    {
      auto file_sizes = read_sizes_from_file<dimension, data_t>(filename_ext);
      input.resize(file_sizes);
    }

    auto tensor_view = input.view();

    // Assert that the view has a valid pointer if size > 0
    boba_always_assert(
      (input.size() == 0) || tensor_view.data() != nullptr,
      "tensor_view has null data pointer even though tensor.size() > 0 in read_from_file");

    std::filebuf file_buffer;
    checkpoint_objects();
    bool file_buffer_check = file_buffer.open(filename_ext, std::ios::in);
    boba_always_assert(file_buffer_check, "File buffer failure, file = " + filename_ext);

    if (file_buffer_check)
    {
      checkpoint_objects();
      std::istream file(&file_buffer);
      boba_always_assert(file.good(), "Failed to read file into stream.");
      size_t line_number = 0;
      for (std::string line; std::getline(file, line);)
      {
        std::istringstream linestream(line);
        if ((line_number > 0) or not(has_header))
        {
          auto [mid, value] = read_line_to_array<dimension, data_t>(line, is_zero_based);
          tensor_view(mid) = value;
        }
        line_number++;
      }
      checkpoint_objects();
    }
  }
}

/**
 * \brief
 * Writes raw data to file but without sizing or index information.
 * Similar to numpy's savetxt function.
 */

template <size_t dimension, execution_space space, typename data_t>
void savetxt(const Tensor<dimension, space, data_t>& input, std::string_view _filename = "")
{
  BOBA_CALI_OBJECT_MARK

  std::string filename(_filename);
  if (_filename == "")
  {
    std::string tensor_name(input.name());
    filename = tensor_name + "_txt";
  }
  if constexpr (space != execution_space::CPU)
  {
    Tensor<dimension, execution_space::CPU, data_t> host_data = input;
    boba::savetxt(host_data, filename);
  }
  else
  {
    std::ofstream file;
    std::string print_filename = "";

    if (filename.empty())
    {
      print_filename = input.m_name;
    }
    else
    {
      print_filename = std::string(filename);
    }
    file.open(print_filename.c_str());

    boba_always_assert(file.good(),
                       "Failed to open file " + print_filename +
                         "\n If the filename includes a path, make sure that path exists!");

    auto tensor_view = input.const_view();

    auto precision = std::numeric_limits<real_type_t<data_t>>::max_digits10;
    file << std::setprecision(precision);
    for (index_t i = 0; i < tensor_view.size(); i++)
    {
      data_t x = tensor_view(i);
      file << x << "\n";
    }
  }
}

/**
 * \brief
 * Writes raw data to file but without sizing or index information.
 * Similar to numpy's savetxt function.
 */

template <size_t dimension, execution_space space, typename data_t>
void loadtxt(Tensor<dimension, space, data_t>& input, std::string_view _filename = "")
{
  BOBA_CALI_OBJECT_MARK

  std::string filename(_filename);
  if (_filename == "")
  {
    std::string tensor_name(input.name());
    filename = tensor_name + "_txt";
  }
  if constexpr (space != execution_space::CPU)
  {
    Tensor<dimension, execution_space::CPU, data_t> host_data = input;
    boba::loadtxt(host_data, filename);
    input = host_data;
  }
  else
  {
    checkpoint_objects();
    auto filename_ext = std::string(filename);

    std::filebuf file_buffer;
    checkpoint_objects();
    bool file_buffer_check = file_buffer.open(filename_ext, std::ios::in);
    boba_always_assert(file_buffer_check, "File buffer failure, file = " + filename_ext);

    auto tensor_view = input.view();

    if (file_buffer_check)
    {
      checkpoint_objects();
      std::istream file(&file_buffer);
      boba_always_assert(file.good(), "Failed to read file.");
      size_t line_number = 0;
      for (std::string line; std::getline(file, line);)
      {
        std::istringstream linestream(line);
        std::string item;
        std::getline(linestream, item, ' ');
        data_t value = std::stod(item);
        boba_always_assert_lt(line_number, static_cast<size_t>(input.size()), "Inconsistent file and tensor size.");
        tensor_view(line_number) = value;
        line_number++;
      }
      checkpoint_objects();
      boba_always_assert_equal(static_cast<size_t>(input.size()), line_number, "Tensor description inconsistent with file data.");
    }
  }
}

template <size_t dimension, execution_space space, typename data_t>
std::ostream& operator<<(std::ostream& os, const Tensor<dimension, space, data_t>& rhs)
{
  return print(rhs, 0_z, os);
}

template <size_t dimension, execution_space space, typename data_t>
void boba::Tensor<dimension, space, data_t>::print(size_t indent) const
{
  boba::print(*this, indent);
}

template <execution_space space, typename data_t, std::size_t dimension>
void print(const TensorView<DefaultAccessor<data_t>, dimension>& view)
{
  if constexpr (space == ::boba::execution_space::CPU)
  {

    loop<space, 1>(view.size(),
                   [=] __boba_host_device__(size_t i)
    {
      auto mid = view.multiindex(i);
      std::cout << "view(" << mid[0] + 1;
      for (size_t d = 1; d < dimension; d++)
      {
        std::cout << ", " << mid[d] + 1;
      }
      std::cout << " ) = " << view(i) << ";\n";
    });
  }
}

/**
 * \brief
 * Dumps a tensor to a MATLAB mat file.
 */

template <size_t dimension, execution_space space, typename data_t>
void write_to_mat_file(
  const Tensor<dimension, space, data_t>& input,
  std::string _filename = "")
{
  BOBA_CALI_OBJECT_MARK

  auto filename = _filename;
  if (_filename == "")
  {
    filename = input.name();
  }

  detail::MatFile matlab_file(filename, "w");
  matlab_file.write_array(filename, input);
}

/**
 * \brief
 * Reads a tensor to a MATLAB mat file.
 */
template <size_t dimension, execution_space space, typename data_t>
void read_from_mat_file(
  Tensor<dimension, space, data_t>& input,
  std::string _filename = "")
{
  BOBA_CALI_OBJECT_MARK

  auto filename = _filename;
  if (_filename == "")
  {
    filename = input.name();
  }

  detail::MatFile matlab_file(filename, "r");
  matlab_file.read_array(filename, input);
}

template <size_t dimension, execution_space space, typename data_t>
void write_to_hdf5_file(
  const Tensor<dimension, space, data_t>& input,
  std::string filename,
  std::string object_name = "")
{
  BOBA_CALI_OBJECT_MARK

  if (object_name.empty())
  {
    object_name = input.name();
  }

  detail::HDF5File h5_file(filename, "w");
  h5_file.write_array(object_name, input);
}

/**
 * \brief
 * Reads a tensor from an hdf5 file
 * @param[in] _filename hdf5 file
 * @param[in] in_object_name variable name in the hdf5 file
 */
template <size_t dimension, execution_space space, typename data_t>
void read_from_hdf5_file(
  Tensor<dimension, space, data_t>& input,
  std::string filename,
  std::string object_name = "")
{
  BOBA_CALI_OBJECT_MARK

  if (object_name.empty())
  {
    object_name = input.name();
  }

  detail::HDF5File h5_file(filename, "r");
  h5_file.read_array(object_name, input);
}

} // namespace boba
