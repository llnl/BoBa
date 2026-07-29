// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#ifdef BOBA_MATLAB
#include <mat.h>
#endif

// -----------------------------------------------------
// MatFile interface
// -----------------------------------------------------
namespace boba
{
namespace detail
{
#ifdef BOBA_MATLAB

struct MatFile
{
  /**
   * @brief Opens a MATLAB MAT-file for reading or writing.
   * @param filename MAT-file stem without the `.mat` suffix.
   * @param permissions Access mode string: `r` or `w`.
   */
  MatFile(std::string filename, std::string permissions)
  {
    // open MAT-file
    m_filename = filename;
    m_permissions = permissions;
    pmat = matOpen((m_filename + ".mat").c_str(), permissions.c_str());
    boba_always_assert(pmat != nullptr, "Failed to open file " + filename);
  }

  /**
   * @brief destructor
   */
  ~MatFile()
  {
    matClose(pmat);
  }

  /**
   * @brief Reads a tensor array into a non-host tensor by staging through host memory.
   */
  template <size_t dimension, execution_space space, typename data_t>
    requires(space != host_space)
  void read_array(std::string array_name, ::boba::Tensor<dimension, space, data_t>& destination)
  {
    ::boba::Tensor<dimension, host_space, data_t> host_destination;
    this->read_array(array_name, host_destination);
    destination = host_destination;
  }

  /**
   * @brief Reads a tensor array from a MAT-file into host memory.
   */
  template <size_t dimension, typename data_t>
  void read_array(std::string array_name, ::boba::Tensor<dimension, host_space, data_t>& destination)
  {
    boba_always_assert_equal(m_permissions, std::string("r"), "Incorrect permissions for opened file.");

    // extract the specified variable
    mxArray* source = matGetVariable(pmat, array_name.c_str());
    boba_always_assert(source != nullptr, "Failed find variable " + array_name + " in file" + m_filename);

    // unpack matlab array into destination
    unpack_tensor_from_matlab_array(source, destination);

    // cleanup
    mxDestroyArray(source);
  }

  /**
   * @brief Writes a non-host tensor array by staging through host memory.
   */
  template <size_t dimension, execution_space space, typename data_t>
    requires(space != host_space)
  void write_array(std::string array_name, const ::boba::Tensor<dimension, space, data_t>& source)
  {
    ::boba::Tensor<dimension, host_space, data_t> host_source = source;
    this->write_array(array_name, host_source);
  }

  /**
   * @brief Writes a host tensor array to a MAT-file.
   */
  template <size_t dimension, typename data_t>
  void write_array(std::string array_name, const ::boba::Tensor<dimension, host_space, data_t>& source)
  {
    boba_always_assert_equal(m_permissions, std::string("w"), "Incorrect permissions for opened file.");

    // pack source into matlab array
    mxArray* array = nullptr;
    pack_tensor_into_matlab_array(source, array);

    // Write the array to the MAT-file
    auto error_code = matPutVariable(pmat, array_name.c_str(), array);
    if (error_code != 0)
    {
      matClose(pmat);
      mxDestroyArray(array);
      boba_error("Error writing variable to MAT-file!");
    }

    // Clean up
    mxDestroyArray(array);
  }

  /**
   * @brief Reads a cell array into a fixed-size non-host tensor array.
   */
  template <size_t dimension_out, size_t dimension_in, execution_space space, typename data_t>
    requires(space != host_space)
  void read_cell_array(const std::string& array_name, ::boba::Array<::boba::Tensor<dimension_in, space, data_t>, dimension_out>& destination)
  {
    ::boba::Array<::boba::Tensor<dimension_in, host_space, data_t>, dimension_out> host_destination;
    read_cell_array(array_name, host_destination);
    destination = host_destination;
  }

  /**
   * @brief Reads a cell array into a fixed-size host tensor array.
   */
  template <size_t dimension_out, size_t dimension_in, typename data_t>
  void read_cell_array(const std::string& array_name, ::boba::Array<::boba::Tensor<dimension_in, host_space, data_t>, dimension_out>& destination)
  {
    boba_always_assert_equal(m_permissions, std::string("r"), "Incorrect permissions for opened file.");

    // extract the specified variable
    mxArray* cell = matGetVariable(pmat, array_name.c_str());
    boba_always_assert(cell != nullptr, "Failed find variable " + array_name + " in file" + m_filename);

    boba_always_assert(mxIsCell(cell), "This function is only supported for cell arrays");
    boba_always_assert(not(mxIsEmpty(cell)), "Cell array appears empty.");

    // determine size of cell array; we expect it to be one dimensional
    auto cell_ndim = static_cast<size_t>(mxGetNumberOfDimensions(cell));
    const mwSize* cell_dims = mxGetDimensions(cell);

    // note: matlab num_dim is always at least 2; if num_dim is 2, we ensure size of the either the first or second dimension is one
    boba_always_assert((cell_ndim == 2_z) && (static_cast<size_t>(cell_dims[0]) == 1_z || static_cast<size_t>(cell_dims[1]) == 1_z), "Cell array must be one dimensional");

    auto cell_numel = static_cast<size_t>(cell_dims[0] * cell_dims[1]);
    boba_always_assert_equal(cell_numel, dimension_out, "Length of cell array does not match outer dimension");

    // retrieve cells one by one
    for (std::size_t cell_index = 0; cell_index < cell_numel; ++cell_index)
    {
      mxArray* array = mxGetCell(cell, static_cast<mwIndex>(cell_index));
      unpack_tensor_from_matlab_array(array, destination[cell_index]);
    }

    // cleanup
    mxDestroyArray(cell);
  }

  /**
   * @brief Writes a fixed-size non-host tensor array as a MAT-file cell array.
   */
  template <size_t dimension_out, size_t dimension_in, execution_space space, typename data_t>
    requires(space != host_space)
  void write_cell_array(const std::string& array_name, const ::boba::Array<::boba::Tensor<dimension_in, space, data_t>, dimension_out>& source)
  {
    ::boba::Array<::boba::Tensor<dimension_in, host_space, data_t>, dimension_out> host_source = source;
    write_cell_array(array_name, host_source);
  }

  /**
   * @brief Writes a fixed-size host tensor array as a MAT-file cell array.
   */
  template <size_t dimension_out, size_t dimension_in, typename data_t>
  void write_cell_array(const std::string& array_name, const ::boba::Array<::boba::Tensor<dimension_in, host_space, data_t>, dimension_out>& source)
  {
    boba_always_assert_equal(m_permissions, std::string("w"), "Incorrect permissions for opened file.");

    // define the cell array
    const mwSize cell_dims[2] = {static_cast<mwSize>(dimension_out), static_cast<mwSize>(1)};
    mxArray* cell = mxCreateCellArray(static_cast<mwSize>(2), cell_dims);

    // fill in the array elements
    for (std::size_t cell_index = 0; cell_index < dimension_out; ++cell_index)
    {
      mxArray* array = nullptr;
      pack_tensor_into_matlab_array(source[cell_index], array);
      mxSetCell(cell, static_cast<mwIndex>(cell_index), array);
    }

    // Write cell array to the MAT file
    auto error_code = matPutVariable(pmat, array_name.c_str(), cell);
    if (error_code != 0)
    {
      matClose(pmat);
      mxDestroyArray(cell);
      boba_error("Error writing variable to MAT-file");
    }

    // Clean up
    mxDestroyArray(cell);
  }

  /**
   * @brief Reads a cell array into a non-host tensor vector.
   */
  template <size_t dimension, execution_space space, typename data_t>
    requires(space != host_space)
  void read_cell_array(const std::string& array_name, std::vector<::boba::Tensor<dimension, space, data_t>>& destination)
  {
    std::vector<::boba::Tensor<dimension, host_space, data_t>> host_destination;
    read_cell_array(array_name, host_destination);
    destination = host_destination;
  }

  /**
   * @brief Reads a cell array into a host tensor vector.
   */
  template <size_t dimension, typename data_t>
  void read_cell_array(const std::string& array_name, std::vector<::boba::Tensor<dimension, host_space, data_t>>& destination)
  {
    boba_always_assert_equal(m_permissions, std::string("r"), "Incorrect permissions for opened file.");

    // extract the specified variable
    mxArray* cell = matGetVariable(pmat, array_name.c_str());
    boba_always_assert(cell != nullptr, "Failed find variable " + array_name + " in file" + m_filename);

    boba_always_assert(mxIsCell(cell), "This function is only supported for cell arrays");
    boba_always_assert(not(mxIsEmpty(cell)), "Cell array appears empty.");

    // determine size of cell array; we expect it to be one dimensional
    auto cell_num_dim = static_cast<size_t>(mxGetNumberOfDimensions(cell));
    const mwSize* cell_size_mat = mxGetDimensions(cell);

    // note: matlab num_dim is always at least 2; if num_dim is 2, we ensure size of the either the first or second dimension is one
    boba_always_assert((cell_num_dim == 2_z) && (static_cast<size_t>(cell_size_mat[0]) == 1_z || static_cast<size_t>(cell_size_mat[1]) == 1_z), "Cell array must be one dimensional");

    auto cell_length = static_cast<size_t>(cell_size_mat[0] * cell_size_mat[1]);
    destination.resize(cell_length);

    // retrieve cells one by one
    for (std::size_t cell_index = 0; cell_index < cell_length; ++cell_index)
    {
      mxArray* arr = mxGetCell(cell, static_cast<mwIndex>(cell_index));
      unpack_tensor_from_matlab_array(arr, destination[cell_index]);
    }

    // cleanup
    mxDestroyArray(cell);
  }

  /**
   * @brief Writes a non-host tensor vector as a MAT-file cell array.
   */
  template <size_t dimension, execution_space space, typename data_t>
    requires(space != host_space)
  void write_cell_array(const std::string& array_name, const std::vector<::boba::Tensor<dimension, space, data_t>>& source)
  {
    std::vector<::boba::Tensor<dimension, host_space, data_t>> host_source = source;
    write_cell_array(array_name, host_source);
  }

  /**
   * @brief Writes a host tensor vector as a MAT-file cell array.
   */
  template <size_t dimension, typename data_t>
  void write_cell_array(const std::string& array_name, const std::vector<::boba::Tensor<dimension, host_space, data_t>>& source)
  {
    boba_always_assert_equal(m_permissions, std::string("w"), "Incorrect permissions for opened file.");

    // define the cell array
    const size_t cell_numel = source.size();
    const mwSize cell_dims[2] = {static_cast<mwSize>(cell_numel), static_cast<mwSize>(1)};
    mxArray* cell = mxCreateCellArray(static_cast<mwSize>(2), cell_dims);

    // fill in the array elements
    for (std::size_t cell_index = 0; cell_index < cell_numel; ++cell_index)
    {
      mxArray* array = nullptr;
      pack_tensor_into_matlab_array(source[cell_index], array);
      mxSetCell(cell, static_cast<mwIndex>(cell_index), array);
    }

    // Write cell array to the MAT file
    auto error_code = matPutVariable(pmat, array_name.c_str(), cell);
    if (error_code != 0)
    {
      matClose(pmat);
      mxDestroyArray(cell);
      boba_error("Error writing variable to MAT-file");
    }

    // Clean up
    mxDestroyArray(cell);
  }

private:
  /**
   * @brief Unpacks a MATLAB numeric array into a host tensor.
   */
  template <size_t dimension, typename data_t>
  void unpack_tensor_from_matlab_array(const mxArray* matlab_array, ::boba::Tensor<dimension, host_space, data_t>& boba_tensor)
  {
    if constexpr (std::is_same_v<data_t, float>)
    {
      boba_always_assert(mxIsSingle(matlab_array) and not(mxIsComplex(matlab_array)), "Mistmatch in mat file data and tensor data types");
    }
    else if constexpr (std::is_same_v<data_t, double>)
    {
      boba_always_assert(mxIsDouble(matlab_array) and not(mxIsComplex(matlab_array)), "Mistmatch in mat file data and tensor data types");
    }
    else if constexpr (std::is_same_v<data_t, int>)
    {
      boba_always_assert(mxIsInt32(matlab_array) and not(mxIsComplex(matlab_array)), "Mistmatch in mat file data and tensor data types");
    }
    else if constexpr (std::is_same_v<data_t, size_t>)
    {
      boba_always_assert(mxIsUint64(matlab_array) and not(mxIsComplex(matlab_array)), "Mistmatch in mat file data and tensor data types");
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<float>>)
    {
      boba_always_assert(mxIsSingle(matlab_array) and mxIsComplex(matlab_array), "Mistmatch in mat file data and tensor data types");
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<double>>)
    {
      boba_always_assert(mxIsDouble(matlab_array) and mxIsComplex(matlab_array), "Mistmatch in mat file data and tensor data types");
    }
    else
    {
      boba_error("Unknown data type");
    }

    boba_always_assert(not(mxIsEmpty(matlab_array)), "MATLAB array appears empty.");

    // number of dimensions
    auto matlab_array_ndim = static_cast<size_t>(mxGetNumberOfDimensions(matlab_array));
    boba_always_assert_le(matlab_array_ndim, dimension, "Variable and tensor dimenstionality mismatch."); // MATLAB drops singleton dimensions at the end

    // data pointer to matlab array
    data_t* matlab_array_data;
    if constexpr (std::is_same_v<data_t, float>)
    {
      matlab_array_data = reinterpret_cast<data_t*>(mxGetSingles(matlab_array));
    }
    else if constexpr (std::is_same_v<data_t, double>)
    {
      matlab_array_data = reinterpret_cast<data_t*>(mxGetDoubles(matlab_array));
    }
    else if constexpr (std::is_same_v<data_t, int>)
    {
      matlab_array_data = reinterpret_cast<data_t*>(mxGetInt32s(matlab_array));
    }
    else if constexpr (std::is_same_v<data_t, size_t>)
    {
      matlab_array_data = reinterpret_cast<data_t*>(mxGetUint64s(matlab_array));
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<float>>)
    {
      matlab_array_data = reinterpret_cast<data_t*>(mxGetComplexSingles(matlab_array));
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<double>>)
    {
      matlab_array_data = reinterpret_cast<data_t*>(mxGetComplexDoubles(matlab_array));
    }
    boba_always_assert(matlab_array_data != nullptr, "Failed to get pointer.");

    // mode sizes
    const mwSize* matlab_array_size = mxGetDimensions(matlab_array);
    auto boba_tensor_sizes = ::boba::filled_array<dimension>(1_z);
    for (size_t d = 0; d < matlab_array_ndim; d++)
    {
      boba_tensor_sizes[d] = matlab_array_size[d];
    }

    // boba tensor memory allocation
    boba_tensor.resize(boba_tensor_sizes);

    // copy
    boba::detail::memcpy<boba::host_space, boba::host_space>(
      boba_tensor.data(), matlab_array_data, static_cast<size_t>(boba_tensor.size()));
  }

  // warnings:
  // - matlab_array is overwritten; old data may be lost.
  // - after the function exits, matlab_array may need to be freed.
  /**
   * @brief Packs a host tensor into a newly allocated MATLAB numeric array.
   */
  template <size_t dimension, typename data_t>
  void pack_tensor_into_matlab_array(const ::boba::Tensor<dimension, host_space, data_t>& boba_tensor, mxArray*& matlab_array)
  {
    // mode sizes
    constexpr mwSize ndim = dimension;
    mwSize matlab_array_size[ndim];
    for (mwSize d = 0; d < ndim; d++)
    {
      matlab_array_size[d] = boba_tensor.sizes(static_cast<size_t>(d));
    }

    // allocate memory for MATLAB array
    if constexpr (std::is_same_v<data_t, float>)
    {
      matlab_array = mxCreateNumericArray(ndim, matlab_array_size, mxSINGLE_CLASS, mxREAL);
    }
    else if constexpr (std::is_same_v<data_t, double>)
    {
      matlab_array = mxCreateNumericArray(ndim, matlab_array_size, mxDOUBLE_CLASS, mxREAL);
    }
    else if constexpr (std::is_same_v<data_t, int>)
    {
      matlab_array = mxCreateNumericArray(ndim, matlab_array_size, mxINT32_CLASS, mxREAL);
    }
    else if constexpr (std::is_same_v<data_t, size_t>)
    {
      matlab_array = mxCreateNumericArray(ndim, matlab_array_size, mxUINT64_CLASS, mxREAL);
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<float>>)
    {
      matlab_array = mxCreateNumericArray(ndim, matlab_array_size, mxSINGLE_CLASS, mxCOMPLEX);
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<double>>)
    {
      matlab_array = mxCreateNumericArray(ndim, matlab_array_size, mxDOUBLE_CLASS, mxCOMPLEX);
    }
    else
    {
      boba_error("Unknown data type");
    }

    // copy
    data_t* matlab_data = static_cast<data_t*>(mxGetData(matlab_array));
    boba::detail::memcpy<boba::host_space, boba::host_space>(
      matlab_data, boba_tensor.data(), static_cast<size_t>(boba_tensor.size()));
  }

  std::string m_filename;
  std::string m_permissions;
  MATFile* pmat = nullptr;
};

#else

struct MatFile
{
  /**
   * @brief Opens a MAT-file placeholder when MATLAB support is disabled.
   * @param filename MAT-file stem without the `.mat` suffix.
   * @param permissions Access mode string.
   */
  MatFile(std::string filename, std::string permissions)
  {
    ignore(filename);
    ignore(permissions);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }

  /**
   * @brief destructor
   */
  ~MatFile()
  {
  }

  /**
   * @brief Reports that MAT-file tensor reads are unavailable.
   */
  template <size_t dimension, execution_space space, typename data_t>
  void read_array(std::string array_name, ::boba::Tensor<dimension, space, data_t>& destination)
  {
    ignore(array_name);
    ignore(destination);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }

  /**
   * @brief Reports that MAT-file tensor writes are unavailable.
   */
  template <size_t dimension, execution_space space, typename data_t>
  void write_array(std::string array_name, const ::boba::Tensor<dimension, space, data_t>& source)
  {
    ignore(array_name);
    ignore(source);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }

  /**
   * @brief Reports that MAT-file fixed-size cell-array reads are unavailable.
   */
  template <size_t dimension_out, size_t dimension_in, execution_space space, typename data_t>
  void read_cell_array(const std::string& array_name, ::boba::Array<::boba::Tensor<dimension_in, space, data_t>, dimension_out>& destination)
  {
    ignore(array_name);
    ignore(destination);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }

  /**
   * @brief Reports that MAT-file fixed-size cell-array writes are unavailable.
   */
  template <size_t dimension_out, size_t dimension_in, execution_space space, typename data_t>
  void write_cell_array(const std::string& array_name, const ::boba::Array<::boba::Tensor<dimension_in, space, data_t>, dimension_out>& source)
  {
    ignore(array_name);
    ignore(source);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }

  /**
   * @brief Reports that MAT-file vector cell-array reads are unavailable.
   */
  template <size_t dimension, execution_space space, typename data_t>
  void read_cell_array(const std::string& array_name, std::vector<::boba::Tensor<dimension, space, data_t>>& destination)
  {
    ignore(array_name);
    ignore(destination);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }

  /**
   * @brief Reports that MAT-file vector cell-array writes are unavailable.
   */
  template <size_t dimension, execution_space space, typename data_t>
  void write_cell_array(const std::string& array_name, const std::vector<::boba::Tensor<dimension, space, data_t>>& source)
  {
    ignore(array_name);
    ignore(source);
    boba_error("Boba was not compiled with MATLAB.  Try adding BOBA_MATLAB=1 after your 'make' command ");
  }
};

#endif

} // namespace detail
} // namespace boba
