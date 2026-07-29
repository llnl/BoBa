// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#ifdef BOBA_HDF5
#include "hdf5.h"
#endif

// -----------------------------------------------------
// HDF5File interface
// -----------------------------------------------------
namespace boba
{
namespace detail
{

#ifdef BOBA_HDF5

/**
 * @brief Permutes tensor storage for HDF5 row-major interchange.
 * @tparam space Tensor execution space.
 * @tparam dimension Tensor rank.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @param tensor_permutation Tensor to permute in place.
 * @param permutations Axis permutation order.
 */
template <execution_space space, size_t dimension, typename data_t>
void permute_hdf5(
  Tensor<dimension, space, data_t>& tensor_permutation,
  const Array<index_t, dimension>& permutations)
{
  boba_always_assert(is_valid_permutation(permutations), "Invalid permutation.");
  if (permutations == range<index_t, dimension>())
  {
    // permutation is the identity (0, 1, ..., dimension)
    return;
  }

  Array<index_t, dimension> local_permutations = permutations;
  Array<index_t, dimension> local_sizes_old = tensor_permutation.sizes();
  Array<index_t, dimension> local_sizes_new = permute(local_sizes_old, local_permutations);

  Tensor<dimension, space, data_t> old_tensor(local_sizes_old);

  checkpoint();
  boba::detail::memcpy<space, space>(
    old_tensor.data(), tensor_permutation.const_data(), static_cast<size_t>(old_tensor.size()));

  // Resize output tensor
  checkpoint();
  tensor_permutation.reshape(local_sizes_new);

  if (tensor_permutation.size() == 0)
  {
    return;
  }

  //
  // Fallback scheme
  //
  auto old_view = old_tensor.const_view();
  auto tensor_permutation_view = tensor_permutation.view();
  ::boba::loop<space, 1>(tensor_permutation_view.size(),
                         [=] __boba_host_device__(index_t i)
  {
    auto multiindex_old = old_view.multiindex(i);
    auto multiindex_new = permute(multiindex_old, local_permutations);
    auto value = old_view(multiindex_old);
    tensor_permutation_view(multiindex_new) = value;
  });
}

struct HDF5File
{
  /**
   * @brief Opens an HDF5 file for reading or writing.
   * @param filename Path to the HDF5 file.
   * @param permissions Access mode string: `r`, `w`, or `rw`.
   */
  HDF5File(std::string_view filename, std::string permissions)
  {
    // open hdf5-file
    m_filename = filename;
    m_permissions = permissions;
    if (m_permissions == "w")
    {
      // Create a new file or truncate if it exists
      h5_file = H5Fcreate(m_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    else if (m_permissions == "rw")
    {
      // Create a new file or truncate if it exists
      h5_file = H5Fopen(m_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    }
    else if (m_permissions == "r")
    {
      // Open an existing file
      h5_file = H5Fopen(m_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    }
    else
    {
      boba_error("unsupported permissions " + permissions + ", options are [r, w, rw]");
    }
    boba_always_assert(h5_file >= 0, "Failed to open HDF5 file " + m_filename);
  }

  /**
   * @brief destructor
   */
  ~HDF5File()
  {
    H5Fclose(h5_file);
  }

  /**
   * @brief Reads a tensor array into a non-host tensor by staging through host memory.
   * @tparam dimension Tensor rank.
   * @tparam space Destination execution space.
   * @tparam data_t Scalar type.
   * @tparam index_t Index type.
   * @param array_name Dataset prefix to read.
   * @param destination Destination tensor.
   */
  template <size_t dimension, execution_space space, typename data_t>
    requires(space != host_space)
  void read_array(std::string array_name, ::boba::Tensor<dimension, space, data_t>& destination)
  {
    ::boba::Tensor<dimension, host_space, data_t> host_destination;
    read_array(array_name, host_destination);
    destination = host_destination;
  }

  /**
   * @brief Reads a tensor array from HDF5 into host memory.
   * @tparam dimension Tensor rank.
   * @tparam data_t Scalar type.
   * @tparam index_t Index type.
   * @param array_name Dataset prefix to read.
   * @param input Destination tensor.
   */
  template <size_t dimension, typename data_t>
  void read_array(std::string array_name, ::boba::Tensor<dimension, host_space, data_t>& input)
  {
    boba::Array<size_t, dimension> reverse;
    for (size_t d = 0; d < dimension; d++)
    {
      reverse[d] = (dimension - 1) - d;
    }
    // Permute backwards
    // Hdf5 is row-major, while boba is col-major!
    // this is fixed by a permutation back and forth
    boba::detail::permute_hdf5(input, reverse);

    // Read "sizes" field
    hid_t sizes_dataset = H5Dopen(h5_file, (array_name + "sizes").c_str(), H5P_DEFAULT);
    boba_always_assert(sizes_dataset >= 0, "Missing sizes field");
    hid_t sizes_dspace = H5Dget_space(sizes_dataset);
    hid_t sizes_dtype = H5Dget_type(sizes_dataset);
    int sizes_ndims = H5Sget_simple_extent_ndims(sizes_dspace);
    boba_always_assert(H5Tequal(sizes_dtype, H5T_NATIVE_LONG),
                       "sizes not type long");
    boba_always_assert(sizes_ndims == 1, "sizes not 1d array");
    hsize_t sizes_dim = 0;
    H5Sget_simple_extent_dims(sizes_dspace, &sizes_dim, nullptr);
    boba_always_assert(sizes_dim == static_cast<hsize_t>(dimension), "sizes array has wrong length");
    long lsizes[dimension];
    int status = H5Dread(sizes_dataset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, lsizes);
    boba_always_assert(status >= 0, "Failed to read dataset");

    Array<index_t, dimension> new_sizes;
    for (size_t d = 0; d < dimension; ++d)
    {
      boba_always_assert_nonnegative(lsizes[d], "tensor size must be nonnegative");
      new_sizes[d] = static_cast<index_t>(lsizes[d]);
    }
    input.resize(new_sizes);

    // Read "data" field
    hid_t data_dataset = H5Dopen(h5_file, (array_name + "data").c_str(), H5P_DEFAULT);
    boba_always_assert(data_dataset >= 0, "Missing data field");
    hid_t data_dspace = H5Dget_space(data_dataset);
    hid_t data_dtype = H5Dget_type(data_dataset);
    int data_ndims = H5Sget_simple_extent_ndims(data_dspace);
    boba_always_assert(data_ndims == 1, "data not 1d array");
    hsize_t data_dim = 0;
    H5Sget_simple_extent_dims(data_dspace, &data_dim, nullptr);
    boba_always_assert(data_dim > 0, "data array empty");
    boba_always_assert(data_dim == static_cast<hsize_t>(input.size()), "data size does not match tensor sizes");
    if constexpr (std::is_same_v<data_t, float>)
    {
      boba_always_assert(H5Tequal(data_dtype, H5T_NATIVE_FLOAT), "data not type float");
      status = H5Dread(data_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
    }
    else if constexpr (std::is_same_v<data_t, double>)
    {
      boba_always_assert(H5Tequal(data_dtype, H5T_NATIVE_DOUBLE), "data not type double");
      status = H5Dread(data_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
    }
    else if constexpr (std::is_same_v<data_t, int>)
    {
      boba_always_assert(H5Tequal(data_dtype, H5T_NATIVE_INT), "data not type double");
      status = H5Dread(data_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
    }
    else if constexpr (std::is_same_v<data_t, size_t>)
    {
      boba_always_assert(H5Tequal(data_dtype, H5T_NATIVE_ULONG), "data not type double");
      status = H5Dread(data_dataset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<float>>)
    {
      hid_t complex_type_id = H5Tcreate(H5T_COMPOUND, sizeof(boba::complex<float>));
      H5Tinsert(complex_type_id, "real", 0, H5T_NATIVE_FLOAT);
      H5Tinsert(complex_type_id, "imag", sizeof(float), H5T_NATIVE_FLOAT);
      boba_always_assert(H5Tequal(data_dtype, complex_type_id), "data not type complex float");
      status = H5Dread(data_dataset, complex_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
      H5Tclose(complex_type_id);
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<double>>)
    {
      hid_t complex_type_id = H5Tcreate(H5T_COMPOUND, sizeof(boba::complex<double>));
      H5Tinsert(complex_type_id, "real", 0, H5T_NATIVE_DOUBLE);
      H5Tinsert(complex_type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
      boba_always_assert(H5Tequal(data_dtype, complex_type_id), "data not type complex double");
      status = H5Dread(data_dataset, complex_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, input.data());
      H5Tclose(complex_type_id);
    }
    else
    {
      boba_error("Unknown data type");
    }
    boba_always_assert(status >= 0, "Failed to read dataset");

    // Cleanup
    H5Tclose(data_dtype);
    H5Sclose(data_dspace);
    H5Dclose(data_dataset);
    H5Tclose(sizes_dtype);
    H5Sclose(sizes_dspace);
    H5Dclose(sizes_dataset);

    // Permute back to col-major boba ordering
    boba::detail::permute_hdf5(input, reverse);
  }

  /**
   * @brief Writes a non-host tensor array by staging through host memory.
   * @tparam dimension Tensor rank.
   * @tparam space Source execution space.
   * @tparam data_t Scalar type.
   * @tparam index_t Index type.
   * @param array_name Dataset prefix to write.
   * @param source Source tensor.
   */
  template <size_t dimension, execution_space space, typename data_t>
    requires(space != host_space)
  void write_array(std::string array_name, const ::boba::Tensor<dimension, space, data_t>& source)
  {
    ::boba::Tensor<dimension, host_space, data_t> host_source = source;
    write_array(array_name, host_source);
  }

  /**
   * @brief Writes a host tensor array to HDF5.
   * @tparam dimension Tensor rank.
   * @tparam data_t Scalar type.
   * @tparam index_t Index type.
   * @param array_name Dataset prefix to write.
   * @param input_boba_ordering Source tensor in BoBa storage order.
   */
  template <size_t dimension, typename data_t>
  void write_array(std::string array_name, const ::boba::Tensor<dimension, host_space, data_t>& input_boba_ordering)
  {
    auto input_hdf5_ordering = input_boba_ordering;
    boba::Array<size_t, dimension> reverse;
    for (size_t d = 0; d < dimension; d++)
    {
      reverse[d] = (dimension - 1) - d;
    }
    // Permute backwards
    // Hdf5 is row-major, while boba is col-major!
    // this is fixed by a permutation back and forth
    boba::detail::permute_hdf5(input_hdf5_ordering, reverse);

    // Write "data" field
    hsize_t data_dim = input_hdf5_ordering.size();
    hid_t data_dataspace = H5Screate_simple(1, &data_dim, nullptr);
    hid_t data_dataset;
    int status;
    if constexpr (std::is_same_v<data_t, float>)
    {
      data_dataset = H5Dcreate(h5_file, (array_name + "data").c_str(), H5T_NATIVE_FLOAT, data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(data_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_hdf5_ordering.data());
    }
    else if constexpr (std::is_same_v<data_t, double>)
    {
      data_dataset = H5Dcreate(h5_file, (array_name + "data").c_str(), H5T_NATIVE_DOUBLE, data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(data_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_hdf5_ordering.data());
    }
    else if constexpr (std::is_same_v<data_t, int>)
    {
      data_dataset = H5Dcreate(h5_file, (array_name + "data").c_str(), H5T_NATIVE_INT, data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(data_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_hdf5_ordering.data());
    }
    else if constexpr (std::is_same_v<data_t, size_t>)
    {
      data_dataset = H5Dcreate(h5_file, (array_name + "data").c_str(), H5T_NATIVE_ULONG, data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(data_dataset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_hdf5_ordering.data());
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<float>>)
    {
      hid_t complex_type_id = H5Tcreate(H5T_COMPOUND, sizeof(boba::complex<float>));
      H5Tinsert(complex_type_id, "real", 0, H5T_NATIVE_FLOAT);
      H5Tinsert(complex_type_id, "imag", sizeof(float), H5T_NATIVE_FLOAT);
      data_dataset = H5Dcreate(h5_file, (array_name + "data").c_str(), complex_type_id, data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(data_dataset, complex_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_hdf5_ordering.data());
      H5Tclose(complex_type_id);
    }
    else if constexpr (std::is_same_v<data_t, boba::complex<double>>)
    {
      hid_t complex_type_id = H5Tcreate(H5T_COMPOUND, sizeof(boba::complex<double>));
      H5Tinsert(complex_type_id, "real", 0, H5T_NATIVE_DOUBLE);
      H5Tinsert(complex_type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
      data_dataset = H5Dcreate(h5_file, (array_name + "data").c_str(), complex_type_id, data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(data_dataset, complex_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_hdf5_ordering.data());
      H5Tclose(complex_type_id);
    }
    else
    {
      boba_error("Unknown data type");
    }
    boba_always_assert(status >= 0, "Failed to write dataset");

    // Write "sizes" field
    // index_t is a template param, so convert to long explicitly
    long lsizes[dimension];
    std::copy(input_hdf5_ordering.m_sizes.begin(), input_hdf5_ordering.m_sizes.end(), lsizes);
    hsize_t sizes_dim = dimension;
    hid_t sizes_dataspace = H5Screate_simple(1, &sizes_dim, nullptr);
    hid_t sizes_dataset = H5Dcreate(h5_file, (array_name + "sizes").c_str(), H5T_NATIVE_LONG, sizes_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(sizes_dataset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, lsizes);
    boba_always_assert(status >= 0, "Failed to write dataset");

    // Cleanup
    H5Dclose(sizes_dataset);
    H5Sclose(sizes_dataspace);
    H5Dclose(data_dataset);
    H5Sclose(data_dataspace);
  }

  /**
   * @brief Writes a scalar integer dataset.
   * @param value_name Dataset name.
   * @param value Value to write.
   */
  void write_int(std::string value_name, int value)
  {
    hid_t dset_id;

    // Dataset does not exist, create a scalar dataspace
    hid_t space = H5Screate(H5S_SCALAR);
    if (space < 0)
    {
      throw std::runtime_error("write_int: failed to create scalar dataspace");
    }

    // Create the dataset
    dset_id = H5Dcreate2(
      h5_file,
      value_name.c_str(),
      H5T_NATIVE_INT,
      space,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);

    H5Sclose(space);

    if (dset_id < 0)
    {
      throw std::runtime_error("write_int: failed to create dataset '" + value_name + "'");
    }

    herr_t status = H5Dwrite(
      dset_id,
      H5T_NATIVE_INT,
      H5S_ALL,
      H5S_ALL,
      H5P_DEFAULT,
      &value);

    boba_always_assert_nonnegative(status, "H5Dwrite failed");

    H5Dclose(dset_id);
  }

  /**
   * @brief Reads a scalar integer dataset.
   * @param value_name Dataset name.
   * @return Loaded integer value.
   */
  [[nodiscard]]
  int read_int(const std::string& value_name)
  {
    hid_t dataset = H5Dopen(h5_file, value_name.c_str(), H5P_DEFAULT);
    boba_always_assert_nonnegative(dataset, "H5Dopen failed");

    int value = 0;
    herr_t status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    boba_always_assert_nonnegative(status, "H5Dread failed");

    H5Dclose(dataset);
    return value;
  }

private:
  std::string m_filename;
  std::string m_permissions;
  hid_t h5_file;
};

#else

struct HDF5File
{
  /**
   * @brief Reports that HDF5 support is unavailable.
   */
  void not_compiled_error()
  {
    boba_error("Boba was not compiled with hdf5.  Try adding BOBA_HDF5=1 after your 'make' command ");
  }

  /**
   * @brief Opens an HDF5 file handle placeholder when HDF5 support is disabled.
   * @param filename Path to the HDF5 file.
   * @param permissions Access mode string.
   */
  HDF5File(std::string_view filename, std::string permissions)
  {
    ignore(filename);
    ignore(permissions);
    not_compiled_error();
  }

  /**
   * @brief destructor
   */
  ~HDF5File()
  {
  }

  /**
   * @brief Reports that HDF5 tensor reads are unavailable.
   * @param array_name Dataset prefix to read.
   * @param destination Destination tensor.
   */
  template <size_t dimension, execution_space space, typename data_t>
  void read_array(std::string array_name, ::boba::Tensor<dimension, space, data_t>& destination)
  {
    ignore(array_name);
    ignore(destination);
    not_compiled_error();
  }

  /**
   * @brief Reports that HDF5 tensor writes are unavailable.
   * @param array_name Dataset prefix to write.
   * @param source Source tensor.
   */
  template <size_t dimension, execution_space space, typename data_t>
  void write_array(std::string array_name, const ::boba::Tensor<dimension, space, data_t>& source)
  {
    ignore(array_name);
    ignore(source);
    not_compiled_error();
  }

  /**
   * @brief Reports that HDF5 scalar writes are unavailable.
   * @param value_name Dataset name.
   * @param value Value to write.
   */
  void write_int(std::string value_name, int value)
  {
    ignore(value_name);
    ignore(value);
    not_compiled_error();
  }

  /**
   * @brief Reports that HDF5 scalar reads are unavailable.
   * @param value_name Dataset name.
   * @return This function does not return successfully.
   */
  [[nodiscard]]
  int read_int(const std::string& value_name)
  {
    ignore(value_name);
    not_compiled_error();
    return 0;
  }
};

#endif

} // namespace detail
} // namespace boba
