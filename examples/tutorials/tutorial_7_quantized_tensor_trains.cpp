// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
#include "../tests/common.hpp"

/*
  This tutorial introduces quantized tensor trains (QTTs).

  A QTT represents a vector of length base^exponent by viewing its single
  long index as a sequence of smaller base-sized indices. For example, when
  base = 2, a vector of length 2^d can be viewed as a d-dimensional tensor
  with each mode having size 2, and that tensor can then be compressed in
  tensor train form.
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main() {

  boba::splash();
  boba::init();

  bool check = 1;

  checkpoint();
  {
    boba_print("Quantized tensor train from a constant vector");

    /*
      A QTT is defined by a base and an exponent.
      The full vector length is

        base^exponent

      Here we choose base = 2 and exponent = 4, so the QTT represents
      a vector of length 2^4 = 16.
    */
    size_t base = 2;
    size_t exponent = 4;
    size_t full_size = ::boba::pow(base, exponent);

    /*
      Let us first build the full vector explicitly and fill it with ones.
    */
    ::boba::Vector<space, double> vector_A({full_size});
    vector_A.rename("ones");
    vector_A.fill_with(1.0);

    /*
      Method 1: compress the full vector into a QTT.
    */
    ::boba::QuantizedTensorTrain<space, double> qtt_A(base, exponent);
    qtt_A.rename("compressed_ones_qtt");
    qtt_A.compress(vector_A);

    boba_print(vector_A.size());
    boba_print(qtt_A.get_full_size());
    boba_print(qtt_A.get_number_elements());
    boba_print(qtt_A.compression_rate());

    /*
      Since this vector is constant, it should have a very compact QTT
      representation with very low ranks.
    */

    boba_print(qtt_A.ranks_string());
    /*
      The QTT ranks are also called bond dimensions. These are the sizes of
      the internal indices connecting neighboring cores.

      For example, ranks (1, 2, 3, 1) mean the cores have shapes
      (1, base, 2), (2, base, 3), and (3, base, 1).

      Small bond dimensions usually indicate strong compression.
    */

    boba_print("We also could have just used a special function for constant vectors");

    /*
      Method 2: directly construct a rank-one QTT filled with ones.
    */
    ::boba::QuantizedTensorTrain<space, double> qtt_A2(base, exponent);
    qtt_A2.rename("direct_ones_qtt");
    qtt_A2.fill_with(1.0);

    boba_print(qtt_A2.get_full_size());
    boba_print(qtt_A2.get_number_elements());
    boba_print(qtt_A2.compression_rate());
    boba_print(qtt_A2.ranks_string());

    /*
      These two QTTs should represent the same vector.
    */
    auto difference = qtt_A - qtt_A2;
    auto error = ::boba::norm_frobenius(difference);
    boba_print(error);
    pass_or_fail(check, error, 1.0e-6);
  }
  checkpoint();
  {
    boba_print("What is the quantized tensor train made of?");

    size_t base = 2;
    size_t exponent = 4;
    ::boba::QuantizedTensorTrain<space, double> qtt(base, exponent);
    qtt.rename("qtt_A");
    qtt.fill_with(1.0);

    /*
      Printing the QTT shows all of its cores.
    */
    qtt.print();

    /*
      We can also inspect an individual core directly.
    */
    qtt.cores[0].print();

    /*
      Since QTTs are built from tensor train cores, we can modify those
      cores directly.
    */
    auto core_view = qtt.cores[0].view();

    ::boba::loop<space, 3>(core_view.sizes(),
      [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto [rank_left, index, rank_right] = mid;
      core_view({rank_left, index, rank_right}) = static_cast<double>(index);
    });

    /*
      Now the first core is no longer constant, so the represented vector
      has changed.
    */
    qtt.print();
  }

  checkpoint();
  {
    boba_print("\n Structured versus unstructured vectors");

    size_t base = 2;
    size_t exponent = 6;
    size_t full_size = ::boba::pow(base, exponent);

    /*
      First, build a structured vector from a smooth function of the index.
    */
    ::boba::Vector<space, double> structured_vector({full_size});
    {
      auto structured_view = structured_vector.view();
      ::boba::loop<space, 1>(full_size,
        [=]__boba_host_device__(size_t i)
      {
        structured_view(i) = boba::exp(-static_cast<double>(i)/static_cast<double>(full_size));
      });
    }

    ::boba::QuantizedTensorTrain<space, double> structured_qtt(base, exponent);
    structured_qtt.rename("structured_qtt");
    structured_qtt.compress(structured_vector);

    boba_print("structured vector");
    boba_print(structured_qtt.get_full_size());
    boba_print(structured_qtt.get_number_elements());
    boba_print(structured_qtt.compression_rate());
    boba_print(structured_qtt.ranks_string());

    /*
      Now build a vector with random entries.
    */
    ::boba::Vector<space, double> random_vector({full_size});
    random_vector.fill_with_random();

    ::boba::QuantizedTensorTrain<space, double> random_qtt(base, exponent);
    random_qtt.rename("random_qtt");
    random_qtt.compress(random_vector);

    boba_print("random vector");
    boba_print(random_qtt.get_full_size());
    boba_print(random_qtt.get_number_elements());
    boba_print(random_qtt.compression_rate());
    boba_print(random_qtt.ranks_string());

    /*
      The structured vector 'can' compresses extremely well, with rank one QTT cores.
      By contrast, the random vector has much larger QTT ranks and in this case costing
      even more storage than the vector it estimates. QTT is therefore most
      effective for vectors with low-rank structure in their quantized form.
    */
  }
  checkpoint();
  {
    boba_print("\n Importance of quantized coordinates");

    /*
      A QTT of base 2 and exponent 3 represents a vector of length

        2^3 = 8

      The key idea is that the long vector index is reinterpreted as a
      multiindex with three coordinates, each taking values in {0, 1}.
      In other words, the vector is viewed as a tensor with sizes {2, 2, 2}.
    */

    size_t base = 2;
    size_t exponent = 3;
    size_t full_size = ::boba::pow(base, exponent);

    ::boba::Array<size_t, 3> quantized_sizes{base, base, base};
    ::boba::Multiindexer<3> qtt_indexer(quantized_sizes);

    boba_print(full_size);
    boba_print(quantized_sizes);

    /*
      The Multiindexer lets us convert between the long vector index and the
      quantized multiindex used by the QTT.
    */
    for(size_t i = 0; i < full_size; i++)
    {
      auto mid = qtt_indexer.multiindex(i);
      std::cout << "index " << i << " <-> {"
                << mid[0] << ", " << mid[1] << ", " << mid[2] << "}" << std::endl;
    }

    /*
      This quantized multiindex is the coordinate system in which QTT
      compression takes place. A QTT is therefore not just a compressed
      vector format: it is a tensor train representation of the vector
      after its index has been split into small base-sized coordinates.
    */
  }
    checkpoint();
  {
    boba_print("\n Padding a vector to fit a QTT");

    /*
      A base-2 QTT represents vectors whose length is exactly 2^d.
      If a vector length is not a power of 2, one simple strategy is to
      pad it with zeros up to the next power of 2.

      Here we start with a vector of length 10 and pad it to length 16.
    */

    size_t original_size = 10;
    size_t padded_exponent = 4;
    size_t base = 2;
    size_t padded_size = ::boba::pow(base, padded_exponent);

    ::boba::Vector<space, double> original_vector({original_size});
    {
      auto original_view = original_vector.view();
      ::boba::loop<space, 1>(original_size,
        [=]__boba_host_device__(size_t i)
      {
        original_view(i) = boba::exp(-static_cast<double>(i)/static_cast<double>(original_size));
      });
    }

    ::boba::Vector<space, double> padded_vector({padded_size});
    padded_vector.fill_with(0.0);

    {
      auto original_view = original_vector.const_view();
      auto padded_view = padded_vector.view();

      ::boba::loop<space, 1>(original_size,
        [=]__boba_host_device__(size_t i)
      {
        padded_view(i) = original_view(i);
      });
    }

    boba_print(original_size);
    boba_print(padded_size);

    ::boba::QuantizedTensorTrain<space, double> padded_qtt(base, padded_exponent);
    padded_qtt.rename("padded_qtt");
    padded_qtt.compress(padded_vector);

    boba_print(padded_qtt.get_full_size());
    boba_print(padded_qtt.get_number_elements());
    boba_print(padded_qtt.compression_rate());
    boba_print(padded_qtt.ranks_string());

    /*
      The QTT represents the padded vector, so to recover the original data
      we compare only the first original_size entries after unrolling.
    */
    auto decompressed_padded_vector = padded_qtt.decompress();

    ::boba::Vector<space, double> recovered_vector({original_size});
    {
      auto recovered_view = recovered_vector.view();
      auto decompressed_view = decompressed_padded_vector.const_view();

      ::boba::loop<space, 1>(original_size,
        [=]__boba_host_device__(size_t i)
      {
        recovered_view(i) = decompressed_view(i);
      });
    }

    auto error = ::boba::norm_difference_frobenius(recovered_vector, original_vector);
    boba_print(error);
    pass_or_fail(check, error, 1.0e-6);

    /*
      Padding is a simple way to use QTT on data whose size is not already
      of the form base^exponent.
    */
  }
  checkpoint();
  {
    boba_print("\n Adding a trivial quantized dimension");
     /*
      In the previous section, we padded the entries of a vector so that its
      full size matched a power of 2. There is another related idea: instead
      of padding the data itself, we can enlarge the quantized representation
      by adding an extra trivial quantized mode.

      In this section, we compare a smaller QTT and a larger QTT that differ
      by one such trivial mode. This increases the represented power-of-two
      size while keeping the QTT ranks equal to one.

      This is different from padding a physical vector with zeros. Here we are
      enlarging the tensorized representation itself. That can be useful when
      building larger structured QTT or QTT-matrix objects, or when aligning
      quantized dimensions so that vectors and operators can be represented in
      a compatible tensor-product form.
    */

    size_t base = 2;

    ::boba::QuantizedTensorTrain<space, double> qtt_small(base, 3);
    qtt_small.fill_with(1.0);

    ::boba::QuantizedTensorTrain<space, double> qtt_large(base, 4);
    qtt_large.fill_with(1.0);

    boba_print("small QTT");
    boba_print(qtt_small.get_full_size());
    boba_print(qtt_small.get_number_elements());
    boba_print(qtt_small.ranks_string());

    boba_print("large QTT");
    boba_print(qtt_large.get_full_size());
    boba_print(qtt_large.get_number_elements());
    boba_print(qtt_large.ranks_string());

    /*
      The second QTT has one more quantized mode. This is different from
      padding a vector with zeros: we are enlarging the quantized tensor
      structure itself by adding another trivial factor.
    */
  }
  checkpoint();

  boba::finalize();
  return final_check(check);
}
// clang-format on
