// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace boba
{

/**
 * \brief Selects `op(A)` in sparse products such as `y = alpha * op(A) * x + beta * y`.
 *
 * `transpose` is the plain transpose; complex values are not conjugated.
 */
enum struct sparse_op
{
  none,
  transpose
};

} // namespace boba
