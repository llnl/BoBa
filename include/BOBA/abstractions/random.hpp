// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <random>

namespace boba::random
{

using seed_type = std::uint64_t;
using engine_type = std::mt19937_64;

/**
 * @brief Owns reproducible random-number-generator state.
 *
 * A context initialized with the same seed reproduces the same random sequence.
 * A context is not thread-safe; callers running concurrently must provide a
 * separate, distinctly seeded context to each thread or synchronize access.
 */
class RandomContext
{
public:
  using result_type = engine_type::result_type;

  /** @brief Constructs a context seeded from `std::random_device`. */
  RandomContext();

  /**
   * @brief Constructs a context with a reproducible seed.
   * @param seed Seed for the random sequence.
   */
  explicit RandomContext(seed_type seed);

  /**
   * @brief Resets the generator to the beginning of the given seed's sequence.
   * @param seed New seed for the generator.
   */
  void set_seed(seed_type seed);

  /** @brief Returns this context's seed. */
  [[nodiscard]] seed_type current_seed() const;

  /** @brief Returns the minimum value produced by this generator. */
  [[nodiscard]] static constexpr result_type min()
  {
    return engine_type::min();
  }

  /** @brief Returns the maximum value produced by this generator. */
  [[nodiscard]] static constexpr result_type max()
  {
    return engine_type::max();
  }

  /** @brief Produces the next value and advances the generator state. */
  result_type operator()();

private:
  seed_type seed_;
  engine_type engine_;
};

/**
 * @brief Returns the process-wide default random context.
 * The returned context is not thread-safe. Concurrent callers are responsible
 * for synchronization or for supplying separate `RandomContext` instances.
 *
 * @return Shared context used by no-argument random operations.
 */
[[nodiscard]] RandomContext& default_context();

/**
 * @brief Resets the process-wide default generator to the given seed.
 * @param seed New seed for the default generator.
 */
inline void set_seed(seed_type seed)
{
  default_context().set_seed(seed);
}

/**
 * @brief Returns the effective seed of the process-wide default sequence.
 *
 * The initial seed is obtained from `std::random_device` on first use.
 */
[[nodiscard]] inline seed_type current_seed()
{
  return default_context().current_seed();
}

} // namespace boba::random
