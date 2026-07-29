// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace boba
{

/**
 * \brief Class describing Krylov-type solvers
 */
struct KrylovMethods
{
  static constexpr size_t gmres = 0;
  static constexpr size_t conjugate_gradient = 1;
  static constexpr size_t bicgstab = 2;
  static constexpr size_t flexible_gmres = 3;
  static constexpr size_t nonorthogonal_gmres = 4;
  static constexpr size_t number_of_methods = 5;
};

/**
 * \brief Krylov solver class
 *
 * \tparam operator_t Operator type
 * \tparam vector_t Vector type
 *
 * \note The operator_t and vector_t types must support the matrix-vector multiplication interface
 *       (`operator_t * vector_t -> vector_t`)
 */
template <typename operator_t, typename vector_t>
struct Krylov
{
  using data_t = typename vector_t::data_t;

  size_t outer_iterations = 0;
  size_t inner_iterations = 0;
  data_t relative_threshold = 1.e-08;
  data_t absolute_threshold = 1.e-14;

  bool verbose = false;

  operator_t linear_operator;

  bool left_preconditioner_set = false;
  operator_t left_preconditioner;

  bool right_preconditioner_set = false;
  operator_t right_preconditioner;

  size_t outer_iterations_width = 0;
  size_t inner_iterations_width = 0;
  size_t iterations_width = 0;

  size_t method = KrylovMethods::number_of_methods;

  data_t convergence_tolerance = 0.0;

  size_t used_outer_iterations = 0;
  size_t used_inner_iterations = 0;

  /**
   * \brief Construct a Krylov solver object
   *
   * \param[in] _outer_iterations Number of outer iterations
   * \param[in] _inner_iterations Number of inner iterations
   * \param[in] _relative_threshold Relative convergence threshold
   */
  Krylov(size_t _outer_iterations, size_t _inner_iterations, data_t _relative_threshold)
  {
    setup(_outer_iterations, _inner_iterations, _relative_threshold);
  }

  /**
   * \brief Set the matrix/operator for the Krylov solver
   *
   * \param[in] _matrix The matrix/operator \f$A\f$
   */
  void set_matrix(operator_t const& _matrix)
  {
    linear_operator = _matrix;
  }

  /**
   * \brief Set the left preconditioner
   *
   * \param[in] _left_preconditioner The left preconditioner \f$L\f$
   *
   * \note Given left preconditioner \f$L\f$, the linear system \f$A x = b\f$ is transformed into \f$L A x = L b\f$
   *       (rather than the more traditional \f$L^{-1} A x = L^{-1} b\f$).
   */
  void set_left_preconditioner(operator_t const& _left_preconditioner)
  {
    left_preconditioner = _left_preconditioner;
    left_preconditioner_set = true;
  }

  /**
   * \brief Set the right preconditioner
   *
   * \param[in] _right_preconditioner The right preconditioner \f$R\f$
   *
   * \note Given right preconditioner \f$R\f$, the linear system \f$A x = b\f$ in transformed into \f$A R u = b\f$ with
   *       \f$x = R u\f$ (rather than the more traditional \f$A R^{-1} u = b\f$ with \f$x = R^{-1} u\f$).
   *
   * \note Right preconditioner is not used for conjugate gradient.
   */
  void set_right_preconditioner(operator_t const& _right_preconditioner)
  {
    right_preconditioner = _right_preconditioner;
    right_preconditioner_set = true;
  }

  /**
   * \brief Sets up the Krylov solver
   *
   * \param[in] _outer_iterations Number of outer iterations
   * \param[in] _inner_iterations Number of inner iterations
   * \param[in] _relative_threshold Relative convergence threshold
   */
  void setup(size_t _outer_iterations, size_t _inner_iterations, data_t _relative_threshold)
  {
    outer_iterations = _outer_iterations;
    inner_iterations = _inner_iterations;
    relative_threshold = _relative_threshold;
  }

  /**
   * \brief Returns the string explaining the method
   *
   * \param[in] krylov_method Method identifier
   *
   * \return String representation of the method.
   */
  [[nodiscard]]
  static std::string method_string(size_t krylov_method)
  {
    switch (krylov_method)
    {
    case KrylovMethods::gmres:
      return "GMRES";
    case KrylovMethods::conjugate_gradient:
      return "Conjugate Gradient";
    case KrylovMethods::bicgstab:
      return "Bilinear CG Stabilized";
    case KrylovMethods::flexible_gmres:
      return "Flexible GMRES";
    case KrylovMethods::nonorthogonal_gmres:
      return "Nonorthogonal GMRES";
    default:
      return "Invalid method choice " + std::to_string(static_cast<size_t>(krylov_method));
    }
  }

  /**
   * \brief Computes the residual \f$b - A x\f$
   *
   * \param[in] input Current iterate \f$x\f$
   * \param[in] rhs Right hand side \f$b\f$
   *
   * \return Residual \f$r = b - A x\f$.
   */
  [[nodiscard]]
  vector_t compute_unpreconditioned_residual(vector_t const& input, vector_t const& rhs) const
  {
    checkpoint();
    vector_t residual = rhs - linear_operator * input;
    residual.rename("residual");
    return residual;
  }

  /**
   * \brief Computes the preconditioned residual \f$L (b - A x)\f$
   *
   * \param[in] input Current iterate \f$x\f$
   * \param[in] rhs Right hand side \f$b\f$
   *
   * \return Residual \f$r = L (b - A x)\f$.
   */
  [[nodiscard]]
  vector_t compute_residual(vector_t const& input, vector_t const& rhs) const
  {
    checkpoint();
    vector_t residual = compute_unpreconditioned_residual(input, rhs);
    if (left_preconditioner_set)
    {
      residual = left_preconditioner * residual;
    }
    return residual;
  }

  /**
   * \brief Runs the Krylov solver
   *
   * \note Wraps the Krylov solver in outer iterations, giving us restarted krylov
   *
   * \param[in] rhs Right hand side \f$b\f$
   * \param[in] x_in Initial guess \f$x_0\f$
   * \param[out] x_out Final answer \f$x\f$.
   */
  void solve(vector_t const& rhs, vector_t const& x_in, vector_t& x_out)
  {
    if (is_env_nonempty("BOBA_KRYLOV_VERBOSE"))
    {
      verbose = true;
    }

    boba_always_assert(outer_iterations > 0, "Nonpositive outer_iterations");
    boba_always_assert(inner_iterations > 0, "Nonpositive inner_iterations");

    outer_iterations_width = static_cast<size_t>(std::floor(std::log10(outer_iterations) + 1.0));
    inner_iterations_width = static_cast<size_t>(std::floor(std::log10(inner_iterations) + 1.0));
    iterations_width = static_cast<size_t>(std::floor(std::log10(outer_iterations * inner_iterations) + 1.0));

    checkpoint();
    x_out = x_in;
    used_inner_iterations = 0;
    used_outer_iterations = 0;
    bool converged = false;

    auto residual_norm = ::boba::norm_frobenius(compute_unpreconditioned_residual(x_out, rhs));
    convergence_tolerance = ::boba::max(absolute_threshold, residual_norm * relative_threshold);

    for (size_t i = 0; (not converged) and (i < outer_iterations); i++)
    {
      used_outer_iterations++;

      checkpoint();
      vector_t x_inner = x_out;

      switch (method)
      {
      case KrylovMethods::gmres:
        converged = gmres_inner_iteration_solve(rhs, x_inner, x_out);
        break;
      case KrylovMethods::conjugate_gradient:
        converged = cg_inner_iteration_solve(rhs, x_inner, x_out);
        break;
      case KrylovMethods::bicgstab:
        converged = bicgstab_inner_iteration_solve(rhs, x_inner, x_out);
        break;
      case KrylovMethods::flexible_gmres:
        converged = flexible_gmres_inner_iteration_solve(rhs, x_inner, x_out);
        break;
      case KrylovMethods::nonorthogonal_gmres:
        converged = nonorthogonal_gmres_inner_iteration_solve(rhs, x_inner, x_out);
        break;
      default:
        boba_error("Krylov method unset.");
        break;
      }
    }
  }

  /**
   * \brief Solve \f$A x = b\f$ via GMRES
   *
   * Based on https://en.wikipedia.org/wiki/Generalized_minimal_residual_method. The preconditioned version
   * is based off Yousef Saad's "A short course on preconditioned Krylov subspace methods" 2005 lecture slides
   * (https://www-users.cse.umn.edu/~saad/Calais/PREC.pdf)
   *
   * \param[in] rhs Right hand side \f$b\f$
   * \param[in] x_in Initial guess \f$x_0\f$
   * \param[in] x_out Final answer \f$x\f$
   *
   * \return `true` if the inner iteration converged, `false` otherwise.
   */
  bool gmres_inner_iteration_solve(vector_t const& rhs, vector_t const& x_in, vector_t& x_out)
  {
    BOBA_CALI_MARK

    BOBA_CALI_BEGIN("setup");

    x_out = x_in;

    checkpoint();
    auto residual = compute_residual(x_out, rhs);
    residual.round();

    auto residual_norm = ::boba::norm_frobenius(residual);

    if (verbose)
    {
      std::cout << "- outer = " << std::setw(outer_iterations_width) << used_outer_iterations                     //
                << "  inner = " << std::setw(inner_iterations_width) << 0                                         //
                << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                       //
                << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm               //
                << "  compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() //
                << std::endl;
    }

    if (residual_norm < convergence_tolerance)
    {
      return true;
    }

    std::vector<vector_t> vector_sequence(inner_iterations + 1);

    Matrix<execution_space::CPU, data_t> hessenberg_matrix({inner_iterations + 1, inner_iterations});
    hessenberg_matrix.fill_with_zeros();

    Vector<execution_space::CPU, data_t> residual_norms({inner_iterations + 1});
    Vector<execution_space::CPU, data_t> givens_cos({inner_iterations});
    Vector<execution_space::CPU, data_t> givens_sin({inner_iterations});

    residual *= 1.0 / residual_norm;

    vector_sequence.at(0) = residual;
    residual_norms({0}) = residual_norm;

    BOBA_CALI_SWITCH("setup", "iteration_loops");

    bool converged = false;

    size_t iteration = 0;
    for (size_t i = 0; (not converged) and (i < inner_iterations); i++)
    {
      iteration++;
      used_inner_iterations++;

      checkpoint();
      auto z = vector_sequence.at(i);
      if (right_preconditioner_set)
      {
        z = right_preconditioner * z;
        z.round();
      }

      vector_sequence.at(i + 1) = linear_operator * z;
      vector_sequence.at(i + 1).round();

      if (left_preconditioner_set)
      {
        vector_sequence.at(i + 1) = left_preconditioner * vector_sequence.at(i + 1);
        vector_sequence.at(i + 1).round();
      }

      // modified Gram-Schmidt
      for (size_t j = 0; j <= i; ++j)
      {
        auto t = inner_product(vector_sequence.at(i + 1), vector_sequence.at(j));
        hessenberg_matrix({j, i}) = t;
        vector_sequence.at(i + 1) -= t * vector_sequence.at(j);
        vector_sequence.at(i + 1).round();
      }

      {
        auto t = ::boba::norm_frobenius(vector_sequence.at(i + 1));
        hessenberg_matrix({i + 1, i}) = t;
        if (t > std::numeric_limits<data_t>::epsilon())
        {
          vector_sequence.at(i + 1) *= 1.0 / t;
        }
      }

      // Givens rotation
      if (i != 0)
      {
        for (size_t k = 1; k <= i; ++k)
        {
          auto t = hessenberg_matrix({k - 1, i});
          hessenberg_matrix({k - 1, i}) = givens_sin({k - 1}) * hessenberg_matrix({k, i}) + givens_cos({k - 1}) * t;
          hessenberg_matrix({k, i}) = givens_cos({k - 1}) * hessenberg_matrix({k, i}) - givens_sin({k - 1}) * t;
        }
      }

      auto gamma = std::hypot(hessenberg_matrix({i, i}), hessenberg_matrix({i + 1, i}));
      if (gamma < std::numeric_limits<data_t>::epsilon())
      {
        givens_cos({i}) = 0.0;
        givens_sin({i}) = 0.0;
      }
      else
      {
        givens_cos({i}) = hessenberg_matrix({i, i}) / gamma;
        givens_sin({i}) = hessenberg_matrix({i + 1, i}) / gamma;
      }

      hessenberg_matrix({i, i}) = givens_cos({i}) * hessenberg_matrix({i, i}) + givens_sin({i}) * hessenberg_matrix({i + 1, i});

      residual_norms({i + 1}) = -givens_sin({i}) * residual_norms({i});
      residual_norms({i}) = givens_cos({i}) * residual_norms({i});

      residual_norm = boba::abs(residual_norms({i + 1}));

      if (verbose)
      {
        std::cout << "  outer = " << std::setw(outer_iterations_width) << used_outer_iterations                                         //
                  << "  inner = " << std::setw(inner_iterations_width) << i + 1                                                         //
                  << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                                           //
                  << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm                                   //
                  << "  compression_rate = " << std::scientific << std::setprecision(2) << vector_sequence.at(i + 1).compression_rate() //
                  << std::endl;
      }

      if (residual_norm < convergence_tolerance)
      {
        converged = true;
      }
    }

    BOBA_CALI_SWITCH("iteration_loops", "back_substitution");

    residual_norms({iteration - 1}) = residual_norms({iteration - 1}) / hessenberg_matrix({iteration - 1, iteration - 1});
    for (size_t kk = iteration; kk > 1; --kk)
    {
      const size_t k = kk - 1;
      auto t = residual_norms({k - 1});
      for (size_t j = k + 1; j <= iteration; ++j)
      {
        t -= hessenberg_matrix({k - 1, j - 1}) * residual_norms({j - 1});
      }
      residual_norms({k - 1}) = t / hessenberg_matrix({k - 1, k - 1});
    }

    BOBA_CALI_SWITCH("back_substitution", "linear_combination");

    for (size_t j = 0; j < iteration; ++j)
    {
      vector_t z = vector_sequence.at(j);

      if (right_preconditioner_set)
      {
        z = right_preconditioner * z;
        z.round();
      }

      x_out += residual_norms({j}) * z;
      x_out.round();
    }

    BOBA_CALI_END("linear_combination");

    return converged;
  }

  /**
   * \brief  Solve \f$A x = b\f$ via Conjugate Gradient
   *
   * Based on Yousef Saad's "A short course on preconditioned Krylov subspace methods" 2005 lecture slides
   * (https://www-users.cse.umn.edu/~saad/Calais/PREC.pdf).
   *
   * \param[in] rhs Right hand side \f$b\f$
   * \param[in] x_in Initial guess \f$x_0\f$
   * \param[in] x_out Final answer \f$x\f$
   *
   * \return `true` if the inner iteration converged, `false` otherwise.
   */
  bool cg_inner_iteration_solve(vector_t const& rhs, vector_t const& x_in, vector_t& x_out)
  {
    BOBA_CALI_MARK

    BOBA_CALI_BEGIN("setup");

    x_out = x_in;

    checkpoint();
    vector_t r = compute_unpreconditioned_residual(x_in, rhs);
    r.round();

    checkpoint();
    vector_t z = r;
    if (left_preconditioner_set)
    {
      z = left_preconditioner * z;
      z.round();
    }

    vector_t p = z;

    data_t r_dot_z = inner_product(r, z);

    auto residual_norm = boba::sqrt(r_dot_z);

    if (verbose)
    {
      std::cout << "- outer = " << std::setw(outer_iterations_width) << used_outer_iterations                     //
                << "  inner = " << std::setw(inner_iterations_width) << 0                                         //
                << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                       //
                << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm               //
                << "  compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() //
                << std::endl;
    }

    if (residual_norm < convergence_tolerance)
    {
      return true;
    }

    BOBA_CALI_SWITCH("setup", "iteration_loops");

    bool converged = false;

    for (size_t k = 0; (not converged) and (k < inner_iterations); k++)
    {
      used_inner_iterations++;

      checkpoint();
      vector_t q = linear_operator * p;
      q.round();

      data_t q_dot_p = inner_product(q, p);

      data_t alpha = r_dot_z / q_dot_p;

      checkpoint();
      x_out += alpha * p;
      x_out.round();

      checkpoint();
      r -= alpha * q;
      r.round();

      checkpoint();
      z = r;
      if (left_preconditioner_set)
      {
        z = left_preconditioner * z;
        z.round();
      }

      data_t r_dot_z_new = inner_product(r, z);

      residual_norm = boba::sqrt(r_dot_z_new);

      if (verbose)
      {
        std::cout << "  outer = " << std::setw(outer_iterations_width) << used_outer_iterations                     //
                  << "  inner = " << std::setw(inner_iterations_width) << k + 1                                     //
                  << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                       //
                  << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm               //
                  << "  compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() //
                  << std::endl;
      }

      if (residual_norm < convergence_tolerance)
      {
        converged = true;
        break;
      }

      data_t beta = r_dot_z_new / r_dot_z;

      checkpoint();
      p = z + beta * p;
      p.round();

      r_dot_z = r_dot_z_new;
    }

    BOBA_CALI_END("iteration_loops");

    return converged;
  }

  /**
   * \brief Solve \f$A x = b\f$ via bilinear stabilized Conjugate Gradient.
   *
   * Based on https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
   *
   * \param[in] rhs Right hand side \f$b\f$
   * \param[in] x_in Initial guess \f$x_0\f$
   * \param[in] x_out Final answer \f$x\f$
   *
   * \return `true` if the inner iteration converged, `false` otherwise.
   */
  bool bicgstab_inner_iteration_solve(vector_t const& rhs, vector_t const& x_in, vector_t& x_out)
  {
    BOBA_CALI_MARK

    BOBA_CALI_BEGIN("setup");

    x_out = x_in;

    checkpoint();
    vector_t r = compute_unpreconditioned_residual(x_out, rhs);
    r.round();

    checkpoint();
    data_t residual_norm = ::boba::norm_frobenius(left_preconditioner_set ? left_preconditioner * r : r);

    if (verbose)
    {
      std::cout << "- outer = " << std::setw(outer_iterations_width) << used_outer_iterations                     //
                << "  inner = " << std::setw(inner_iterations_width) << 0                                         //
                << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                       //
                << "                   "                                                                          //
                << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm               //
                << "  compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() //
                << std::endl;
    }

    if (residual_norm < convergence_tolerance)
    {
      return true;
    }

    vector_t r_hat = r;

    data_t rho = inner_product(r_hat, r);

    vector_t p = r;

    BOBA_CALI_SWITCH("setup", "iteration_loops");

    bool converged = false;

    for (size_t k = 0; k < inner_iterations; k++)
    {
      used_inner_iterations++;

      checkpoint();
      vector_t y = p;
      if (left_preconditioner_set)
      {
        y = left_preconditioner * y;
        y.round();
      }
      if (right_preconditioner_set)
      {
        y = right_preconditioner * y;
        y.round();
      }

      checkpoint();
      vector_t v = linear_operator * y;
      v.round();

      checkpoint();
      auto alpha = rho / inner_product(r_hat, v);

      checkpoint();
      vector_t h = x_out + alpha * y;
      h.round();

      checkpoint();
      vector_t s = r - alpha * v;
      s.round();

      auto s_norm = ::boba::norm_frobenius(s);

      if (verbose)
      {
        std::cout << "  outer = " << std::setw(outer_iterations_width) << used_outer_iterations //
                  << "  inner = " << std::setw(inner_iterations_width) << k + 1                 //
                  << "  iteration = " << std::setw(iterations_width) << used_inner_iterations   //
                  << "  s_norm = " << std::scientific << std::setprecision(2) << s_norm;
      }

      checkpoint();
      if (s_norm < convergence_tolerance)
      {
        x_out = h;
        converged = true;
        std::cout << "                            compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() << std::endl;
        break;
      }

      checkpoint();
      vector_t s_hat = s;
      if (left_preconditioner_set)
      {
        s_hat = left_preconditioner * s_hat;
        s_hat.round();
      }
      vector_t z = s_hat;
      if (right_preconditioner_set)
      {
        z = right_preconditioner * z;
        z.round();
      }

      checkpoint();
      vector_t t = linear_operator * z;
      t.round();

      checkpoint();
      vector_t t_hat = t;
      if (left_preconditioner_set)
      {
        t_hat = left_preconditioner * t_hat;
        t_hat.round();
      }
      auto omega = inner_product(t_hat, s_hat) / inner_product(t_hat, t_hat);

      checkpoint();
      x_out = h + omega * z;
      x_out.round();

      checkpoint();
      r = s - omega * t;
      r.round();

      checkpoint();
      residual_norm = ::boba::norm_frobenius(r);

      if (verbose)
      {
        std::cout << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm               //
                  << "  compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() //
                  << std::endl;
      }

      checkpoint();
      if (residual_norm < convergence_tolerance)
      {
        converged = true;
        break;
      }

      checkpoint();
      auto rho_new = inner_product(r_hat, r);

      checkpoint();
      auto beta = (rho_new / rho) * (alpha / omega);

      checkpoint();
      vector_t u = p - omega * v;
      u.round();
      p = r + beta * u;
      p.round();

      checkpoint();
      rho = rho_new;
    }

    BOBA_CALI_END("iteration_loops");

    return converged;
  }

  /**
   * \brief Solve \f$A x = b\f$ via flexible GMRES
   *
   * Based on "A Flexible Inner-Outer Preconditioned GMRES Algorithm" by Youcef Saad
   * (https://epubs.siam.org/doi/10.1137/0914028)
   *
   * \param[in] rhs Right hand side \f$b\f$
   * \param[in] x_in Initial guess \f$x_0\f$
   * \param[in] x_out Final answer \f$x\f$
   *
   * \return `true` if the inner iteration converged, `false` otherwise.
   */
  bool flexible_gmres_inner_iteration_solve(vector_t const& rhs, vector_t const& x_in, vector_t& x_out)
  {
    BOBA_CALI_MARK

    BOBA_CALI_BEGIN("setup");

    x_out = x_in;

    checkpoint();
    auto residual = compute_residual(x_out, rhs);
    residual.round();

    auto residual_norm = ::boba::norm_frobenius(residual);

    if (verbose)
    {
      std::cout << "- outer = " << std::setw(outer_iterations_width) << used_outer_iterations                     //
                << "  inner = " << std::setw(inner_iterations_width) << 0                                         //
                << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                       //
                << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm               //
                << "  compression_rate = " << std::scientific << std::setprecision(2) << x_out.compression_rate() //
                << std::endl;
    }

    if (residual_norm < convergence_tolerance)
    {
      return true;
    }

    std::vector<vector_t> vector_sequence(inner_iterations + 1);
    std::vector<vector_t> krylov_basis(inner_iterations);

    Matrix<execution_space::CPU, data_t> hessenberg_matrix({inner_iterations + 1, inner_iterations});
    hessenberg_matrix.fill_with_zeros();

    Vector<execution_space::CPU, data_t> residual_norms({inner_iterations + 1});
    Vector<execution_space::CPU, data_t> givens_cos({inner_iterations});
    Vector<execution_space::CPU, data_t> givens_sin({inner_iterations});

    residual *= 1.0 / residual_norm;

    vector_sequence.at(0) = residual;
    residual_norms({0}) = residual_norm;

    BOBA_CALI_SWITCH("setup", "iteration_loops");

    bool converged = false;

    size_t iteration = 0;
    for (size_t i = 0; (not converged) and (i < inner_iterations); i++)
    {
      iteration++;
      used_inner_iterations++;

      checkpoint();
      krylov_basis.at(i) = vector_sequence.at(i);
      if (right_preconditioner_set)
      {
        krylov_basis.at(i) = right_preconditioner * krylov_basis.at(i);
        krylov_basis.at(i).round();
      }

      vector_sequence.at(i + 1) = linear_operator * krylov_basis.at(i);
      vector_sequence.at(i + 1).round();

      if (left_preconditioner_set)
      {
        vector_sequence.at(i + 1) = left_preconditioner * vector_sequence.at(i + 1);
        vector_sequence.at(i + 1).round();
      }

      // modified Gram-Schmidt
      for (size_t j = 0; j <= i; ++j)
      {
        auto t = inner_product(vector_sequence.at(i + 1), vector_sequence.at(j));
        hessenberg_matrix({j, i}) = t;
        vector_sequence.at(i + 1) -= t * vector_sequence.at(j);
        vector_sequence.at(i + 1).round();
      }

      {
        auto t = ::boba::norm_frobenius(vector_sequence.at(i + 1));
        hessenberg_matrix({i + 1, i}) = t;
        if (t > std::numeric_limits<data_t>::epsilon())
        {
          vector_sequence.at(i + 1) *= 1.0 / t;
        }
      }

      // Givens rotation
      if (i != 0)
      {
        for (size_t k = 1; k <= i; ++k)
        {
          auto t = hessenberg_matrix({k - 1, i});
          hessenberg_matrix({k - 1, i}) = givens_sin({k - 1}) * hessenberg_matrix({k, i}) + givens_cos({k - 1}) * t;
          hessenberg_matrix({k, i}) = givens_cos({k - 1}) * hessenberg_matrix({k, i}) - givens_sin({k - 1}) * t;
        }
      }

      auto gamma = std::hypot(hessenberg_matrix({i, i}), hessenberg_matrix({i + 1, i}));
      if (gamma < std::numeric_limits<data_t>::epsilon())
      {
        givens_cos({i}) = 0.0;
        givens_sin({i}) = 0.0;
      }
      else
      {
        givens_cos({i}) = hessenberg_matrix({i, i}) / gamma;
        givens_sin({i}) = hessenberg_matrix({i + 1, i}) / gamma;
      }

      hessenberg_matrix({i, i}) = givens_cos({i}) * hessenberg_matrix({i, i}) + givens_sin({i}) * hessenberg_matrix({i + 1, i});

      residual_norms({i + 1}) = -givens_sin({i}) * residual_norms({i});
      residual_norms({i}) = givens_cos({i}) * residual_norms({i});

      residual_norm = boba::abs(residual_norms({i + 1}));

      if (verbose)
      {
        std::cout << "  outer = " << std::setw(outer_iterations_width) << used_outer_iterations                                         //
                  << "  inner = " << std::setw(inner_iterations_width) << i + 1                                                         //
                  << "  iteration = " << std::setw(iterations_width) << used_inner_iterations                                           //
                  << "  residual_norm = " << std::scientific << std::setprecision(2) << residual_norm                                   //
                  << "  compression_rate = " << std::scientific << std::setprecision(2) << vector_sequence.at(i + 1).compression_rate() //
                  << std::endl;
      }

      if (residual_norm < convergence_tolerance)
      {
        converged = true;
      }
    }

    BOBA_CALI_SWITCH("iteration_loops", "back_substitution");

    residual_norms({iteration - 1}) = residual_norms({iteration - 1}) / hessenberg_matrix({iteration - 1, iteration - 1});
    for (size_t kk = iteration; kk > 1; --kk)
    {
      const size_t k = kk - 1;
      auto t = residual_norms({k - 1});
      for (size_t j = k + 1; j <= iteration; ++j)
      {
        t -= hessenberg_matrix({k - 1, j - 1}) * residual_norms({j - 1});
      }
      residual_norms({k - 1}) = t / hessenberg_matrix({k - 1, k - 1});
    }

    BOBA_CALI_SWITCH("back_substitution", "linear_combination");

    for (size_t j = 0; j < iteration; ++j)
    {
      x_out += residual_norms({j}) * krylov_basis.at(j);
      x_out.round();
    }

    BOBA_CALI_END("linear_combination");

    return converged;
  }

  /**
   * \brief Solve \f$A x = b\f$ via a projection-based GMRES method.
   *
   * \param[in] rhs Right hand side \f$b\f$
   * \param[in] x_in Initial guess \f$x_0\f$
   * \param[in] x_out Final answer \f$x\f$
   *
   * \return `true` if the inner iteration converged, `false` otherwise.
   */
  bool nonorthogonal_gmres_inner_iteration_solve(vector_t const& rhs, vector_t const& x_in, vector_t& x_out)
  {
    BOBA_CALI_MARK
    data_t acceptance_reduction = static_cast<data_t>(0.9);

    BOBA_CALI_BEGIN("setup");

    if (!(acceptance_reduction >= static_cast<data_t>(0) &&
          acceptance_reduction < static_cast<data_t>(1)))
    {
      throw std::invalid_argument("gmres: acceptance_reduction must be in [0,1).");
    }

    // Initialize working variables based on the input
    vector_t x = x_in;
    vector_t b = rhs;

    // Storage for the absolute norm of the residual
    std::vector<data_t> residual_norms;
    residual_norms.reserve(inner_iterations + 1);

    // Storage for the various trial and test spaces in the inner iteration
    std::vector<vector_t> V;
    std::vector<vector_t> right_preconditioner_result;
    std::vector<vector_t> left_preconditioned_rhs;

    V.reserve(inner_iterations + 1);
    right_preconditioner_result.reserve(inner_iterations + 1);
    left_preconditioned_rhs.reserve(inner_iterations + 1);

    // Apply left preconditioning to b and store its norm for later use
    vector_t b_tilde = left_preconditioner_set ? left_preconditioner * b : b;
    b_tilde.orthogonalize();
    auto norm_b_tilde = ::boba::norm_frobenius(b_tilde);

    vector_t r = compute_residual(x, b);
    r.round();
    r.orthogonalize();
    auto norm_r = ::boba::norm_frobenius(r);
    residual_norms.push_back(norm_r);

    // Check if the initial guess was good enough
    if ((norm_r <= absolute_threshold) or (norm_r <= relative_threshold * norm_b_tilde))
    {
      x_out = x;
      return true;
    }

    // Counters and convergence flag for diagnostics
    size_t iters = 0;
    bool converged = false;

    // Declarations for variables reused within the inner loop iterations
    // We define z := M_right(v) and w := M_left(A(z)) to avoid redundant evaluations
    vector_t z;
    vector_t w;

    // Normalize the current residual and seed the basis for the trial space
    V.push_back((1 / norm_r) * r);

    // Record the best candidate found during inner iterations
    // and initialize this using the current solution
    vector_t x_best = x;
    data_t best_rproj_norm = norm_r;

    // Flag to check for issues with inner residual reduction
    bool terminate_inner_phase = false;

    BOBA_CALI_SWITCH("setup", "inner_iterations");

    // Inner enrichment of the trial/test spaces (Arnoldi-like)
    for (size_t m = 0; m < inner_iterations; ++m)
    {
      z = right_preconditioner_set ? right_preconditioner * V[m] : V[m];
      z.orthogonalize();

      auto Az = linear_operator * z;
      w = left_preconditioner_set ? left_preconditioner * Az : Az;
      w.orthogonalize();

      right_preconditioner_result.push_back(z);
      left_preconditioned_rhs.push_back(w);

      // Solve (V^T V) alpha = V^T w (on the host)
      boba::Matrix<host_space, data_t> gramian({m + 1, m + 1});
      boba::Vector<host_space, data_t> proj_rhs({m + 1});

      for (size_t i = 0; i < m + 1; ++i)
      {
        proj_rhs({i}) = inner_product(V[i], w);

        for (size_t j = 0; j < m + 1; ++j)
        {
          gramian({i, j}) = inner_product(V[i], V[j]);
        }
      }

      auto alpha = boba::backsolve(gramian, proj_rhs);

      // Compute the new trial vector: v = w - V*alpha
      // This step removes the projection of w onto the basis V
      // at least approximately
      vector_t v_new = w;

      for (size_t i = 0; i < m + 1; ++i)
      {
        v_new -= alpha({i}) * V[i];
        v_new.round();
      }

      v_new.orthogonalize();
      const data_t v_norm = ::boba::norm_frobenius(v_new);

      V.push_back((1.0 / v_norm) * v_new);

      // Solve (W^T W) y = W^T r (on the host)
      // W = left_preconditioned_rhs
      for (size_t i = 0; i < m + 1; ++i)
      {
        proj_rhs({i}) = inner_product(left_preconditioned_rhs[i], r);

        for (size_t j = 0; j < m + 1; ++j)
        {
          gramian({i, j}) = inner_product(left_preconditioned_rhs[i], left_preconditioned_rhs[j]);
        }
      }

      auto y = boba::backsolve(gramian, proj_rhs);

      // Compute the projected residual: r_proj = r - left_preconditioned_rhs * y
      // This step removes parts of the residual that overlap with the current test space
      vector_t r_proj = r;

      for (size_t i = 0; i < m + 1; ++i)
      {
        r_proj -= y({i}) * left_preconditioned_rhs[i];
        r_proj.round();
      }

      r_proj.orthogonalize();
      const data_t r_proj_norm = ::boba::norm_frobenius(r_proj);

      // Compute the updated solution candidate: x_new = x +  right_preconditioner_result * y
      vector_t x_new = x;
      for (size_t i = 0; i < m + 1; ++i)
      {
        x_new += y({i}) * right_preconditioner_result[i];
        x_new.round();
      }

      // Update the best available solution, if it produces a lower residual
      if (r_proj_norm < best_rproj_norm)
      {
        best_rproj_norm = r_proj_norm;
        x_best = x_new;
      }

      // Terminate the inner loop (i.e., restart) if the achieved reduction is sufficient
      if (r_proj_norm < (1 - acceptance_reduction) * norm_r)
      {
        terminate_inner_phase = true;
        break;
      }
    }

    BOBA_CALI_SWITCH("inner_iterations", "update_solution");

    // Update the residual using the best available solution
    x = x_best;
    r = compute_residual(x, b_tilde);
    r.round();
    r.orthogonalize();
    data_t norm_r_new = ::boba::norm_frobenius(r);
    residual_norms.push_back(norm_r_new);

    iters++;
    used_inner_iterations++;

    if (verbose)
    {
      if (norm_r_new >= norm_r)
      {
        boba_warn("gmres: residual did not decrease.");
      }
      if (!terminate_inner_phase)
      {
        boba_warn("gmres: inner termination criterion not met; accepted best inner candidate.");
      }
      std::cout << "Iteration : " << iters << std::endl;
      std::cout << "Residual (abs) : " << std::scientific << norm_r_new << std::endl;
      std::cout << "Residual (rel) : " << std::scientific << norm_r_new / norm_b_tilde << std::endl;
    }

    // Update the current norm_r for the next pass
    norm_r = norm_r_new;

    // Break from the (outer) iteration if we meet the convergence criteria
    if (norm_r_new <= absolute_threshold || norm_r_new <= relative_threshold * norm_b_tilde)
    {
      converged = true;
    }

    x_out = x;

    BOBA_CALI_END("update_solution");

    return converged;
  }
};

} // namespace boba
