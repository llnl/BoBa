# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

BOBA_TOP_DIR=.

TESTS_DIR=examples/tests
EXAMPLES_DIR=examples/exercises
TUTOR_DIR=examples/tutorials
CLANG_FORMAT ?= clang-format

STYLE_SOURCES := $(shell find include source examples cmake -type f \( \
	-name '*.c' -o \
	-name '*.cc' -o \
	-name '*.cpp' -o \
	-name '*.cxx' -o \
	-name '*.h' -o \
	-name '*.hh' -o \
	-name '*.hpp' -o \
	-name '*.hxx' -o \
	-name '*.cu' -o \
	-name '*.cuh' \
\))

include ${BOBA_TOP_DIR}/Makefile_boba

###############################

.PHONY: all clean cleanfiles install style style-check

install:
	mkdir -p ${PREFIX} && /bin/cp -Rp include ${PREFIX}

clean:
	rm -f *.o *.out *.d
	rm -f ${TUTOR_DIR}/tutorial_mat_file/*.o ${TUTOR_DIR}/tutorial_mat_file/*.out ${TUTOR_DIR}/tutorial_mat_file/*.d

cleanfiles: clean
	rm -rf file_* *core

style:
	${CLANG_FORMAT} -i ${STYLE_SOURCES}

style-check:
	${CLANG_FORMAT} --dry-run --Werror ${STYLE_SOURCES}

all: \
	boba${NAME_FLAG}.o \
	tutorial_inputs_and_debugging \
	tutorial_0_objects_loops \
	tutorial_1_tensors \
	tutorial_2_linear_algebra \
	tutorial_3_tensor_trains \
	tutorial_4_tucker \
	tutorial_5_dimension_trees \
	tutorial_6_htucker \
	tutorial_7_quantized_tensor_trains \
	tutorial_8_canonical_polyadic \
	test_boba_linear_algebra \
	test_nnmf \
	test_nnls \
	test_ttm_norm \
	test_tt_solvers \
	test_boba_tensor_train \
	test_simplicial_multiindexer \
	test_tt_pow \
	test_quantized_tensor_train \
	test_qtt_search \
	test_qtt_fft \
	test_qtt_support_functions \
	test_tensor_train_matrix \
	test_objects \
	test_tensor_functions \
	test_folding \
	test_backsolve \
	test_argparse \
	test_fft \
	test_abstractions \
	test_krylov \
	test_preconditioner \
	test_tensor_train_matrix \
	test_quantized_tensor_train_matrix \
	test_tensors \
	test_subtensorviews \
	test_cp \
	test_tensor_completion \
	test_cross \
	test_cur \
	test_block_operator \
	test_amen_block \
	test_io \
	test_orthogonalize \
	test_sum_and_round \
	test_static_views \
	example_explicit \
	example_explicit_burgers \
	example_implicit \
	example_implicit_block \
	example_eigel

#######################################################
# Tutorials
#######################################################
tutorial_inputs_and_debugging: tutorial_inputs_and_debugging${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_inputs_and_debugging${NAME_FLAG}.out: tutorial_inputs_and_debugging${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_inputs_and_debugging${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_inputs_and_debugging.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_inputs_and_debugging.cpp -o tutorial_inputs_and_debugging${NAME_FLAG}.o

#######################################################
tutorial_0_objects_loops: tutorial_0_objects_loops${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_0_objects_loops${NAME_FLAG}.out: tutorial_0_objects_loops${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_0_objects_loops${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_0_objects_loops.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_0_objects_loops.cpp -o tutorial_0_objects_loops${NAME_FLAG}.o

#######################################################
tutorial_1_tensors: tutorial_1_tensors${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_1_tensors${NAME_FLAG}.out: tutorial_1_tensors${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_1_tensors${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_1_tensors.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_1_tensors.cpp -o tutorial_1_tensors${NAME_FLAG}.o

#######################################################
tutorial_2_linear_algebra: tutorial_2_linear_algebra${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_2_linear_algebra${NAME_FLAG}.out: tutorial_2_linear_algebra${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_2_linear_algebra${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_2_linear_algebra.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_2_linear_algebra.cpp -o tutorial_2_linear_algebra${NAME_FLAG}.o


#######################################################
tutorial_3_tensor_trains: tutorial_3_tensor_trains${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_3_tensor_trains${NAME_FLAG}.out: tutorial_3_tensor_trains${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_3_tensor_trains${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_3_tensor_trains.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_3_tensor_trains.cpp -o tutorial_3_tensor_trains${NAME_FLAG}.o

#######################################################
tutorial_4_tucker: tutorial_4_tucker${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_4_tucker${NAME_FLAG}.out: tutorial_4_tucker${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_4_tucker${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_4_tucker.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_4_tucker.cpp -o tutorial_4_tucker${NAME_FLAG}.o

#######################################################
tutorial_8_canonical_polyadic: tutorial_8_canonical_polyadic${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_8_canonical_polyadic${NAME_FLAG}.out: tutorial_8_canonical_polyadic${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_8_canonical_polyadic${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_8_canonical_polyadic.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_8_canonical_polyadic.cpp -o tutorial_8_canonical_polyadic${NAME_FLAG}.o

#######################################################
tutorial_5_dimension_trees: tutorial_5_dimension_trees${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_5_dimension_trees${NAME_FLAG}.out: tutorial_5_dimension_trees${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_5_dimension_trees${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_5_dimension_trees.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_5_dimension_trees.cpp -o tutorial_5_dimension_trees${NAME_FLAG}.o


#######################################################
tutorial_6_htucker: tutorial_6_htucker${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_6_htucker${NAME_FLAG}.out: tutorial_6_htucker${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_6_htucker${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_6_htucker.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_6_htucker.cpp -o tutorial_6_htucker${NAME_FLAG}.o

#######################################################
tutorial_7_quantized_tensor_trains: tutorial_7_quantized_tensor_trains${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_7_quantized_tensor_trains${NAME_FLAG}.out: tutorial_7_quantized_tensor_trains${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_7_quantized_tensor_trains${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_7_quantized_tensor_trains.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_7_quantized_tensor_trains.cpp -o tutorial_7_quantized_tensor_trains${NAME_FLAG}.o

###############################
ifdef BOBA_MATLAB
tutorial_mat_file: ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.out
	echo "Done making ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.out"

${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.out: ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.o: ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects.cpp -o ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.o
else
tutorial_mat_file:
	echo "Skipping ${TUTOR_DIR}/tutorial_mat_file/step_2_modify_objects${NAME_FLAG}.out"
endif

#######################################################
# Tests
#######################################################
test_objects: test_objects${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_objects${NAME_FLAG}.out: test_objects${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_objects${NAME_FLAG}.o: ${TESTS_DIR}/test_objects.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_objects.cpp -o test_objects${NAME_FLAG}.o

###############################
test_tensor_functions: test_tensor_functions${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_tensor_functions${NAME_FLAG}.out: test_tensor_functions${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_tensor_functions${NAME_FLAG}.o: ${TESTS_DIR}/test_tensor_functions.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_tensor_functions.cpp -o test_tensor_functions${NAME_FLAG}.o

###############################
test_folding: test_folding${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_folding${NAME_FLAG}.out: test_folding${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_folding${NAME_FLAG}.o: ${TESTS_DIR}/test_folding.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_folding.cpp -o test_folding${NAME_FLAG}.o

###############################
test_ttm_norm: test_ttm_norm${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_ttm_norm${NAME_FLAG}.out: test_ttm_norm${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_ttm_norm${NAME_FLAG}.o: ${TESTS_DIR}/test_ttm_norm.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_ttm_norm.cpp -o test_ttm_norm${NAME_FLAG}.o

###############################
test_tt_solvers: test_tt_solvers${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_tt_solvers${NAME_FLAG}.out: test_tt_solvers${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_tt_solvers${NAME_FLAG}.o: ${TESTS_DIR}/test_tt_solvers.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_tt_solvers.cpp -o test_tt_solvers${NAME_FLAG}.o
###############################
test_backsolve: test_backsolve${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_backsolve${NAME_FLAG}.out: test_backsolve${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_backsolve${NAME_FLAG}.o: ${TESTS_DIR}/test_backsolve.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_backsolve.cpp -o test_backsolve${NAME_FLAG}.o

###############################
test_argparse: test_argparse${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_argparse${NAME_FLAG}.out: test_argparse${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_argparse${NAME_FLAG}.o: ${TESTS_DIR}/test_argparse.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_argparse.cpp -o test_argparse${NAME_FLAG}.o

###############################
test_abstractions: test_abstractions${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_abstractions${NAME_FLAG}.out: test_abstractions${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_abstractions${NAME_FLAG}.o: ${TESTS_DIR}/test_abstractions.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_abstractions.cpp -o test_abstractions${NAME_FLAG}.o

###############################
test_boba_linear_algebra: test_boba_linear_algebra${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_boba_linear_algebra${NAME_FLAG}.out: test_boba_linear_algebra${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_boba_linear_algebra${NAME_FLAG}.o: ${TESTS_DIR}/test_boba_linear_algebra.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_boba_linear_algebra.cpp -o test_boba_linear_algebra${NAME_FLAG}.o

###############################
test_nnmf: test_nnmf${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_nnmf${NAME_FLAG}.out: test_nnmf${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_nnmf${NAME_FLAG}.o: ${TESTS_DIR}/test_nnmf.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_nnmf.cpp -o test_nnmf${NAME_FLAG}.o

###############################
test_nnls: test_nnls${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_nnls${NAME_FLAG}.out: test_nnls${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_nnls${NAME_FLAG}.o: ${TESTS_DIR}/test_nnls.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_nnls.cpp -o test_nnls${NAME_FLAG}.o

###############################
test_boba_tensor_train: test_boba_tensor_train${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_boba_tensor_train${NAME_FLAG}.out: test_boba_tensor_train${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_boba_tensor_train${NAME_FLAG}.o: ${TESTS_DIR}/test_boba_tensor_train.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_boba_tensor_train.cpp -o test_boba_tensor_train${NAME_FLAG}.o

###############################
test_simplicial_multiindexer: test_simplicial_multiindexer${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_simplicial_multiindexer${NAME_FLAG}.out: test_simplicial_multiindexer${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_simplicial_multiindexer${NAME_FLAG}.o: ${TESTS_DIR}/test_simplicial_multiindexer.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_simplicial_multiindexer.cpp -o test_simplicial_multiindexer${NAME_FLAG}.o

###############################
test_tt_pow: test_tt_pow${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_tt_pow${NAME_FLAG}.out: test_tt_pow${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_tt_pow${NAME_FLAG}.o: ${TESTS_DIR}/test_tt_pow.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_tt_pow.cpp -o test_tt_pow${NAME_FLAG}.o

###############################
test_quantized_tensor_train: test_quantized_tensor_train${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_quantized_tensor_train${NAME_FLAG}.out: test_quantized_tensor_train${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_quantized_tensor_train${NAME_FLAG}.o: ${TESTS_DIR}/test_quantized_tensor_train.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_quantized_tensor_train.cpp -o test_quantized_tensor_train${NAME_FLAG}.o
###############################
ifdef BOBA_CPU
test_cur: test_cur${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_cur${NAME_FLAG}.out: test_cur${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_cur${NAME_FLAG}.o: ${TESTS_DIR}/test_cur.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_cur.cpp -o test_cur${NAME_FLAG}.o
else
test_cur:
	echo "Skipping $@${NAME_FLAG}.out"
endif
###############################
test_fft: test_fft${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_fft${NAME_FLAG}.out: test_fft${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_fft${NAME_FLAG}.o: ${TESTS_DIR}/test_fft.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_fft.cpp -o test_fft${NAME_FLAG}.o
###############################
test_qtt_search: test_qtt_search${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_qtt_search${NAME_FLAG}.out: test_qtt_search${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_qtt_search${NAME_FLAG}.o: ${TESTS_DIR}/test_qtt_search.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_qtt_search.cpp -o test_qtt_search${NAME_FLAG}.o
###############################
test_qtt_fft: test_qtt_fft${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_qtt_fft${NAME_FLAG}.out: test_qtt_fft${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_qtt_fft${NAME_FLAG}.o: ${TESTS_DIR}/test_qtt_fft.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_qtt_fft.cpp -o test_qtt_fft${NAME_FLAG}.o
###############################
test_qtt_support_functions: test_qtt_support_functions${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_qtt_support_functions${NAME_FLAG}.out: test_qtt_support_functions${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_qtt_support_functions${NAME_FLAG}.o: ${TESTS_DIR}/test_qtt_support_functions.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_qtt_support_functions.cpp -o test_qtt_support_functions${NAME_FLAG}.o
###############################
test_tensor_train_matrix: test_tensor_train_matrix${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_tensor_train_matrix${NAME_FLAG}.out: test_tensor_train_matrix${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_tensor_train_matrix${NAME_FLAG}.o: ${TESTS_DIR}/test_tensor_train_matrix.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_tensor_train_matrix.cpp -o test_tensor_train_matrix${NAME_FLAG}.o

###############################
test_quantized_tensor_train_matrix: test_quantized_tensor_train_matrix${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_quantized_tensor_train_matrix${NAME_FLAG}.out: test_quantized_tensor_train_matrix${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_quantized_tensor_train_matrix${NAME_FLAG}.o: ${TESTS_DIR}/test_quantized_tensor_train_matrix.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_quantized_tensor_train_matrix.cpp -o test_quantized_tensor_train_matrix${NAME_FLAG}.o

###############################
test_krylov: test_krylov${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_krylov${NAME_FLAG}.out: test_krylov${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_krylov${NAME_FLAG}.o: ${TESTS_DIR}/test_krylov.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_krylov.cpp -o test_krylov${NAME_FLAG}.o

#######################################################
test_preconditioner: test_preconditioner${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_preconditioner${NAME_FLAG}.out: test_preconditioner${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_preconditioner${NAME_FLAG}.o: ${TESTS_DIR}/test_preconditioner.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_preconditioner.cpp -o test_preconditioner${NAME_FLAG}.o

###############################
test_tensors: test_tensors${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_tensors${NAME_FLAG}.out: test_tensors${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_tensors${NAME_FLAG}.o: ${TESTS_DIR}/test_tensors.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_tensors.cpp -o test_tensors${NAME_FLAG}.o

###############################
test_cp: test_cp${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_cp${NAME_FLAG}.out: test_cp${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_cp${NAME_FLAG}.o: ${TESTS_DIR}/test_cp.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_cp.cpp -o test_cp${NAME_FLAG}.o

###############################
test_tensor_completion: test_tensor_completion${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_tensor_completion${NAME_FLAG}.out: test_tensor_completion${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_tensor_completion${NAME_FLAG}.o: ${TESTS_DIR}/test_tensor_completion.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_tensor_completion.cpp -o test_tensor_completion${NAME_FLAG}.o

###############################
test_orthogonalize: test_orthogonalize${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_orthogonalize${NAME_FLAG}.out: test_orthogonalize${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_orthogonalize${NAME_FLAG}.o: ${TESTS_DIR}/test_orthogonalize.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_orthogonalize.cpp -o test_orthogonalize${NAME_FLAG}.o

###############################
ifdef BOBA_CPU
test_cross: test_cross${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_cross${NAME_FLAG}.out: test_cross${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_cross${NAME_FLAG}.o: ${TESTS_DIR}/test_cross.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_cross.cpp -o test_cross${NAME_FLAG}.o
else
test_cross:
	echo "Skipping $@${NAME_FLAG}.out"
endif
###############################
test_block_operator: test_block_operator${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_block_operator${NAME_FLAG}.out: test_block_operator${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_block_operator${NAME_FLAG}.o: ${TESTS_DIR}/test_block_operator.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_block_operator.cpp -o test_block_operator${NAME_FLAG}.o
###############################
test_amen_block: test_amen_block${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_amen_block${NAME_FLAG}.out: test_amen_block${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_amen_block${NAME_FLAG}.o: ${TESTS_DIR}/test_amen_block.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_amen_block.cpp -o test_amen_block${NAME_FLAG}.o
###############################
test_subtensorviews: test_subtensorviews${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_subtensorviews${NAME_FLAG}.out: test_subtensorviews${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_subtensorviews${NAME_FLAG}.o: ${TESTS_DIR}/test_subtensorviews.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_subtensorviews.cpp -o test_subtensorviews${NAME_FLAG}.o
###############################
test_io: test_io${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_io${NAME_FLAG}.out: test_io${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_io${NAME_FLAG}.o: ${TESTS_DIR}/test_io.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_io.cpp -o test_io${NAME_FLAG}.o
###############################
test_sum_and_round: test_sum_and_round${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_sum_and_round${NAME_FLAG}.out: test_sum_and_round${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_sum_and_round${NAME_FLAG}.o: ${TESTS_DIR}/test_sum_and_round.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_sum_and_round.cpp -o test_sum_and_round${NAME_FLAG}.o
###############################
test_static_views: test_static_views${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

test_static_views${NAME_FLAG}.out: test_static_views${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

test_static_views${NAME_FLAG}.o: ${TESTS_DIR}/test_static_views.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${TESTS_DIR}/test_static_views.cpp -o test_static_views${NAME_FLAG}.o
###############################
# Examples
#######################################################
example_explicit: example_explicit${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_explicit${NAME_FLAG}.out: example_explicit${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_explicit${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_explicit/example_explicit.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${EXAMPLES_DIR}/example_explicit/example_explicit.cpp -o example_explicit${NAME_FLAG}.o

###############################
ifdef BOBA_CPU
example_explicit_burgers: example_explicit_burgers${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_explicit_burgers${NAME_FLAG}.out: example_explicit_burgers${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_explicit_burgers${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_explicit_burgers.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${EXAMPLES_DIR}/example_explicit_burgers.cpp -o example_explicit_burgers${NAME_FLAG}.o
else
example_explicit_burgers:
	echo "Skipping $@${NAME_FLAG}.out"
endif
###############################
example_implicit: example_implicit${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_implicit${NAME_FLAG}.out: example_implicit${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_implicit${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_implicit.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${EXAMPLES_DIR}/example_implicit.cpp -o example_implicit${NAME_FLAG}.o

###############################
example_implicit_block: example_implicit_block${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_implicit_block${NAME_FLAG}.out: example_implicit_block${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_implicit_block${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_implicit_block/example_implicit_block.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${EXAMPLES_DIR}/example_implicit_block/example_implicit_block.cpp -o example_implicit_block${NAME_FLAG}.o

###############################
example_eigel: example_eigel${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_eigel${NAME_FLAG}.out: example_eigel${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_eigel${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_eigel/main.cpp ${BOBA_INC}
	${COMPILE} ${OPTS} -c ${EXAMPLES_DIR}/example_eigel/main.cpp -o example_eigel${NAME_FLAG}.o

###############################
ifdef BOBA_ENABLE_MPI
example_permutation_finder_parallel: example_permutation_finder_parallel${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_permutation_finder_parallel${NAME_FLAG}.out: example_permutation_finder_parallel${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_permutation_finder_parallel${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_permutation_finder_parallel.cpp ${BOBA_INC} boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} -c ${EXAMPLES_DIR}/example_permutation_finder_parallel.cpp -o example_permutation_finder_parallel${NAME_FLAG}.o
else
example_permutation_finder_parallel:
	echo "Skipping $@${NAME_FLAG}.out"
endif

DEPFILES := $(wildcard *.d) $(wildcard ${TUTOR_DIR}/tutorial_mat_file/*.d)
-include ${DEPFILES}
