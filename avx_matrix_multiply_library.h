
#define NR_C_MAX 20

SAL_i32 cmat_mulx_avx_dot_product_length_8 (SAL_cf32 *A, 	/* left input matrix */
  	                                        SAL_i32 A_tcols, 	/* left input column stride */
  	                                        SAL_cf32 *B, 	/* right input matrix */
  	                                        SAL_i32 B_tcols, 	/* right input column stride */
  	                                        SAL_cf32 *C, 	/* output matrix */
  	                                        SAL_i32 nr_c, 	/* rows count in C */
                                            SAL_i32 C_tcols, 	/* output column stride */
  	                                        SAL_i32 nc_c); 	/* column count in C */

SAL_i32 zmat_mulx_avx_dot_product_length_8 (SAL_zf32 *A, 	/* left input matrix */
  	                                        SAL_i32 A_tcols, 	/* left input column stride */
  	                                        SAL_zf32 *B, 	/* right input matrix */
  	                                        SAL_i32 B_tcols, 	/* right input column stride */
  	                                        SAL_zf32 *C, 	/* output matrix */
  	                                        SAL_i32 nr_c, 	/* rows count in C */
                                            SAL_i32 C_tcols, 	/* output column stride */
  	                                        SAL_i32 nc_c); 	/* column count in C */

SAL_i32 cmat_mulx_avx_fma_dot_product_length_8 (SAL_cf32 *A, 	/* left input matrix */
  	                                            SAL_i32 A_tcols, 	/* left input column stride */
  	                                            SAL_cf32 *B, 	/* right input matrix */
  	                                            SAL_i32 B_tcols, 	/* right input column stride */
  	                                            SAL_cf32 *C, 	/* output matrix */
  	                                            SAL_i32 nr_c, 	/* rows count in C */
                                                SAL_i32 C_tcols, 	/* output column stride */
  	                                            SAL_i32 nc_c); 	/* column count in C */
