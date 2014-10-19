SAL_i32 cmat_mulx_avx_nr_c_5_dot_product_length_8 (SAL_cf32 *A, 	/* left input matrix */
  	                                               SAL_i32 A_tcols, 	/* left input column stride */
  	                                               SAL_cf32 *B, 	/* right input matrix */
  	                                               SAL_i32 B_tcols, 	/* right input column stride */
  	                                               SAL_cf32 *C, 	/* output matrix */
  	                                               SAL_i32 C_tcols, 	/* output column stride */
  	                                               SAL_i32 nc_c); 	/* column count in C */
                                                   