typedef struct
{
    float data[8];
} packed_8;

typedef struct
{
    packed_8 *real;
    packed_8 *imag;
} packed_8_split;

typedef struct
{
    float *real;
    float *imag;
} split;

typedef struct
{
    float real;
    float imag;
} complex;

typedef struct
{
    complex data[8];
} packed_8_interleave;

void packed_8_split_matrix_multiply (const packed_8_split *const weights,
                                     const split *const samples,
                                     const split *const outputs,
                                     const size_t num_samples,
                                     const size_t num_sets);

void packed_8_interleave_matrix_multiply (const packed_8_interleave *const  weights,
                                          const complex *const *const samples,
                                          complex *const *const outputs,
                                          const size_t num_samples,
                                          const size_t num_sets);

void packed_8_interleave_matrix_multiply_no_alias (const packed_8_interleave *const  weights,
                                          const complex *const *const samples,
                                          complex *const *const outputs,
                                          const size_t num_samples,
                                          const size_t num_sets);
