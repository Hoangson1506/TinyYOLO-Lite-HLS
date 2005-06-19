#ifndef YOLO_UTILS_H
#define YOLO_UTILS_H

#include <ap_fixed.h>
#include <algorithm>

typedef ap_fixed<16, 6> data_t; 
typedef ap_fixed<32, 12> mac_t;

#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 0

template<
    int Ci,
    int Hi,
    int Wi,
    int Co,
    int Ho,
    int Wo,
    int K,
    int S,
    int P,
    bool apply_relu
>
void conv2d_1(
    const data_t input[Ci][Hi][Wi],
    const data_t weights[Co][Ci][K][K],
    const data_t bias[Co],
    data_t output[Co][Ho][Wo]
) {
	mac_t acc;

	Loop_Co: for (int co = 0; co < Co; co++) {
		Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
			Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
				Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
					Loop_Kh: for (int kh = 0; kh < K; kh++) {
						Loop_Kw: for (int kw = 0; kw < K; kw++) {

							if (ci == 0 && kh == 0 && kw == 0) {
								acc = bias[co];
							}

							int hi = ho * S + kh - P;
							int wi = wo * S + kw - P;

							acc += input[ci][hi][wi] * weights[co][ci][kh][kw];

							if (ci == Ci - 1 && kh == K - 1 && kw == K - 1) {
								if (apply_relu) {
									output[co][ho][wo] = (acc > 0) ? (data_t)acc : (data_t)0;
								} else {
									output[co][ho][wo] = (data_t)acc;
								}
							}
						}
					}
				}
			}
		}
	}
};

template<
    int Ci,
    int Hi,
    int Wi,
    int Co,
    int Ho,
    int Wo,
    int K,
    int S,
    int P,
    bool apply_relu
>
void conv2d_2(
    const data_t input[Ci][Hi][Wi],
    const data_t weights[Co][Ci][K][K],
    const data_t bias[Co],
    data_t output[Co][Ho][Wo]
) {
	mac_t acc;

	Loop_Co: for (int co = 0; co < Co; co++) {
		Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
			Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
				Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
					Loop_Kh: for (int kh = 0; kh < K; kh++) {
						Loop_Kw: for (int kw = 0; kw < K; kw++) {

							if (ci == 0 && kh == 0 && kw == 0) {
								acc = bias[co];
							}

							int hi = ho * S + kh - P;
							int wi = wo * S + kw - P;

							acc += input[ci][hi][wi] * weights[co][ci][kh][kw];

							if (ci == Ci - 1 && kh == K - 1 && kw == K - 1) {
								if (apply_relu) {
									output[co][ho][wo] = (acc > 0) ? (data_t)acc : (data_t)0;
								} else {
									output[co][ho][wo] = (data_t)acc;
								}
							}
						}
					}
				}
			}
		}
	}
};

template<
    int Ci,
    int Hi,
    int Wi,
    int Co,
    int Ho,
    int Wo,
    int K,
    int S,
    int P,
    bool apply_relu
>
void conv2d_3(
    const data_t input[Ci][Hi][Wi],
    const data_t weights[Co][Ci][K][K],
    const data_t bias[Co],
    data_t output[Co][Ho][Wo]
) {
	mac_t acc = 0;

    Loop_Co: for (int co = 0; co < Co; co++) {
        Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
            Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
                Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
                	Loop_Kh: for (int kh = 0; kh < K; kh++) {
                		Loop_Kw: for (int kw = 0; kw < K; kw++) {

                			int hi = (ho * S) + (kh - P);
							int wi = (wo * S) + (kw - P);

							acc += input[ci][hi][wi] * weights[co][ci][kh][kw];

							if (ci == Ci - 1 && kh == K - 1 && kw == K - 1) {
								acc += bias[co];

								if (apply_relu) {
									output[co][ho][wo] = (acc > 0) ? (data_t)acc : (data_t)0;
								} else {
									output[co][ho][wo] = (data_t)acc;
								}

								acc = 0;
							}
                		}
					}
				}
            }
        }
    }
};

template<
    int Ci,
    int Hi,
    int Wi,
    int Co,
    int Ho,
    int Wo,
    int K,
    int S,
    int P,
    bool apply_relu
>
void conv2d_head(
	const data_t input[Ci][Hi][Wi],
	const data_t weights[Co][Ci][K][K],
	const data_t bias[Co],
	data_t output[Co][Ho][Wo]
) {
	mac_t acc;

	Loop_Co: for (int co = 0; co < Co; co++) {
		Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
			Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
				Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
					Loop_Kh: for (int kh = 0; kh < K; kh++) {
						Loop_Kw: for (int kw = 0; kw < K; kw++) {

							if (ci == 0 && kh == 0 && kw == 0) {
								acc = bias[co];
							}

							int hi = ho * S + kh - P;
							int wi = wo * S + kw - P;

							acc += input[ci][hi][wi] * weights[co][ci][kh][kw];

							if (ci == Ci - 1 && kh == K - 1 && kw == K - 1) {
								if (apply_relu) {
									output[co][ho][wo] = (acc > 0) ? (data_t)acc : (data_t)0;
								} else {
									output[co][ho][wo] = (data_t)acc;
								}
							}
						}
					}
				}
			}
		}
	}
};

template<
    int Ci,
    int Hi,
    int Wi,
    int Ho,
    int Wo,
    int K,
    int S,
    int P
>
void maxpool2d(
    const data_t input[Ci][Hi][Wi],
    data_t output[Ci][Ho][Wo]
) {
    data_t max_val;

    Loop_Ci: for (int ci = 0; ci < Ci; ci++) {
        Loop_Ho: for (int ho = 0; ho < Ho; ho++) {
            Loop_Wo: for (int wo = 0; wo < Wo; wo++) {
            	Loop_Kh: for (int kh = 0; kh < K; kh++) {
            		Loop_Kw: for (int kw = 0; kw < K; kw++) {

            			int hi = ho * S + kh - P;
						int wi = wo * S + kw - P;

						data_t val = input[ci][hi][wi];

						if (kh == 0 && kw == 0) {
							max_val = val;
						} else {
							if (val > max_val) {
								max_val = val;
							}
						}

						if (kh == K - 1 && kw == K - 1) {
							output[ci][ho][wo] = max_val;
						}
            		}
                }
            }
        }
    }
};

#endif
