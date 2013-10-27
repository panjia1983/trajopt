add_library(sht cospmls.cpp FST_semi_fly.cpp FST_semi_memo.cpp makeweights.cpp naive_synthesis.cpp pmls.cpp primitive.cpp seminaive.cpp)
target_link_libraries(sht fftw3)