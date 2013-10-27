#pragma once

#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <fftw3.h>
#include <boost/tr1/complex.hpp>
#include "sht/makeweights.h"
#include "sht/FST_semi_memo.h"
#include "sht/cospmls.h"


namespace trajopt
{

// spherical harmonics transform with complex input
std::vector<std::vector<std::complex<double> > > SHT(const std::vector<std::vector<std::complex<double> > >& input);

// spherical harmonics transform with real input
std::vector<std::vector<std::complex<double> > > SHT(const std::vector<std::vector<double> >& input);

// inverse spherical harmonics transform
std::vector<std::vector<std::complex<double> > > Inv_SHT(const std::vector<std::vector<std::complex<double> > >& input);

  
std::vector<double> collectGlobalSHTFeature(const std::vector<std::vector<std::complex<double> > >& sht_coeffs, int eff_bw);

std::vector<double> collectShapeSHTFeature(const std::vector<std::vector<std::complex<double> > >& sht_coeffs, int eff_bw);

// generate S^2 sequence
std::vector<double> S2_sequence(int n);


}
