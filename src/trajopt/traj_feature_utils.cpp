#include "traj_feature_utils.hpp"
#include <cstdio>
#include <cstdlib>

namespace trajopt
{


// spherical harmonics transform
std::vector<std::vector<std::complex<double> > > SHT(const std::vector<std::vector<std::complex<double> > >& input)
{
  int i, bw, size ;
  int l, m, dummy;
  int cutoff, order ;
  int rank, howmany_rank ;
  double *rdata, *idata ;
  double *rcoeffs, *icoeffs ;
  double *weights ;
  double *seminaive_naive_tablespace, *workspace;
  double **seminaive_naive_table ;
  double tstart, tstop;
  fftw_plan dctPlan, fftPlan ;
  fftw_iodim dims[1], howmany_dims[1];

  bw = input.size() / 2;

  /*** ASSUMING WILL SEMINAIVE ALL ORDERS ***/
  cutoff = bw ;
  size = 2*bw;

  /* allocate memory */
  rdata = (double *) malloc(sizeof(double) * (size * size));
  idata = (double *) malloc(sizeof(double) * (size * size));
  rcoeffs = (double *) malloc(sizeof(double) * (bw * bw));
  icoeffs = (double *) malloc(sizeof(double) * (bw * bw));
  weights = (double *) malloc(sizeof(double) * 4 * bw);
  seminaive_naive_tablespace =
    (double *) malloc(sizeof(double) *
                      (Reduced_Naive_TableSize(bw,cutoff) +
                       Reduced_SpharmonicTableSize(bw,cutoff)));
  workspace = (double *) malloc(sizeof(double) * 
                                ((8 * (bw*bw)) + 
                                 (7 * bw)));


  /****
       At this point, check to see if all the memory has been
       allocated. If it has not, there's no point in going further.
  ****/

  if ( (rdata == NULL) || (idata == NULL) ||
       (rcoeffs == NULL) || (icoeffs == NULL) ||
       (seminaive_naive_tablespace == NULL) ||
       (workspace == NULL) )
  {
    std::cerr << "Error in allocating memory" << std::endl;
    return std::vector<std::vector<std::complex<double> > >();
  }

  /* now precompute the Legendres */
  // std::cout << "Generating seminaive_naive tables..." << std::endl;
  seminaive_naive_table = SemiNaive_Naive_Pml_Table(bw, cutoff,
                                                    seminaive_naive_tablespace,
                                                    workspace);

  /* construct fftw plans */

  /* make DCT plan -> note that I will be using the GURU
     interface to execute these plans within the routines*/

  /* forward DCT */
  dctPlan = fftw_plan_r2r_1d( 2*bw, weights, rdata,
                              FFTW_REDFT10, FFTW_ESTIMATE ) ;
      
  /*
    fftw "preamble" ;
    note that this plan places the output in a transposed array
  */
  rank = 1 ;
  dims[0].n = 2*bw ;
  dims[0].is = 1 ;
  dims[0].os = 2*bw ;
  howmany_rank = 1 ;
  howmany_dims[0].n = 2*bw ;
  howmany_dims[0].is = 2*bw ;
  howmany_dims[0].os = 1 ;
  
  /* forward fft */
  fftPlan = fftw_plan_guru_split_dft( rank, dims,
                                      howmany_rank, howmany_dims,
                                      rdata, idata,
                                      workspace, workspace+(4*bw*bw),
                                      FFTW_ESTIMATE );


  /* now make the weights */
  makeweights( bw, weights );


  /* now read in samples */
  for(i = 0; i < size*size; i++)
  {
    int j = i / size;
    int k = i % size;
    *(rdata + i) = input[j][k].real();
    *(idata + i) = input[j][k].imag();
  }

  
  /* now do the forward spherical transform */
  FST_semi_memo(rdata, idata,
                rcoeffs, icoeffs,
                bw,
                seminaive_naive_table,
                workspace,
                0,
                cutoff,
                &dctPlan,
                &fftPlan,
                weights );

  std::vector<std::vector<std::complex<double> > > output(bw);
  for(l = 0; l < bw; l++)
  {
    for(m = -l; m < l + 1; m++)
    {
      dummy = seanindex(m, l, bw);
      output[l].push_back(std::complex<double>(rcoeffs[dummy], icoeffs[dummy]));
    }
  }
  
  /* clean up */
  fftw_destroy_plan( fftPlan );
  fftw_destroy_plan( dctPlan );

  free(workspace);
  free(seminaive_naive_table);
  free(seminaive_naive_tablespace);
  free(weights);
  free(icoeffs);
  free(rcoeffs);
  free(idata);
  free(rdata);

  return output;
  
}

// spherical harmonics transform for real values
std::vector<std::vector<std::complex<double> > > SHT(const std::vector<std::vector<double> >& input_)
{
  std::vector<std::vector<std::complex<double> > > input(input_.size());
  for(std::size_t i = 0; i < input.size(); ++i)
    input[i].resize(input_[i].size());
  for(std::size_t i = 0; i < input.size(); ++i)
  {
    for(std::size_t j = 0; j < input[i].size(); ++j)
    {
      input[i][j] = std::complex<double>(input_[i][j], 0);
    }
  }

  return SHT(input);
}


std::vector<double> collectGlobalSHTFeature(const std::vector<std::vector<std::complex<double> > >& sht_coeffs, int eff_bw)
{
  int bw = sht_coeffs.size();
  if(eff_bw < bw) bw = eff_bw;
  std::vector<double> feature(bw);
  for(int l = 0; l < bw; ++l)
  {
    double dummy = 0;
    for(int m = -l; m < l + 1; ++m)
    {
      double coeff_norm = std::norm(sht_coeffs[l][m+l]);
      dummy += coeff_norm * coeff_norm;
    }

    feature[l] = sqrt(dummy);
  }

  return feature;
}

std::vector<double> collectShapeSHTFeature(const std::vector<std::vector<std::complex<double> > >& sht_coeffs, int eff_bw)
{
  int bw = sht_coeffs.size();
  if(eff_bw < bw) bw = eff_bw;
  std::vector<double> feature;
  for(int l = 0; l < bw; ++l)
  {
    for(int m = 0; m < l + 1; ++m) // ignore the negative one
    {
      double coeff_norm = std::norm(sht_coeffs[l][m+l]);
      feature.push_back(coeff_norm);
    }
  }

  return feature;
}

// inverse spherical harmonics transform
std::vector<std::vector<std::complex<double> > > Inv_SHT(const std::vector<std::vector<std::complex<double> > >& input)
{
  int i, bw, size, cutoff ;
  int rank, howmany_rank ;
  double *rcoeffs, *icoeffs, *rdata, *idata ;
  double *workspace, *weights ;
  double *seminaive_naive_tablespace, *trans_seminaive_naive_tablespace ;
  double **seminaive_naive_table, **trans_seminaive_naive_table;
  double tstart, tstop;
  fftw_plan idctPlan, ifftPlan ;
  fftw_iodim dims[1], howmany_dims[1];

  bw = input.size();
  size = 2*bw;

  /*** ASSUMING WILL SEMINAIVE ALL ORDERS ***/
  cutoff = bw ;

  /* allocate lots of memory */
  rcoeffs = (double *) malloc(sizeof(double) * (bw * bw));
  icoeffs = (double *) malloc(sizeof(double) * (bw * bw));
  rdata = (double *) malloc(sizeof(double) * (size * size));
  idata = (double *) malloc(sizeof(double) * (size * size));
  weights = (double *) malloc(sizeof(double) * 4 * bw);
  
  workspace = (double *) malloc(sizeof(double) * 
				((8 * (bw*bw)) + 
				 (10 * bw)));

  seminaive_naive_tablespace =
    (double *) malloc(sizeof(double) *
		      (Reduced_Naive_TableSize(bw,cutoff) +
		       Reduced_SpharmonicTableSize(bw,cutoff)));

  trans_seminaive_naive_tablespace =
    (double *) malloc(sizeof(double) *
		      (Reduced_Naive_TableSize(bw,cutoff) +
		       Reduced_SpharmonicTableSize(bw,cutoff)));

  /****
       At this point, check to see if all the memory has been
       allocated. If it has not, there's no point in going further.
  ****/

  if ( (rdata == NULL) || (idata == NULL) ||
       (rcoeffs == NULL) || (icoeffs == NULL) ||
       (weights == NULL) ||
       (seminaive_naive_tablespace == NULL) ||
       (trans_seminaive_naive_tablespace == NULL) ||
       (workspace == NULL) )
    {
      std::cerr << "Error in allocating memory" << std::endl;
      return std::vector<std::vector<std::complex<double> > >();
    }

  /* now precompute the Legendres */
  seminaive_naive_table = SemiNaive_Naive_Pml_Table(bw, cutoff,
						    seminaive_naive_tablespace,
						    workspace);

  trans_seminaive_naive_table =
    Transpose_SemiNaive_Naive_Pml_Table(seminaive_naive_table,
					bw, cutoff,
					trans_seminaive_naive_tablespace,
					workspace);

  /* construct fftw plans */

  /* make iDCT plan -> note that I will be using the GURU
     interface to execute this plan within the routine*/
      
  /* inverse DCT */
  idctPlan = fftw_plan_r2r_1d( 2*bw, weights, rdata,
			       FFTW_REDFT01, FFTW_ESTIMATE );

  /*
    now plan for inverse fft - note that this plans assumes
    that I'm working with a transposed array, e.g. the inputs
    for a length 2*bw transform are placed every 2*bw apart,
    the output will be consecutive entries in the array
  */
  rank = 1 ;
  dims[0].n = 2*bw ;
  dims[0].is = 2*bw ;
  dims[0].os = 1 ;
  howmany_rank = 1 ;
  howmany_dims[0].n = 2*bw ;
  howmany_dims[0].is = 1 ;
  howmany_dims[0].os = 2*bw ;

  /* inverse fft */
  ifftPlan = fftw_plan_guru_split_dft( rank, dims,
				       howmany_rank, howmany_dims,
				       rdata, idata,
				       workspace, workspace+(4*bw*bw),
				       FFTW_ESTIMATE );

  /* now make the weights */
  makeweights( bw, weights );

  /* now read in coefficients */
  for(int l = 0; l < bw; l++)
  {
    for(int m = -l; m < l+1; m++)
    {
      int dummy = seanindex(m, l, bw);
      *(rcoeffs + dummy) = input[l][m+l].real();
      *(icoeffs + dummy) = input[l][m+l].imag();
    }
  }
  
  /* do the inverse spherical transform */
  InvFST_semi_memo(rcoeffs,icoeffs,
		   rdata, idata,
		   bw,
		   trans_seminaive_naive_table,
		   workspace,
		   0,
		   cutoff,
		   &idctPlan,
		   &ifftPlan );

  std::vector<std::vector<std::complex<double> > > output(size);
  for(i = 0; i < size; i++)
    output[i].resize(size);

  for(i = 0; i < size* size; i++)
  {
    int j = i / size;
    int k = i % size;
    output[j][k] = std::complex<double>(rdata[i], idata[i]);
    
  }
  
  
  /* now clean up */

  fftw_destroy_plan( ifftPlan );
  fftw_destroy_plan( idctPlan );

  free(trans_seminaive_naive_table);
  free(seminaive_naive_table);
  free(trans_seminaive_naive_tablespace);
  free(seminaive_naive_tablespace);
  free(workspace);
  free(weights);
  free(idata);
  free(rdata);
  free(icoeffs);
  free(rcoeffs);

  return output;
}



void ang2vec(double theta, double phi, double *vec) {

  double sz;
  double PI=M_PI;

  if( theta<0. || theta>PI) {
    //fprintf(stderr, "%s (%d): theta out of range: %f\n", __FILE__, __LINE__, theta);
    exit(0);
  }

  sz = sin(theta);

  vec[0] = sz * cos(phi) ;
  vec[1] = sz * sin(phi) ;
  vec[2] = cos(theta)    ;
}

void mk_pix2xy(int *pix2x, int *pix2y) {

  /* =======================================================================
   * subroutine mk_pix2xy
   * =======================================================================
   * constructs the array giving x and y in the face from pixel number
   * for the nested (quad-cube like) ordering of pixels
   *
   * the bits corresponding to x and y are interleaved in the pixel number
   * one breaks up the pixel number by even and odd bits
   * =======================================================================
   */

  int i, kpix, jpix, IX, IY, IP, ID;
  for (i = 0; i < 1023; i++) pix2x[i]=0;
  
  for( kpix=0;kpix<1024;kpix++ ) {
    jpix = kpix;
    IX = 0;
    IY = 0;
    IP = 1 ;//              ! bit position (in x and y)
    while( jpix!=0 ){// ! go through all the bits
      ID = (int)fmod(jpix,2);//  ! bit value (in kpix), goes in ix
      jpix = jpix/2;
      IX = ID*IP+IX;
      
      ID = (int)fmod(jpix,2);//  ! bit value (in kpix), goes in iy
      jpix = jpix/2;
      IY = ID*IP+IY;
      
      IP = 2*IP;//         ! next bit (in x and y)
    }
    
    pix2x[kpix] = IX;//     ! in 0,31
    pix2y[kpix] = IY;//     ! in 0,31
  }
  
  /* Later */
  return;
}

void pix2ang_nest( long nside, long ipix, double *theta, double *phi) {

  /*
    c=======================================================================
    subroutine pix2ang_nest(nside, ipix, theta, phi)
    c=======================================================================
    c     gives theta and phi corresponding to pixel ipix (NESTED) 
    c     for a parameter nside
    c=======================================================================
  */
    
  int npix, npface, face_num;
  int  ipf, ip_low, ip_trunc, ip_med, ip_hi;
  int     ix, iy, jrt, jr, nr, jpt, jp, kshift, nl4;
  double z, fn, fact1, fact2;
  double piover2=0.5*M_PI;
  int ns_max=8192;
      
  static int pix2x[1024], pix2y[1024];
  //      common /pix2xy/ pix2x, pix2y
      
  int jrll[12], jpll[12];// ! coordinate of the lowest corner of each face
  //      data jrll/2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4/ ! in unit of nside
  //      data jpll/1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7/ ! in unit of nside/2
  jrll[0]=2;
  jrll[1]=2;
  jrll[2]=2;
  jrll[3]=2;
  jrll[4]=3;
  jrll[5]=3;
  jrll[6]=3;
  jrll[7]=3;
  jrll[8]=4;
  jrll[9]=4;
  jrll[10]=4;
  jrll[11]=4;
  jpll[0]=1;
  jpll[1]=3;
  jpll[2]=5;
  jpll[3]=7;
  jpll[4]=0;
  jpll[5]=2;
  jpll[6]=4;
  jpll[7]=6;
  jpll[8]=1;
  jpll[9]=3;
  jpll[10]=5;
  jpll[11]=7;
      
      
  if( nside<1 || nside>ns_max ) {
	fprintf(stderr, "%s (%d): nside out of range: %ld\n", __FILE__, __LINE__, nside);
	exit(0);
  }
  npix = 12 * nside*nside;
  if( ipix<0 || ipix>npix-1 ) {
	fprintf(stderr, "%s (%d): ipix out of range: %ld\n", __FILE__, __LINE__, ipix);
	exit(0);
  }

  /* initiates the array for the pixel number -> (x,y) mapping */
  if( pix2x[1023]<=0 ) mk_pix2xy(pix2x,pix2y);

  fn = 1.*nside;
  fact1 = 1./(3.*fn*fn);
  fact2 = 2./(3.*fn);
  nl4   = 4*nside;

  //c     finds the face, and the number in the face
  npface = nside*nside;

  face_num = ipix/npface;//  ! face number in {0,11}
  ipf = (int)fmod(ipix,npface);//  ! pixel number in the face {0,npface-1}

  //c     finds the x,y on the face (starting from the lowest corner)
  //c     from the pixel number
  ip_low = (int)fmod(ipf,1024);//       ! content of the last 10 bits
  ip_trunc =   ipf/1024 ;//       ! truncation of the last 10 bits
  ip_med = (int)fmod(ip_trunc,1024);//  ! content of the next 10 bits
  ip_hi  =     ip_trunc/1024   ;//! content of the high weight 10 bits

  ix = 1024*pix2x[ip_hi] + 32*pix2x[ip_med] + pix2x[ip_low];
  iy = 1024*pix2y[ip_hi] + 32*pix2y[ip_med] + pix2y[ip_low];

  //c     transforms this in (horizontal, vertical) coordinates
  jrt = ix + iy;//  ! 'vertical' in {0,2*(nside-1)}
  jpt = ix - iy;//  ! 'horizontal' in {-nside+1,nside-1}

  //c     computes the z coordinate on the sphere
  //      jr =  jrll[face_num+1]*nside - jrt - 1;//   ! ring number in {1,4*nside-1}
  jr =  jrll[face_num]*nside - jrt - 1;
  //      cout << "face_num=" << face_num << endl;
  //      cout << "jr = " << jr << endl;
  //      cout << "jrll(face_num)=" << jrll[face_num] << endl;
  //      cout << "----------------------------------------------------" << endl;
  nr = nside;//                  ! equatorial region (the most frequent)
  z  = (2*nside-jr)*fact2;
  kshift = (int)fmod(jr - nside, 2);
  if( jr<nside ) { //then     ! north pole region
    nr = jr;
    z = 1. - nr*nr*fact1;
    kshift = 0;
  }
  else {
	if( jr>3*nside ) {// then ! south pole region
      nr = nl4 - jr;
      z = - 1. + nr*nr*fact1;
      kshift = 0;
	}
  }
  *theta = acos(z);
      
  //c     computes the phi coordinate on the sphere, in [0,2Pi]
  //      jp = (jpll[face_num+1]*nr + jpt + 1 + kshift)/2;//  ! 'phi' number in the ring in {1,4*nr}
  jp = (jpll[face_num]*nr + jpt + 1 + kshift)/2;
  if( jp>nl4 ) jp = jp - nl4;
  if( jp<1 )   jp = jp + nl4;

  *phi = (jp - (kshift+1)*0.5) * (piover2 / nr);

}


std::vector < double >find_point(int base_grid, long int point,long int level,long int healpix_point)
{
  int position=point%4;
  long int quo=0;
  double theta=0,phi=0;
  double vec[3];
  std::vector <double> Point;
  if(base_grid == 6 or base_grid == 7)
  {
    switch(position)//this switch statement translates between sequence of healpix and sequence for uniform points 
    {
    case 0:
      healpix_point+=3;
      break;
    case 1:
      healpix_point+=0;
      break;
    case 2: 
      healpix_point+=2;
      break;
    case 3:
      healpix_point+=1;
      break;
    }	
  }
  else if(base_grid == 3 or base_grid == 1 or base_grid == 9 or base_grid == 11)
  {
    switch(position)//this switch statement translates between sequence of healpix and sequence for uniform points 
    {
    case 0:
      healpix_point+=3;
      break;
    case 1:
      healpix_point+=0;
      break;
    case 2: 
      healpix_point+=1;
      break;
    case 3:
      healpix_point+=2;
      break;
    }	
  }
  else if(base_grid == 2 or base_grid == 0 or base_grid == 8 or base_grid == 10)
  {
    switch(position)//this switch statement translates between sequence of healpix and sequence for uniform points 
    {
    case 0:
      healpix_point+=0;
      break;
    case 1:
      healpix_point+=3;
      break;
    case 2: 
      healpix_point+=1;
      break;
    case 3:
      healpix_point+=2;
      break;
    }	
  }
  else if(base_grid == 4 or base_grid == 5)
  {
    switch(position)//this switch statement translates between sequence of healpix and sequence for uniform points 
    {
    case 0:
      healpix_point+=0;
      break;
    case 1:
      healpix_point+=3;
      break;
    case 2: 
      healpix_point+=2;
      break;
    case 3:
      healpix_point+=1;
      break;
    }	
  }

  quo=point/4;
  if(quo==0)
  {
    long int nside=pow(2,level);
    pix2ang_nest(nside,healpix_point,&theta,&phi);
    ang2vec(theta,phi,vec);
    Point.resize(0);
    Point.push_back(vec[0]);	
    Point.push_back(vec[1]);	
    Point.push_back(vec[2]);
    return Point;
  }
  else
  {
    return find_point(base_grid,quo-1,level+1,4*healpix_point);
  }
}	
		
std::vector<double> S2_sequence(int n)
{
  int basic_sequence[] = {6, 4, 1, 11, 9, 3, 5, 7, 10, 0, 2, 8};

  std::vector<double> result;

  int limit = 0;
  if(n < 12)
    limit = n;
  else
    limit = 12;
  double theta =0, phi = 0;
  double vec[3];

  for(int i = 0; i < limit; ++i)
  {
    pix2ang_nest(1, basic_sequence[i], &theta, &phi);
    ang2vec(theta, phi, vec);
    for(int j = 0; j < 3; ++j)
      result.push_back(vec[j]);
  }


  std::vector<double> points;
  int base_grid = 0, cur_point = 0;
  int point_healpix = 0;
  for(int i = 0; i < n - 12; ++i)
  {
    points.resize(0);
    base_grid = i%12;
    cur_point=i/12;
    point_healpix=4*basic_sequence[base_grid];
    points=find_point(basic_sequence[base_grid], cur_point, 1, point_healpix);
    for(int j = 0; j < 3; ++j)
      result.push_back(points[j]);
  }

  return result;
}

}
