/**
 *  Original version of the specular highlights removal code by Robby T. Tan
 *  reference:
 *  "separating reflection components of textured surfaces using a single image"
 *  by Robby T. Tan, Katsushi Ikeuchi,
 *  IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),
 *  27(2), pp.179-193, February, 2005
 *
 *  This is the original implementation based on the C++ code provided by Prof.
 *  Robby Tan but using Blitz++ arrays:
 *       http://tanrobby.github.io/code.html#
 *       http://tanrobby.github.io/code/highlight.zip
 *
 *  It also implements a modification used in the previous python version that
 *  ignores pixels marked G_DIFFUSE, leading to a smaller number of
 *  iterations per epsilon value, and a flag that ensure no nan and inf values
 *  are produced.
 */

#include <blitz/array.h>

#define SPECULARX		10
#define SPECULARY		11
#define DIFFUSE			12
#define BOUNDARY		13
#define NOISE       14
#define CAMERA_DARK 15

void specular_free_image( blitz::Array<float ,3> &src,
                          blitz::Array<int,2>   &src_i,
					                blitz::Array<float ,3> &sfi,
                          bool check_nan_inf);

void iteration(           blitz::Array<float ,3> &src,
                          blitz::Array<int,2>   &src_i,
                          blitz::Array<float ,3> &sfi,
				                  float  epsilon,
                          bool skip_diffuse,
                          bool check_nan_inf);

int init(                 blitz::Array<float ,3> &src,
                          blitz::Array<int,2>   &src_i,
                          blitz::Array<float ,3> &sfi,
				                  float  epsilon,
                          bool skip_diffuse,
                          bool check_nan_inf);

int reset_labels(         blitz::Array<int,2>   &src_i);

// the main function to remove highlights from a single image
void remove_highlights(   blitz::Array<float ,3> &img,
                          blitz::Array<float ,3> &diff,
                          blitz::Array<float ,3> &sfi,
                          blitz::Array<float ,3> &residue,
                          float  epsilon,
                          bool skip_diffuse,
                          bool check_nan_inf)
{
  // flags
  int dim_x = img.shape()[2];
  int dim_y = img.shape()[1];

  blitz::Array<int,2> img_i(dim_y, dim_x);

  //SPECULAR-FREE IMAGE

  specular_free_image(img, img_i, sfi, check_nan_inf);

  //ITERATIVE PART
  float  step =0.01f;

  // copy source
  diff = img;

  while( epsilon >= 0.0 )
  {
    // run the main iteration
    //printf("*");
    iteration(diff, img_i, sfi, epsilon, check_nan_inf, skip_diffuse);
    epsilon -= step;
    //printf(": %f\n",epsilon);
  }

  // compute residue
  residue = img - diff;
}

// utilities

inline float  tot(float  r, float  g, float  b)
{
  return r + g + b;
}

inline float  max(float  r, float  g, float  b)
{
  float  max_ = r;
  if(g>max_) max_ = g;
  if(b>max_) max_ = b;
  return max_;
}

inline float  r_chroma(float  r, float  g, float  b)
{
  float  tot_ = tot(r,g,b);
  if (tot_!=0)  return r/tot_;
  else          return 0;
}

inline float  g_chroma(float  r, float  g, float  b)
{
  float  tot_ = tot(r,g,b);
  if (tot_!=0)  return g/tot_;
  else          return 0;
}

inline float  b_chroma(float  r, float  g, float  b)
{
  float  tot_ = tot(r,g,b);
  if (tot_!=0)  return b/tot_;
  else          return 0;
}

inline float  max_chroma(float  r, float  g, float  b)
{
  float  tot_ = tot(r,g,b);
  if (tot_!=0)  return max(r,g,b)/tot_;
  else          return 0;
}

// remove the specular component from source

void specular_free_image( blitz::Array<float ,3> &src,
                          blitz::Array<int,2>   &src_i,
					                blitz::Array<float ,3> &sfi,
                          bool check_nan_inf)
{
  float  Lambda=0.6f;
  float  camDark=10.0f;	// for pixels that are too dark
  float  lambdaConst = 3.0f * Lambda - 1.0f;

  //SPECULAR FREE IMAGE
  int dim_x = src.shape()[2];
  int dim_y = src.shape()[1];

  int y,x;
  for(y = 0; y < dim_y; y++){
    for(x = 0; x < dim_x; x++){
      //get the rgb values
      float  r = src(0,y,x);
      float  g = src(1,y,x);
      float  b = src(2,y,x);

      //copy the rgb to the sfi
      sfi(0,y,x) = r;
      sfi(1,y,x) = g;
      sfi(2,y,x) = b;

      // init
      src_i(y,x) = 0;

      //check for camera dark and achromatic pixels
      if(((r < camDark) &&
          (g < camDark) &&
          (b < camDark)))
      {
	       src_i(y,x) = CAMERA_DARK;
	       continue;
      }

      //perform the specular-to-diffuse mechanism
      float  c    = max_chroma(r,g,b);
      float  numr = max(r,g,b) * (3.0f * c - 1.0f);
      float  denm = c * lambdaConst;

      float  dI   = numr / denm;
      if(denm == 0 && check_nan_inf) dI = 0;

      float  sI = (tot(r,g,b) - dI)/3.0f;

      float  dr,dg,db;
      dr = (r - sI);
      dg = (g - sI);
      db = (b - sI);

      if(dr<0) dr=0;
      if(dg<0) dg=0;
      if(db<0) db=0;

      if(dr>255) dr=255;
      if(dg>255) dg=255;
      if(db>255) db=255;

      sfi(0,y,x) = dr;
      sfi(1,y,x) = dg;
      sfi(2,y,x) = db;

    }
  }

}

// to apply specular to diffuse equation or mechanism

inline int specular_2_diffuse(int y, int x, blitz::Array<float ,3> &iro,
                              blitz::Array<int,2> &iro_i,
                              float  maxChroma,
                              bool check_nan_inf)
{
  float  c = max_chroma(iro(0,y,x), iro(1,y,x), iro(2,y,x));
  float  m = max(iro(0,y,x), iro(1,y,x), iro(2,y,x));
  float  t = tot(iro(0,y,x), iro(1,y,x), iro(2,y,x));
  float  numr = (m*(3.0f*c - 1.0f));
  float  denm = (c*(3.0f*maxChroma - 1.0f));

  if(check_nan_inf && abs(denm) < 0.000000001)
  {
    iro_i(y,x)=NOISE;
    return 1;
  }

  float  dI = numr / denm;

  float  sI = (t - dI)/3.0f;

  float  nr = (iro(0,y,x) - sI);
  float  ng = (iro(1,y,x) - sI);
  float  nb = (iro(2,y,x) - sI);

  if(nr<=0 || ng<=0 || nb<=0)
  {
    iro_i(y,x)=NOISE;
    return 1;
  }
  else
  {
    iro(0,y,x) = nr;
    iro(1,y,x) = ng;
    iro(2,y,x) = nb;

    return 0;
  }
}

// specular reduction mechanism

void iteration( blitz::Array<float ,3> &src,
                blitz::Array<int,2>   &src_i,
                blitz::Array<float ,3> &sfi,
				        float  epsilon,
                bool skip_diffuse,
                bool check_nan_inf)
{
  int x,y;
  int dim_x = src.shape()[2];
  int dim_y = src.shape()[1];

  float  thR = 0.1f, thG = 0.1f;

  // to have the initial labels
  int count = init(src,src_i,sfi,epsilon,skip_diffuse,check_nan_inf);
  int pcount;

  while(1)
  {
    for(y=0;y<dim_y-1;y++)
    {
	    for(x=0;x<dim_x-1;x++)
      {

        if(src_i(y,x)==CAMERA_DARK) continue;

        //get the rgb values
        float  r = src(0,y,x);
        float  g = src(1,y,x);
        float  b = src(2,y,x);

        float  cr = r_chroma(r,g,b);		// red chroma
        float  cg = g_chroma(r,g,b);		// green chroma

        //get the rgb values
        float  rx = src(0,y,x+1);
        float  gx = src(1,y,x+1);
        float  bx = src(2,y,x+1);

        float  cr_next_x = r_chroma(rx,gx,bx); // red chroma
        float  cg_next_x = g_chroma(rx,gx,bx); // green chroma

        //get the rgb values
        float  ry = src(0,y+1,x);
        float  gy = src(1,y+1,x);
        float  by = src(2,y+1,x);

        float  cr_next_y = r_chroma(ry,gy,by); // red chroma
        float  cg_next_y = g_chroma(ry,gy,by); // green chroma

        // derivatives
        float  drx = cr_next_x-cr;//pixel right
        float  dgx = cg_next_x-cg;
        float  dry = cr_next_y-cr;//pixel below
        float  dgy = cg_next_y-cg;

        if(src_i(y,x) == SPECULARX)
        {
          //if it is  a boundary in the x direction
          if(fabs(drx) > thR && fabs(dgx) > thG)
          {
            //pixel right
            src_i(y,x)=BOUNDARY;
            continue;
          }

          //if it is a noise
          if(fabs(max_chroma(r,g,b) - max_chroma(rx,gx,bx)) < 0.01)
          {
            src_i(y,x)=NOISE;
            continue;
          }

          //reduce the specularity at x direction
          if(max_chroma(r,g,b) < max_chroma(rx,gx,bx))
          {
            specular_2_diffuse(y,x,src,src_i,max_chroma(rx,gx,bx),check_nan_inf);
            src_i(y,x)=DIFFUSE;
            src_i(y,x+1)=DIFFUSE;
          }
          else
          {
            specular_2_diffuse(y,x+1,src,src_i,max_chroma(r,g,b),check_nan_inf);
            src_i(y,x)=DIFFUSE;
            src_i(y,x+1)=DIFFUSE;
          }
        }

	      if(src_i(y,x) == SPECULARY)
	      {
	        //if it is a boundary in the y direction
	        if(fabs(dry) > thR && fabs(dgy) > thG)
          {
            //pixel right
		        src_i(y,x)=BOUNDARY;
		        continue;
	        }

	        //if it is a noise
	        if(fabs(max_chroma(r,g,b) - max_chroma(ry,gy,by))<0.01)
          {
		        src_i(y,x)=NOISE;
		        continue;
	        }

	        //reduce the specularity in y direction
	        if(max_chroma(r,g,b) < max_chroma(ry,gy,by))
          {
            specular_2_diffuse(y,x,src,src_i,max_chroma(ry,gy,by),check_nan_inf);
            src_i(y,x)=DIFFUSE;
            src_i(y+1,x)=DIFFUSE;
          }
          else
          {
            specular_2_diffuse(y+1,x,src,src_i,max_chroma(r,g,b),check_nan_inf);
            src_i(y,x)=DIFFUSE;
            src_i(y+1,x)=DIFFUSE;
          }
	      }
	    }
    }

    pcount=count;
    count = init(src,src_i,sfi,epsilon,skip_diffuse,check_nan_inf);

    if(count==0)
	    break;
    if(pcount<=count)
	    break;
  }

  reset_labels(src_i);
}

// to have initial labels

int init(   blitz::Array<float ,3> &src,
            blitz::Array<int,2>   &src_i,
            blitz::Array<float ,3> &sfi,
				    float   epsilon,
            bool    skip_diffuse,
            bool    check_nan_inf)
{
  int dim_x = src.shape()[2];
  int dim_y = src.shape()[1];
  int x,y;	// pixel iterators

  int count=0;

  for(y = 1; y < dim_y - 1; y++){
    for(x = 1; x < dim_x - 1; x++){
      switch(src_i(y,x))
      {
        case BOUNDARY:
        case NOISE:
        case CAMERA_DARK:
	        continue;
        case DIFFUSE:
          if(skip_diffuse) continue;
	      break;
      }

      float  src_tot_0 = tot(src(0,y,x), src(1,y,x), src(2,y,x));
      float  src_tot_x = tot(src(0,y,x+1), src(1,y,x+1), src(2,y,x+1));
      float  src_tot_y = tot(src(0,y+1,x), src(1,y+1,x), src(2,y+1,x));

      float  sfi_tot_0 = tot(sfi(0,y,x), sfi(1,y,x), sfi(2,y,x));
      float  sfi_tot_x = tot(sfi(0,y,x+1), sfi(1,y,x+1), sfi(2,y,x+1));
      float  sfi_tot_y = tot(sfi(0,y+1,x), sfi(1,y+1,x), sfi(2,y+1,x));

      float  dlog_src_x = log( fabs( src_tot_x - src_tot_0 ) );
      float  dlog_src_y = log( fabs( src_tot_y - src_tot_0 ) );

      float  dlog_sfi_x = log( fabs( sfi_tot_x - sfi_tot_0 ) );
      float  dlog_sfi_y = log( fabs( sfi_tot_y - sfi_tot_0 ) );

      float  dlogx = (dlog_src_x - dlog_sfi_x);
      float  dlogy = (dlog_src_y - dlog_sfi_y);

      dlogx=fabs(dlogx);
      dlogy=fabs(dlogy);

      // specular in the x direction
      // if(dlogx > epsilon || std::isinf(dlog_src_x) || std::isinf(dlog_sfi_x))
      if(dlogx > epsilon)
      {
  	     src_i(y,x) = SPECULARX;
	       count++;
	       continue;  // go to the next pixel
      }

      //specular in the y direction
      // if(dlogy > epsilon || std::isinf(dlog_src_y) || std::isinf(dlog_sfi_y))
      if(dlogy > epsilon)
      {
	       src_i(y,x)= SPECULARY;
	       count++;
	       continue;
      }

      src_i(y,x) = DIFFUSE;

    }
  }

  return count;	// return the number of specular pixels
}

// to reset the label of the pixels

int reset_labels(blitz::Array<int,2> &src_i)
{
  int dim_x = src_i.shape()[1];
  int dim_y = src_i.shape()[0];

  for(int y=0;y<dim_y;y++)
  {
    for(int x=0;x<dim_x;x++)
    {
      if(src_i(y,x)==CAMERA_DARK) continue;
      src_i(y,x)=0;
    }
  }
  return 0;
}
