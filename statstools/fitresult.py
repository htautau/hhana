# rootpy imports
import rootpy.compiled as C

C.register_code(
    """
    #ifndef PREFIT_ROOFITRESULT
    #define PREFIT_ROOFITRESULT
    #include <RooFitResult.h>
    #include <TMatrixDSym.h>
    class Prefit_RooFitResult: public RooFitResult {
      public:
        Prefit_RooFitResult(RooFitResult* _fitres, bool decorelate=false): RooFitResult(), fitres(_fitres)
        {
          RooFitResult::setConstParList(fitres->constPars());
          RooFitResult::setInitParList(fitres->floatParsInit());
          RooFitResult::setFinalParList(fitres->floatParsInit());
          TMatrixDSym cov = fitres->covarianceMatrix();
          if (decorelate)
            for(int icol=0; icol<cov.GetNcols(); icol++)
              for(int irow=0; irow<cov.GetNrows(); irow++)
                cov(icol, irow) = (icol==irow ? cov(icol, irow) : 0);
          RooFitResult::setCovarianceMatrix(cov);
          
        }
        virtual ~Prefit_RooFitResult()
        {delete fitres;}
      private:
        RooFitResult* fitres;
      protected:
       ClassDef(Prefit_RooFitResult, 5);
    };
    #endif
    """, ["Prefit_RooFitResult"])

C.register_code(
    """
    #ifndef PARTIAL_ROOFITRESULT
    #define PARTIAL_ROOFITRESULT
    #include <RooFitResult.h>
    #include <TMatrixDSym.h>
    #include <RooArgList.h>
    class Partial_RooFitResult: public RooFitResult {
      public:
        Partial_RooFitResult(const RooFitResult& _fitres, Int_t nparams, const Int_t *index_params): RooFitResult(_fitres)
        {
           TMatrixDSym origin_cov = this->covarianceMatrix();
           TMatrixDSym new_cov(0, origin_cov.GetNrows()-1);
           new_cov *= 0.; //set the matrix to 0s.
           //for(int icol=0; icol<new_cov.GetNcols(); icol++)
             //for(int irow=0; irow<new_cov.GetNrows(); irow++)
               //new_cov(icol, irow) = 0.;

           for(int ind=0; ind<nparams; ind++){
             Int_t index = index_params[ind];
             for(int icol=0; icol<new_cov.GetNcols(); icol++)
               new_cov(icol, index) = origin_cov(icol, index);
             for(int irow=0; irow<new_cov.GetNrows(); irow++)
               new_cov(index, irow) = origin_cov(index, irow);
           }
           RooFitResult::setCovarianceMatrix(new_cov);
        }
        virtual ~Partial_RooFitResult(){;}
      protected:
        ClassDef(Partial_RooFitResult, 5);
    };
    #endif
    """, ["Partial_RooFitResult"])


from rootpy.compiled import Prefit_RooFitResult
from rootpy.compiled import Partial_RooFitResult

