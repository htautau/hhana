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

from rootpy.compiled import Prefit_RooFitResult

