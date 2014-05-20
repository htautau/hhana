# rootpy imports
import rootpy.compiled as C

C.register_code(
    """
    #ifndef PREFIT_ROOFITRESULT
    #define PREFIT_ROOFITRESULT
    #include <RooFitResult.h>
    class Prefit_RooFitResult: public RooFitResult {
    public:
      Prefit_RooFitResult(RooFitResult* _fitres);
      virtual ~Prefit_RooFitResult();
    };
    #endif
    """
                
