
After producing the analysis ntuples in the higgstautau package they are
prepared and merged here in the htt package.

Automatically organize ROOT and log files with::

    init-ntup

The above also merges the output from the subjobs for embedding and data.

Then move the running hhskim directory to production::

    mv ntuples/prod/hhskim ntuples/prod/hhskim_old
    mv ntuples/running/hhskim ntuples/prod/hhskim
