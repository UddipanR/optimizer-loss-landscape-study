# Decisions taken after seeing results
1. Corrected the labelling function of the Step 9 correlation table so that negative pooled correlations are labelled as opposite.
2. Added a filter-normalized Hessian measure (Step 8b) after the raw Hessian measures were found to track weight scale.
3. Weight decay controls: AdamW at 0.005 and 0.5 (fixed before running); Lion at 1.0 and 3.0, chosen AFTER the AdamW result (upward direction only).
4. Lion weight decay 3.0 runs are under-fit (train accuracy about 99.6%, late gradient norm about 5) and excluded from sharpness comparisons.
5. Control values were not selected on a validation split, so the controls are a sensitivity analysis and not tuned results.
6. The untuned Lion run (learning rate 0.0001) is excluded from all main analyses.
7. Test error (100 minus test accuracy) was added as a secondary correlation target after finding that the generalization gap understates Adam's deficit.
8. Reference variables (conv norm, weight norm) are labelled as such in the correlation tables, because they are not sharpness measures.
7. Test error (100 minus test accuracy) added as a secondary correlation target after finding that the generalization gap understates Adam's deficit (Adam fits the training set less tightly).
