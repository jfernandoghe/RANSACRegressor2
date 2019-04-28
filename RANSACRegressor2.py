import pandas as pd
import numpy as np
import numpy.linalg as lalg
from numpy import where
from itertools import compress
import io
import re
import random
from itertools import combinations, chain
import math
from math import log
import itertools
import csv
import statistics
####
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.ransac import _dynamic_max_trials
from sklearn.utils import check_random_state, check_array, check_consistent_length
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.random import sample_without_replacement



class RANSACRegressor2(linear_model.RANSACRegressor):
  
  def fitSegm(self, X, y, segmList, sample_weight=None):
    
    merged = list(itertools.chain.from_iterable(segmList))
  
    
    X = check_array(X, accept_sparse='csr')
    y = check_array(y, ensure_2d=False)
    check_consistent_length(X, y)

    
    
    if self.base_estimator is not None:
        base_estimator = clone(self.base_estimator)
    else:
        base_estimator = LinearRegression()

    if self.min_samples is None:
        # assume linear model by default
        min_samples = X.shape[1] + 1 if len(segmList)<2 else X.shape[1]      #MINIMUM SAMPLES   
    elif 0 < self.min_samples < 1:
        min_samples = np.ceil(self.min_samples * X.shape[0])
    elif self.min_samples >= 1:
        if self.min_samples % 1 != 0:
            raise ValueError("Absolute number of samples must be an "
                             "integer value.")
        min_samples = self.min_samples
    else:
        raise ValueError("Value for `min_samples` must be scalar and "
                         "positive.")
    if min_samples > X.shape[0]:
        raise ValueError("`min_samples` may not be larger than number "
                         "of samples: n_samples = %d." % (X.shape[0]))

    if self.stop_probability < 0 or self.stop_probability > 1:
        raise ValueError("`stop_probability` must be in range [0, 1].")

    if self.residual_threshold is None:
        # MAD (median absolute deviation)
        residual_threshold = np.median(np.abs(y - np.median(y)))
    else:
        residual_threshold = self.residual_threshold

    if self.loss == "absolute_loss":
        if y.ndim == 1:
            loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
        else:
            loss_function = lambda \
                y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)

    elif self.loss == "squared_loss":
        if y.ndim == 1:
            loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
        else:
            loss_function = lambda \
                y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)

    elif callable(self.loss):
        loss_function = self.loss

    else:
        raise ValueError(
            "loss should be 'absolute_loss', 'squared_loss' or a callable."
            "Got %s. " % self.loss)


    random_state = check_random_state(self.random_state)

    try:  # Not all estimator accept a random_state
        base_estimator.set_params(random_state=random_state)
    except ValueError:
        pass

    estimator_fit_has_sample_weight = has_fit_parameter(base_estimator,
                                                        "sample_weight")
    estimator_name = type(base_estimator).__name__
    if (sample_weight is not None and not
            estimator_fit_has_sample_weight):
        raise ValueError("%s does not support sample_weight. Samples"
                         " weights are only used for the calibration"
                         " itself." % estimator_name)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    n_inliers_best = 1
    score_best = -np.inf
    inlier_mask_best = None
    X_inlier_best = None
    y_inlier_best = None
    aicc_ = None
    self.n_skips_no_inliers_ = 0
    self.n_skips_invalid_data_ = 0
    self.n_skips_invalid_model_ = 0

    
       
    # Generate a list of indices for each segment
    size_sl = [len(s)-1 for s in segmList]
    n_segments = len(size_sl)
    
    
    # number of data samples
    n_samples = X.shape[0] 
    sample_idxs = np.arange(n_samples)

    n_samples, _ = X.shape
    
    
    self.n_trials_ = 0
    max_trials = self.max_trials
    
 
    
    
    while self.n_trials_ < max_trials:
        self.n_trials_ += 1
        if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips:
            break
       


        
        # choose random sample set
        ## antes:
        ## subset_idxs = sample_without_replacement(n_samples, min_samples, random_state=random_state)
        ## ahora:
        subset_idx_entries = sample_without_replacement(n_segments, min_samples,
                                                 random_state=random_state)
        
        
        
        subset_idxs = np.asarray([segmList[ss][random.randint(0, size_sl[ss])] \
                       for ss in subset_idx_entries])
   
    
        
        
        X_subset = X[subset_idxs]
        y_subset = y[subset_idxs]

        # check if random sample set is valid
        if (self.is_data_valid is not None
                and not self.is_data_valid(X_subset, y_subset)):
            self.n_skips_invalid_data_ += 1
            continue        

        # fit model for current random sample set
        if sample_weight is None:
          base_estimator.fit(X_subset, y_subset)
        else:
          base_estimator.fit(X_subset, y_subset, sample_weight=sample_weight[subset_idxs])

        # check if estimated model is valid
        if (self.is_model_valid is not None and not
                self.is_model_valid(base_estimator, X_subset, y_subset)):
            self.n_skips_invalid_model_ += 1
            continue
            
        # check if estimated model is valid (ii)
        y_pred_subset = base_estimator.predict(X_subset)
        residuals_ii = loss_function(y_subset, y_pred_subset)
        inlier_mask_subset_ii = residuals_ii < residual_threshold
        
        
        if np.sum(inlier_mask_subset_ii)< X.shape[1]:
          self.n_skips_invalid_model_ += 1
          continue      
        
      
            
          
          
          
        # residuals of all data for current random sample model
        y_pred = base_estimator.predict(X[merged])


        residuals_subset = loss_function(y[merged], y_pred)
        # classify data into inliers and outliers
        inlier_mask_subset = residuals_subset < residual_threshold   

        n_inliers_subset = np.sum(inlier_mask_subset)
        
        #check that the all points in sample are inliers
        if n_inliers_subset < min_samples:
            continue
        
        # less inliers -> skip current random sample
        if n_inliers_subset < n_inliers_best:
            self.n_skips_no_inliers_ += 1
            continue

            
        # extract inlier data set        
        inlier_idxs_subset = list(compress(merged, inlier_mask_subset))
        
        X_inlier_subset = X[inlier_idxs_subset]
        y_inlier_subset = y[inlier_idxs_subset]
        

        # score of inlier data set
        score_subset = base_estimator.score(X_inlier_subset,
                                            y_inlier_subset)


        # same number of inliers but worse score -> skip current random
        # sample
        if (n_inliers_subset == n_inliers_best
                and score_subset <= score_best):
            continue

            
    
        # save current random sample as best sample
        n_inliers_best = n_inliers_subset
        score_best = score_subset
        inlier_mask_best = inlier_mask_subset
        X_inlier_best = X_inlier_subset
        y_inlier_best = y_inlier_subset
        

        max_trials = min(
            max_trials,
            _dynamic_max_trials(n_inliers_best, n_samples,
                                min_samples, self.stop_probability))


        # break if sufficient number of inliers or score is reached
        if n_inliers_best >= self.stop_n_inliers or \
                        score_best >= self.stop_score:
            break

  # if none of the iterations met the required criteria
    if inlier_mask_best is None:
        base_estimator.coef_=-999
        if ((self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips):
            raise ValueError(
                "RANSAC skipped more iterations than `max_skips` without"
                " finding a valid consensus set. Iterations were skipped"
                " because each randomly chosen sub-sample failed the"
                " passing criteria. See estimator attributes for"
                " diagnostics (n_skips*).")
        else:
            raise ValueError(
                "RANSAC could not find a valid consensus set. All"
                " `max_trials` iterations were skipped because each"
                " randomly chosen sub-sample failed the passing criteria."
                " See estimator attributes for diagnostics (n_skips*).")
    else:
        if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips:
            warnings.warn("RANSAC found a valid consensus set but exited"
                          " early due to skipping more iterations than"
                          " `max_skips`. See estimator attributes for"
                          " diagnostics (n_skips*).",
                          ConvergenceWarning)
    # estimate final model using all inliers
        base_estimator.fit(X_inlier_best, y_inlier_best)    
        self.estimator_ = base_estimator
        self.inlier_mask_ = inlier_mask_best
        return self
##############
