import pandas as pd
import numpy as np
from typing import List, Callable
from sklearn import base
from sklearn.inspection import permutation_importance


class AMGFeatureSelection:
    # TODO Refactoring
    def __init__(self, model: BaseEstimator,
                 features: List[str], target: str, train_set_name: str, valid_set_name: str,
                 scorer_function: Callable, n_feature_batch: int, min_uplift: float, n_min_feature: int,
                random_state: int = 2021, verbose: bool = True, n_jobs: int=-1):
        """
        Perform feature selection based on metric gain. Dataset should contain column __SET__ which is uset to split
        data on train and valid sets
        :param model: Scikit-learn model, LightGBM, XGBoost models
        :param features: List of features from which select best features
        :param target: column name of target value
        :param train_set_name: name of train subset in column __SET__
        :param valid_set_name: name of valid subset in column __SET__
        :param scorer_function: scikit-learn scoring function
        :param n_feature_batch: number of features per iteration on first stage of selection
        :param min_uplift: Mininum metric gain to save a feature
        :param n_min_feature: Minimum number of features to keep
        :param random_state:
        :param verbose: True - enable some notification, False - off any notifications
        :param n_jobs:
        """
        self.model = base.clone(model)
        self.features = features
        self.target = target
        self.scorer_function = scorer_function
        self.n_feature_batch = n_feature_batch
        self.min_uplift = min_uplift
        self.n_min_feature = n_min_feature
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.train_set_name = train_set_name
        self.valid_set_name = valid_set_name
        self.verbose = verbose

    def _get_features_chunk(self) -> List[str]:
        """
        Return last n_features_batch from sorted features by feature importance
        :return:
        """
        features_chunk = (self.resulted_features[:self.n_feature_batch]
                          if len(self.resulted_features) > self.n_feature_batch else
                          self.resulted_features)
        return features_chunk

    def _permutaion_importance(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                               n_repeats: int = 1) -> List[str]:
        """
        Perform computation of feature importances based on permutation method. If model is boosting
        with feature_importances_ attribute, it is used instead permutation importance method
        :param model: Model which is used to compute importances of features
        :param X: Dataset with features to compute importances
        :param y: Target values
        :param n_repeats: Number of times to permute features
        :return: Sorted list of features from higher to lower importance values
        """

        if hasattr(model, "feature_importances_"): # If model is boosting
            sorted_features = [x for _, x in sorted(zip(model.feature_importances_, X.columns.tolist()), reverse=False)]

        else:
            permutation_importance_scores = permutation_importance(model,
                                                                   X=X,
                                                                   y=y,
                                                                   scoring=self.scorer_function,
                                                                   n_repeats=n_repeats,
                                                                   random_state=self.random_state,
                                                                   n_jobs=self.n_jobs)["importances_mean"]

            # Sorted features from lower to higher values of importances
            sorted_features = [x for _, x in sorted(zip(permutation_importance_scores,
                                                        X.columns.tolist()),
                                                    reverse=False)]
        return sorted_features

    def _fit_base_model(self, df: pd.DataFrame, features: List[str], feature_to_exclude: str = None) -> dict:
        """
        Fit base model on "features" without "feature_to_exclude"
        :param df: Data to fit model
        :param features: list of features
        :param feature_to_exclude: features which should be excluded during fitting
        :return:
        """

        if feature_to_exclude is not None:
            shorted_features = np.setdiff1d(features, feature_to_exclude)
        else:
            shorted_features = features
            feature_to_exclude = "not_remove"

        #### Specially for LightGBM model, metric greater - better
        if isinstance(self.model, lgbm.sklearn.LGBMClassifier):
            estimator = self.model.fit(X=df.loc[self.train_indx, shorted_features],
                           y=df.loc[self.train_indx, self.target],
                           eval_metric=self.model.metric,
                           eval_set=[(df.loc[self.valid_indx, shorted_features],
                                      df.loc[self.valid_indx, self.target])],
                           eval_names=['valid'], verbose=False)
            score = np.max(estimator.evals_result_['valid'][self.model.metric])

        #### Specially for LightGBM model, metric less - better
        elif isinstance(self.model, lgbm.sklearn.LGBMRegressor):
            estimator = self.model.fit(X=df.loc[self.train_indx, shorted_features],
                                       y=df.loc[self.train_indx, self.target],
                                       eval_metric=self.model.metric,
                                       eval_set=[(df.loc[self.valid_indx, shorted_features],
                                                  df.loc[self.valid_indx, self.target])],
                                       eval_names=['valid'], verbose=False)
            score = np.min(estimator.evals_result_['valid'][self.model.metric])
        else:
        #### For Rest cases
            estimator = self.model.fit(df.loc[self.train_indx, shorted_features],
                                       df.loc[self.train_indx, self.target])

            score = self.scorer_function(estimator,
                                         df.loc[self.valid_indx, shorted_features],
                                         df.loc[self.valid_indx, self.target])

        return {"feature": feature_to_exclude, "score": score, "model": estimator}

    def _selection_step(self, df: pd.DataFrame, chunk_of_features: List[str],
                        score_base: float, enable_permutation: bool = True)-> tuple:

        resulted_features = self.resulted_features.copy()

        for feature in chunk_of_features:
            feature_score = self._fit_base_model(df=df, features=self.resulted_features,
                                                 feature_to_exclude=feature)['score']
            uplift = score_base - feature_score
            if uplift < self.min_uplift and len(resulted_features) > self.n_min_feature:
                resulted_features.remove(feature)

        res = self._fit_base_model(df=df, features=resulted_features, feature_to_exclude=None)

        score_base = res["score"]

        if enable_permutation:
            resulted_features = self._permutaion_importance(model=res['model'],
                                                            X=df.loc[self.valid_indx, resulted_features],
                                                            y=df.loc[self.valid_indx, self.target],
                                                            n_repeats=2)
        return list(resulted_features), score_base

    def fit(self, df: pd.DataFrame):
        print("Fitting is started")

        self.train_indx = df[df["__SET__"] == self.train_set_name].index
        self.valid_indx = df[df["__SET__"] == self.valid_set_name].index

        original_estimator = self.model.fit(df.loc[self.train_indx, self.features],
                                                     df.loc[self.train_indx, self.target])

        score_base = self.scorer_function(original_estimator,
                                          df.loc[self.valid_indx, self.features],
                                          df.loc[self.valid_indx, self.target])

        # Sorted features from lower to higher values of importances
        self.resulted_features = self._permutaion_importance(model=original_estimator,
                                                             X=df.loc[self.valid_indx, self.features],
                                                             y=df.loc[self.valid_indx, self.target],
                                                             n_repeats = 1)
        if self.verbose:
            print("Base model fitted")

        doing_first_stage = True
        first_stage_iter = 0
        while doing_first_stage:
            chunk_of_features = self._get_features_chunk()
            prev_features_number = len(self.resulted_features)

            self.resulted_features, score_base = self._selection_step(df=df,
                                                                      chunk_of_features=chunk_of_features,
                                                                      score_base=score_base,
                                                                      enable_permutation=True)

            next_chunk_of_features = self._get_features_chunk()
            features_number = len(self.resulted_features)

            if ((set(next_chunk_of_features) == set(chunk_of_features)) or
                    (len(self.resulted_features) <= self.n_min_feature) or
                    (prev_features_number == features_number)):
                doing_first_stage = False

            if self.verbose:
                first_stage_iter += 1
                print(
                    f"First stage. Iteration: {first_stage_iter}, features left: {len(self.resulted_features)}")

        if self.verbose:
            print("First stage finished")

        doing_second_stage = True if len(self.resulted_features) > self.n_min_feature else False
        second_stage_iter = 0
        while doing_second_stage:
            prev_features_number = len(self.resulted_features)

            self.resulted_features, score_base = self._selection_step(df=df,
                                                                      chunk_of_features=self.resulted_features,
                                                                      score_base=score_base,
                                                                      enable_permutation=True)
            features_number = len(self.resulted_features)

            if (prev_features_number == features_number) or (features_number <= self.n_min_feature):
                doing_second_stage = False

            if self.verbose:
                second_stage_iter += 1
                print(
                    f"Second stage. Iteration: {second_stage_iter}, features left: {len(self.resulted_features)}")

        if self.verbose:
            print('Process finished')

        return self


class SMGFeatureSelection:
    # TODO Statistical metric gain feature selection
    def __init__(self):
        pass

# TODO BorutaPy
