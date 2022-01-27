#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.ode.mbase import ModelBase


class SIR(ModelBase):
    """
    SIR model.

    Args:
        population (int): total population
        theta (float)
        kappa (float)
        rho (float)
        sigma (float)
        phi (float)
        beta (float)
        myu (float)
        gammaa (float)
        delta (float)
        vacrate (float)
    """
    # Model name
    NAME = "SIR"
    # names of parameters
    PARAMETERS = ["theta", "kappa", "rho", "sigma", "phi", "beta", "myu", "gammaa", "delta", "vacrate"]
    DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]", "Vaccination rate[target population/completion time]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.FR,
        "p": ModelBase.V
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([1, 1, 1, 1])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.FR]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "theta": 0.002, "kappa": 0.005, "rho": 0.2, "sigma": 0.075, "vacrate": 0.186,
        },
        ModelBase.Y0_DICT: {
            ModelBase.S: 999_000, ModelBase.CI: 1000, ModelBase.FR: 0, ModelBase.V: 0,
        },
    }

    def __init__(self, population, theta, kappa, rho, sigma, phi, myu, vacrate):
        # Total population
        self.population = self._ensure_population(population)
        # Non-dim parameters
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma
        self.phi = phi
        self.myu = myu
        self.vacrate = vacrate
        self.non_param_dict = {
            "theta": theta, "kappa": kappa, "rho": rho, "sigma": sigma, "phi": phi, "myu": myu, "vacrate": vacrate}

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, i, *_ = X
        dsdt = 0 - self.rho * s * i / n + self.phi - self.myu - self.vacrate
        drdt = self.sigma * i - self.myu
        dvdt = self.vacrate
        didt = 0 - dsdt - drdt - dvdt
        return np.array([dsdt, didt, drdt, dvdt])

    @classmethod
    @deprecate(".param_range()", new=".guess()", version="2.19.1-zeta-fu1")
    def param_range(cls, taufree_df, population, quantiles=(0.1, 0.9)):
        """
        Deprecated. Define the value range of ODE parameters using (X, dX/dt) points.
        In SIR model, X is S, I, R, F here.

        Args:
            taufree_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - t (int): time steps (tau-free)
                    - columns with dimensional variables
            population (int): total population
            quantiles (tuple(int, int)): quantiles to cut, like confidence interval

        Returns:
            dict(str, tuple(float, float)): minimum/maximum values
        """
        cls._ensure_dataframe(taufree_df, name="taufree_df", columns=[cls.TS, *cls.VARIABLES])
        df = taufree_df.copy()
        df = df.loc[(df[cls.S] > 0) & (df[cls.CI] > 0)]
        n, t = population, df[cls.TS]
        s, i, r = df[cls.S], df[cls.CI], df[cls.FR]
        # kappa = (dF/dt) / I when theta -> 0
        # kappa_series = f.diff() / t.diff() / i
        # rho = - n * (dS/dt) / S / I
        rho_series = 0 - n * s.diff() / t.diff() / s / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # vacrate
        vacrate_series = (n - s + i + r).diff() / t.diff()
        # Calculate quantile
        _dict = {
            k: tuple(v.quantile(quantiles).clip(0, 1)) for (k, v)
            in zip(["rho", "sigma", "vacrate"], [rho_series, sigma_series, vacrate_series])
        }
        #_dict["theta"] = (0.0, 1.0)
        return _dict

    @classmethod
    @deprecate(".specialize()", new=".convert()", version="2.19.1-zeta-fu1")
    def specialize(cls, data_df, population):
        """
        Deprecated. Specialize the dataset for this model.

        Args:
            data_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): total population in the place

        Returns:
            (pandas.DataFrame)
                Index
                    reset index
                Columns
                    - any columns @data_df has
                    - Susceptible (int): the number of susceptible cases
        """
        cls._ensure_dataframe(data_df, name="data_df", columns=cls.VALUE_COLUMNS)
        df = data_df.copy()
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.FR] = df[cls.F] + df[cls.R]
        return df

    @classmethod
    @deprecate(".restore()", new=".convert_reverse()", version="2.19.1-zeta-fu1")
    def restore(cls, specialized_df):
        """
        Deprecated. Restore Confirmed/Infected/Recovered/Fatal.
         using a dataframe with the variables of the model.

        Args:
        specialized_df (pandas.DataFrame): dataframe with the variables

            Index
                (object)
            Columns
                - Susceptible (int): the number of susceptible cases
                - Infected (int): the number of currently infected cases
                - Recovered (int): the number of recovered cases
                - Fatal (int): the number of fatal cases
                - any columns

        Returns:
            (pandas.DataFrame)
                Index
                    (object): as-is
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - the other columns @specialzed_df has
        """
        df = specialized_df.copy()
        other_cols = list(set(df.columns) - set(cls.VALUE_COLUMNS))
        df[cls.C] = df[cls.CI] + df[cls.FR]
        df[cls.F] = 0
        df[cls.R] = df[cls.FR]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rt = self.rho  / self.sigma
        except ZeroDivisionError:
            return None
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]

        Returns:
            dict[str, int]
        """
        try:
            return {
                "1/beta [day]": int(tau / 24 / 60 / self.rho),
                "1/gamma [day]": int(tau / 24 / 60 / self.sigma),
                "Vaccination rate [target population/completion time]": float(self.vacrate)
            } 
        except ZeroDivisionError:
            return {p: None for p in self.DAY_PARAMETERS}
