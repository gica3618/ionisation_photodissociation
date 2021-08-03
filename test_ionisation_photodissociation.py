#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:23:27 2021

@author: gianni
"""

#for testing with pytest

import ionisation_photodissociation as ip
import numpy as np
import pytest
from scipy import constants
import os
import itertools


class Test_ISF():

    isf = ip.ISF()

    def test_scaling(self):
        scaled_isf = ip.ISF(scaling=2)
        test_lamb = 100*constants.nano
        assert self.isf.flux(test_lamb) == scaled_isf.flux(test_lamb)/2

    def test_lambda_check(self):
        too_small_lambda = np.array((0.1*self.isf.lambda_min,0.5*self.isf.lambda_min))
        too_large_lambda = np.array((1.01*self.isf.lambda_max,2*self.isf.lambda_max))
        for lamb in (too_small_lambda,too_large_lambda):
            assert np.all(self.isf.flux(lamb) == 0)


def generate_test_atm():
    atm = ip.StellarAtmosphere()
    atm.lambda_grid = np.array([1,2.2,3.3,4.6])
    atm.modelflux = np.array((10.2,20,40.4,90))
    atm.ref_distance = 2
    return atm


class Test_StellarAtmosphere():

    atm = generate_test_atm()

    def test_lamb_limits(self):
        too_large_lamb = np.array((5,7))
        too_small_lamb = np.array((0.1,0.9))
        for lamb in (too_large_lamb,too_small_lamb):
            assert np.all(self.atm.flux(wavelength=lamb,distance=1)==0)

    def test_flux_interpolation(self):
        assert self.atm.flux(wavelength=self.atm.lambda_grid[0],
                             distance=self.atm.ref_distance)\
                                                   == self.atm.modelflux[0]
        assert self.atm.flux(wavelength=self.atm.lambda_grid[0],
                             distance=2*self.atm.ref_distance)\
                                                   == self.atm.modelflux[0]/4

    def test_luminosity(self):
        expected_lum = np.trapz(self.atm.modelflux,self.atm.lambda_grid)\
                                *4*np.pi*self.atm.ref_distance**2
        assert self.atm.luminosity() == expected_lum

    def test_scaling(self):
        scalings = [lambda wavelength: 3.5,
                    lambda wavelength: wavelength**2]
        for scaling in scalings:
            test_atm = generate_test_atm()
            original_model_flux = test_atm.modelflux.copy()
            unscaled_flux = test_atm.flux(wavelength=test_atm.lambda_grid,
                                          distance=test_atm.ref_distance)
            test_atm._scale_spectrum(scaling=scaling)
            scaling_factors = scaling(wavelength=test_atm.lambda_grid)
            assert np.all(original_model_flux*scaling_factors==test_atm.modelflux)
            scaled_flux = test_atm.flux(wavelength=test_atm.lambda_grid,
                                        distance=test_atm.ref_distance)
            assert np.all(unscaled_flux*scaling_factors==scaled_flux)
            expected_luminosity = np.trapz(unscaled_flux*scaling_factors,
                                           test_atm.lambda_grid)\
                                 *4*np.pi*test_atm.ref_distance**2
            assert expected_luminosity == test_atm.luminosity()

    def test_plot_model(self):
        self.atm.plot_model()

    def test_file_writing(self):
        test_filepath = 'test.npz'
        dist = 1*constants.au
        self.atm.write_modelflux_to_file(filepath=test_filepath,distance=dist)
        data = np.load(test_filepath)
        assert np.all(data['wavelength']==self.atm.lambda_grid)
        assert np.all(data['flux']
                        ==self.atm.flux(wavelength=data['wavelength'],distance=dist))
        os.remove(test_filepath)


def general_test_flux_scaling(unscaled_atm,atm_cls,atm_kwargs):
    scalings = [lambda wavelength: 3,lambda wavelength: wavelength/3]
    dist = 1*constants.au
    for scaling in scalings:
        #make a copy in order not to modify atm_kwargs
        scaled_atm_kwargs = atm_kwargs.copy()
        scaled_atm_kwargs['scaling'] = scaling
        scaled_atm = atm_cls(**scaled_atm_kwargs)
        scaling_factors = scaling(wavelength=scaled_atm.lambda_grid)
        assert np.all(unscaled_atm.modelflux*scaling_factors == scaled_atm.modelflux)
        flux_kwargs = {'wavelength':scaled_atm.lambda_grid,'distance':dist}
        unscaled_flux = unscaled_atm.flux(**flux_kwargs)
        scaled_flux = scaled_atm.flux(**flux_kwargs)
        assert np.allclose(scaling_factors*unscaled_flux,scaled_flux,rtol=1e-6,atol=0)
        expected_lum = np.trapz(unscaled_flux*scaling_factors,unscaled_atm.lambda_grid)\
                                                               *4*np.pi*dist**2
        assert np.isclose(expected_lum,scaled_atm.luminosity(),rtol=1e-6,atol=0)

class Test_ATLAS():

    test_grid = np.array((1,2,3,4))
    R_Sun = 6.955e8
    template_init_kwargs = {'Teff':5780,'metallicity':-0.6,'logg':4.3,'Rstar':R_Sun,
                            'calibration_spec':None,'verbose':True}
    template_atm = ip.ATLASModelAtmosphere(**template_init_kwargs)

    def test_grid_checking(self):
        for value in (0,5):
            with pytest.raises(AssertionError):
                ip.ATLASModelAtmosphere.assert_within_grid(value=value,
                                                           grid=self.test_grid)

    def test_grid_searching(self):
        values = np.array((0,2,2.9,7))
        expected_grid_values = np.array((1,2,3,4))
        for v,expec_v in zip(values,expected_grid_values):
            grid_value = ip.ATLASModelAtmosphere.get_closest_grid_value(
                                              value=v,grid=self.test_grid)
            assert grid_value == expec_v

    def test_bad_grid_values(self):
        bad_init_kwargs = {'Teff':[2000,1e5],'metallicity':[-3,3],'logg':[-1,7]}
        for kw,bad_values in bad_init_kwargs.items():
            for v in bad_values:
                kwargs = self.template_init_kwargs.copy()
                kwargs[kw] = v
                with pytest.raises(AssertionError):
                    ip.ATLASModelAtmosphere(**kwargs)

    def test_initialisation(self):
        #test all None:
        kwargs = self.template_init_kwargs.copy()
        kwargs['Rstar'] = None
        with pytest.raises(ValueError):
            ip.ATLASModelAtmosphere(**kwargs)
        #test 2x not None
        for key in ('calibration_spec','obs_luminosity'):
            kwargs = self.template_init_kwargs.copy()
            assert kwargs['Rstar'] is not None
            kwargs[key] = 1
            with pytest.raises(AssertionError):
                ip.ATLASModelAtmosphere(**kwargs)
        #test 3x not None
        kwargs = self.template_init_kwargs.copy()
        assert kwargs['Rstar'] is not None
        kwargs['calibration_spec'] = 1
        kwargs['obs_luminosity'] = 1
        with pytest.raises(AssertionError):
            ip.ATLASModelAtmosphere(**kwargs)

    def test_adopted_grid_values(self):
        assert self.template_atm.Teff == 5750
        assert self.template_atm.metallicity == -0.5
        assert self.template_atm.logg == 4.5

    def test_positive_metallicity(self):
        kwargs = self.template_init_kwargs.copy()
        kwargs['metallicity'] = 0.3
        ip.ATLASModelAtmosphere(**kwargs)

    def test_distance_scaling(self):
        test_lamb = 700*constants.nano
        assert self.template_atm.flux(wavelength=test_lamb,distance=1*constants.au)\
                == 4*self.template_atm.flux(wavelength=test_lamb,distance=2*constants.au)

    def test_RJ_extrapolation(self):
        atm = ip.ATLASModelAtmosphere(**self.template_init_kwargs)
        test_lamb = 2*constants.milli
        dist = 1*constants.au
        RJ_flux = atm.flux(wavelength=test_lamb,distance=dist)
        expected_flux = atm.modelflux[-1]*(atm.lambda_grid[-1]/test_lamb)**4\
                         *(atm.ref_distance/dist)**2
        assert np.isclose(RJ_flux,expected_flux,rtol=1e-3,atol=0)

    def test_calibration_with_luminosity(self):
        test_lum = 3
        test_kwargs = self.template_init_kwargs.copy()
        test_kwargs["Rstar"] = None
        test_kwargs['obs_luminosity'] = test_lum
        test_atm = ip.ATLASModelAtmosphere(**test_kwargs)
        assert test_lum == test_atm.luminosity()

    def test_calibration_with_spec(self):
        ref_atm = ip.ATLASModelAtmosphere(**self.template_init_kwargs)
        calibration_spec = {'wave':ref_atm.lambda_grid,'flux':ref_atm.modelflux/2,
                            'ref_distance':ref_atm.ref_distance}
        kwargs = self.template_init_kwargs.copy()
        kwargs['Rstar'] = None
        kwargs['calibration_spec'] = calibration_spec
        atm = ip.ATLASModelAtmosphere(**kwargs)
        expected_scaling = 0.5
        assert np.isclose(atm.spec_calibration_scaling,expected_scaling,rtol=1e-4,atol=0)

    def test_flux_scaling(self):
        general_test_flux_scaling(unscaled_atm=self.template_atm,
                                  atm_cls=ip.ATLASModelAtmosphere,
                                  atm_kwargs=self.template_init_kwargs)

    def test_plot_model(self):
        atm = ip.ATLASModelAtmosphere(**self.template_init_kwargs)
        atm.plot_model()


class Test_betaPic():

    betaPic = ip.betaPicObsSpectrum()

    def test_flux_reading(self):    
        test_lamb = 1327.75*constants.angstrom
        expected_flux = 10**(1.65467e-001)*constants.erg/constants.centi**2/constants.angstrom
        flux = self.betaPic.flux(wavelength=test_lamb,distance=1*constants.au)
        assert expected_flux == flux

    def test_dilution(self):
        betaPic_diluted = ip.betaPicObsSpectrum(dilution=3)
        assert self.betaPic.lambda_grid.size > betaPic_diluted.lambda_grid.size
        betaPic_int = np.trapz(self.betaPic.modelflux,self.betaPic.lambda_grid)
        betaPic_diluted_int = np.trapz(betaPic_diluted.modelflux,
                                       betaPic_diluted.lambda_grid)
        assert np.isclose(betaPic_int,betaPic_diluted_int,rtol=1e-3,atol=0)

    def test_scaling(self):
        general_test_flux_scaling(
                     unscaled_atm=self.betaPic,atm_cls=ip.betaPicObsSpectrum,
                     atm_kwargs={})

    def test_plot_model(self):
        self.betaPic.plot_model()


class Test_pd_cross_section():

    cs = ip.PhotodissociationCrossSection('CO.hdf5')

    def test_crosssection(self):
        too_small_lamb = 5e-9
        too_large_lamb = 1.8e-7
        for lamb in (too_small_lamb,too_large_lamb):
            assert self.cs.crosssection(lamb) == 0

    def test_plot(self):
        self.cs.plot()


def test_Osterbrock_ionisation_and_recombination():
    for element in ip.ionisation_potential.keys():
        ionisation = ip.OsterbrockIonisationCrossSection(element=element)
        nu_min = ip.ionisation_potential[element]/constants.h
        lamb_max = constants.c/nu_min
        assert ionisation.crosssection(wavelength=10*lamb_max) == 0
        recomb = ip.OsterbrockRecombination(element=element)
        too_small_T, too_large_T = 7000, 16000
        for T in (too_small_T,too_large_T):
            with pytest.raises(AssertionError):
                recomb.recombination_coeff(T)


def test_Nahar_ionisation_and_recombination():
    for element,io_potential in ip.ionisation_potential.items():
        ionisation = ip.NaharIonisationCrossSection(element=element)
        non_ionising_wavelength = 1.1*constants.h*constants.c/io_potential
        assert ionisation.crosssection(non_ionising_wavelength) == 0
        recomb = ip.NaharRecombination(element=element)
        too_small_T, too_large_T = 1, 10e10
        for T in (too_small_T,too_large_T):
            with pytest.raises(AssertionError):
                recomb.recombination_coeff(T)


solar_atmosphere = ip.ATLASModelAtmosphere(
                          Teff=5780,metallicity=0.01,logg=4.43,Rstar=6.955e8)

def pd_crosssections_iterator():
    for mol in ('CO','H2O','OH'):
        yield ip.PhotodissociationCrossSection(f'{mol}.hdf5')
def io_crosssections_iterator():
    for element in ('C','O'):
        for cs in (ip.NaharIonisationCrossSection(element=element),
                   ip.OsterbrockIonisationCrossSection(element=element)):
            yield cs
isf = ip.ISF()


class Test_rate():

    def rate_iterator(self):
        for crosssection in itertools.chain(pd_crosssections_iterator(),
                                            io_crosssections_iterator()):
            for atm in (solar_atmosphere,None):
                rate = ip.Rate(stellar_atmosphere=atm,crosssection=crosssection)
                yield crosssection,atm,rate

    def test_lambda_grid_construction(self):
        for crosssection,atm,rate in self.rate_iterator():
            assert np.all(np.diff(rate.lambda_grid)>0)
            assert np.all(np.unique(rate.lambda_grid) == rate.lambda_grid)
            assert np.all(np.isin(isf.lambda_grid,rate.lambda_grid))
            if hasattr(crosssection,'lambda_grid'):
                assert np.all(np.isin(crosssection.lambda_grid,rate.lambda_grid))
            if atm is not None:
                assert np.all(np.isin(atm.lambda_grid,rate.lambda_grid))
            if not hasattr(crosssection,'lambda_grid') and atm is None:
                assert np.all(rate.lambda_grid==isf.lambda_grid)

    def test_stellar_rate_distance_scaling(self):
        for crosssection,atm,rate in self.rate_iterator():
            if atm is not None:
                assert rate.stellar_rate(distance=1*constants.au)\
                              == 4*rate.stellar_rate(distance=2*constants.au)

    def test_stellar_rate_flux_scaling(self):
        dist = 1*constants.au
        scaling = lambda wavelength: 3.5
        atm_kwargs = {'Teff':6000,'metallicity':0.01,'logg':4.2,'Rstar':7e8}
        for crosssection in itertools.chain(pd_crosssections_iterator(),
                                            io_crosssections_iterator()):
            atm = ip.ATLASModelAtmosphere(**atm_kwargs)
            original_rate = ip.Rate(stellar_atmosphere=atm,crosssection=crosssection)
            original_stellar_rate = original_rate.stellar_rate(distance=dist)
            scaled_atm = ip.ATLASModelAtmosphere(scaling=scaling,**atm_kwargs)
            scaled_rate = ip.Rate(stellar_atmosphere=scaled_atm,crosssection=crosssection)
            scaled_stellar_rate = scaled_rate.stellar_rate(distance=dist)
            scaling_factors = scaling(wavelength=atm.lambda_grid)
            assert np.isclose(scaled_stellar_rate,scaling_factors*original_stellar_rate,
                              rtol=1e-6,atol=0)

    def test_ISF_scaling(self):
        ISF_scaling=6.7
        for crosssection,atm,rate in self.rate_iterator():
            unscaled_isf_rate = rate.isf_rate
            scaled_rate = ip.Rate(stellar_atmosphere=atm,crosssection=crosssection,
                                  ISF_scaling=ISF_scaling)
            assert np.isclose(ISF_scaling*unscaled_isf_rate,scaled_rate.isf_rate,
                              rtol=1e-6,atol=0)

    def test_rate(self):
        for crosssection,atm,rate in self.rate_iterator():
            rate.isf_rate
            if atm is not None:
                rate.stellar_rate(distance=1*constants.au)

    def test_plot_print(self):
        for crosssection,atm,rate in self.rate_iterator():
            rate = ip.Rate(stellar_atmosphere=solar_atmosphere,
                           crosssection=crosssection)
            rate.plot_ISF_rate()
            dist = 1*constants.au
            rate.plot_stellar_rate(distance=dist)
            rate.print_rates(distance=dist)


class Test_pd_and_io_rate():

    def rate_iterator(self):
        for crosssection in pd_crosssections_iterator():
            for rate_cls in (ip.PhotodissociationRate,ip.IonisationRate):
                rate = rate_cls(stellar_atmosphere=solar_atmosphere,
                                crosssection=crosssection)
            yield crosssection,rate

    def test_total_rate(self):
        for crosssection,rate in self.rate_iterator():
            dist = 1*constants.au
            stellar_rate = rate.stellar_rate(distance=dist)
            if isinstance(rate,ip.PhotodissociationRate):
                CR_rate = 0
            elif isinstance(rate,ip.IonisationRate):
                CR_rate = ip.IonisationRate.CR_ionisation_rate
            tot_rate = rate.isf_rate + stellar_rate + CR_rate
            assert rate.total_rate(distance=dist) == tot_rate

    def test_print(self):
        for crosssection,rate in self.rate_iterator():
            rate = ip.PhotodissociationRate(stellar_atmosphere=solar_atmosphere,
                                            crosssection=crosssection)
            rate.print_rates(distance=1*constants.au)


class Test_ionisation_balance():

    elements = ('C','O')
    recombinations = [ip.NaharRecombination,ip.OsterbrockRecombination]
    io_crosssections = [ip.NaharIonisationCrossSection,
                        ip.OsterbrockIonisationCrossSection]

    def io_balance_iterator(self):
        for element in self.elements:
            for recomb,io_cs in itertools.product(self.recombinations,
                                                  self.io_crosssections):
                recombination = recomb(element=element)
                io_rate = ip.IonisationRate(crosssection=io_cs(element=element),
                                            ISF_scaling=1,
                                            stellar_atmosphere=solar_atmosphere)
                balance = ip.IonisationBalance(ionisation_rate=io_rate,
                                               recombination=recombination)
                yield balance

    def test_balance_computations(self):
        #don't know how to test this properly, so mostly just call all functions
        #to make sure that they at least execute without throwing errors
        n = 500/constants.centi**3
        n_ion = 300/constants.centi**3
        n_neutral = n-n_ion
        n_e = 100/constants.centi**3
        distance = 100*constants.au
        T = 8000
        for balance in self.io_balance_iterator():
            balance.get_ionisation_rate(distance=distance)
            balance.get_recomb_coeff(T=T)
            balance.determine_n_neutral(n_ion=n_ion,n_e=n_e,distance=distance,T=T)
            n_neu_e_from_ion = balance.determine_n_neutral_e_from_ion(
                                 n_ion=n_ion,distance=distance,T=T)
            n_neu_e_from_ion_check = balance.determine_n_neutral(
                                       n_ion=n_ion,n_e=n_ion,distance=distance,T=T)
            assert n_neu_e_from_ion == n_neu_e_from_ion_check
            balance.determine_n_ion(n_neutral=n_neutral,n_e=n_e,distance=distance,T=T)
            n_ion_e_from_ion = balance.determine_n_ion_e_from_ion(
                                  n_neutral=n_neutral,distance=distance,T=T)
            n_ion_e_from_ion_check = balance.determine_n_ion(
                                      n_neutral=n_neutral,n_e=n_ion_e_from_ion,
                                      distance=distance,T=T)
            assert np.isclose(n_ion_e_from_ion,n_ion_e_from_ion_check,rtol=1e-6,atol=0)
            balance.determine_ionisation_balance(n=n,n_e=n_e,distance=distance,T=T)
            balance.determine_ionisation_balance_e_from_ion(n=n,distance=distance,T=T)
    
    def test_ionisation_fraction(self):
        for balance in self.io_balance_iterator():
            assert balance.ionisation_fraction(n_neutral=0.3,n_ion=0.7) == 0.7