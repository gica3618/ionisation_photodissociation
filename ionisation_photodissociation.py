#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 08:44:52 2021

@author: gianni
"""

from scipy import constants,optimize
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import h5py
this_folder = os.path.dirname(os.path.abspath(__file__))

R_Sun = 6.955e8
L_Sun = 3.828e26
Rydberg_J = constants.physical_constants['Rydberg constant times hc in J'][0] #J
ionisation_potential = {'C':11.26030*constants.eV, 'O':13.61806*constants.eV} #J


class RadiationSpectrum():
    
    def flux(self,wavelength,**kwargs):
        #W/m2/m
        raise NotImplementedError


class ISF(RadiationSpectrum):
    #interstellar radiation field, original from Draine (1978),
    #here in the form of Lee (1984)
    #(https://ui.adsabs.harvard.edu/abs/1984ApJ...282..172L/abstract)
    lambda_min = 91.2*constants.nano
    lambda_max = 200*constants.nano
    lambda_grid = np.linspace(lambda_min,lambda_max,1000)

    def __init__(self,scaling=(lambda wavelength: 1)):
        self.scaling = scaling

    def flux(self,wavelength):
        #for the power law, the wavelenght has to be in nm
        #photons/m2/s/m:
        photon_flux= 3.2e13*((wavelength/constants.nano)**-3\
                     - 1.61e2*(wavelength/constants.nano)**-4\
                     + 6.41e3*(wavelength/constants.nano)**-5)\
                                   * constants.centi**-2*constants.nano**-1
        photon_energy = constants.h*constants.c/wavelength
        flux = photon_flux*photon_energy
        valid_region = (wavelength>=self.lambda_min) & (wavelength<=self.lambda_max)
        flux = np.where(valid_region,flux,0)
        return flux*self.scaling(wavelength=wavelength)


class StellarAtmosphere(RadiationSpectrum):

    def plot_model(self,label=None):
        fig,ax = plt.subplots()
        ax.plot(self.lambda_grid/constants.nano,self.modelflux,'.-',label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('lambda [nm]')
        ax.set_ylabel('flux at {:g} au [W/m2/m]'.format(self.ref_distance/constants.au))
        if label is not None:
            ax.legend(loc='best')
        return ax

    def flux(self,wavelength,distance):
        return np.interp(wavelength,self.lambda_grid,self.modelflux,left=0,right=0)\
                                      * (self.ref_distance/distance)**2

    def luminosity(self):
        flux_at_ref_distance = self.flux(wavelength=self.lambda_grid,
                                         distance=self.ref_distance)
        return np.trapz(flux_at_ref_distance,self.lambda_grid)\
                                       * 4*np.pi*self.ref_distance**2

    def _scale_spectrum(self,scaling):
        self.modelflux *= scaling(wavelength=self.lambda_grid)

    def write_modelflux_to_file(self,filepath,distance):
        flux = self.flux(wavelength=self.lambda_grid,distance=distance)
        np.savez(filepath,wavelength=self.lambda_grid,flux=flux)


class ATLASModelAtmosphere(StellarAtmosphere):

    Teff_low_grid = np.arange(3000,12999,250)
    Teff_high_grid = np.arange(13000,50001,1000)
    Teff_grid = np.concatenate((Teff_low_grid,Teff_high_grid))
    metallicity_grid = np.array((-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.2,0.5))
    logg_grid = np.arange(0,5.1,0.5)
    model_folder = os.path.join(this_folder,'ck04models')
    max_RJ_wavelength = 3*constants.milli

    @staticmethod
    def assert_within_grid(value,grid):
        assert np.min(grid) <= value <= np.max(grid)

    @staticmethod
    def get_closest_grid_value(value,grid):
        index = np.argmin(np.abs(grid-value))
        return grid[index]

    def __init__(self,Teff,metallicity,logg,Rstar=None,obs_luminosity=None,
                 calibration_spec=None,verbose=False,scaling=None):
        '''There are three ways to set the luminosity of the star:
            1) define Rstar
            2) define obs_luminosity, so that the model flux will be scaled
            3) define calibration (i.e. a spectrum, to which the model spectrum
                will be scaled to)'''
        self.assert_within_grid(value=Teff,grid=self.Teff_grid)
        self.assert_within_grid(value=metallicity,grid=self.metallicity_grid)
        self.assert_within_grid(value=logg,grid=self.logg_grid)
        self.verbose = verbose
        self.read_model(metallicity=metallicity,Teff=Teff,logg=logg)
        self.extrapolate_RJ()
        if Rstar is not None:
            assert obs_luminosity is None and calibration_spec is None
            self.ref_distance = Rstar
        elif obs_luminosity is not None:
            assert Rstar is None and calibration_spec is None
            self.obs_luminosity = obs_luminosity
            print('Rstar not specified, going to scale with luminosity')
            self.calibrate_with_luminosity()
        elif calibration_spec is not None:
            assert Rstar is None and obs_luminosity is None
            self.calibration_spec = calibration_spec
            print('going to calibrate with provided spectrum')
            self.calibrate_with_spectrum()
        else:
            raise ValueError('unable to define absolute flux and/or reference distance')
        #now that modelflux is calibrated, I can apply the scaling:
        if scaling is not None:
            self._scale_spectrum(scaling=scaling)

    def read_model(self,metallicity,Teff,logg):
        self.metallicity = self.get_closest_grid_value(
                             value=metallicity,grid=self.metallicity_grid)
        self.Teff = self.get_closest_grid_value(value=Teff,grid=self.Teff_grid)
        self.logg = self.get_closest_grid_value(value=logg,grid=self.logg_grid)
        if self.verbose:
            print('input metallicity = {:g}, grid metallicity = {:g}'\
                   .format(metallicity,self.metallicity))
            print('input Teff = {:g} K, grid Teff = {:g} K'.format(Teff,self.Teff))
            print('input logg = {:g}, grid logg = {:g}'.format(logg,self.logg))
        self.metallicity_str = 'ck'
        if self.metallicity < 0:
            sign_str = 'm'
        else:
            sign_str = 'p'
        self.metallicity_str += '{:s}{:02d}'.format(
                                        sign_str,np.abs(int(10*self.metallicity)))
        if self.verbose:
            print('metallicity ID: {:s}'.format(self.metallicity_str))
        #this string is the key to access the flux for the specified log(g);
        #for example for log(g)=4, it would be "g40"; for log(g)=4.5 it would be "g45":
        logg_string = ('g%.1f'%self.logg).replace('.','')
        filename = self.metallicity_str+'_{:d}.fits'.format(int(self.Teff))
        if self.verbose:
            print('filename: {:s}'.format(filename))
        filepath = os.path.join(self.model_folder,self.metallicity_str,filename)
        hdulist = fits.open(filepath)
        modeldata = hdulist[1].data
        hdulist.close()
        self.lambda_grid = modeldata['WAVELENGTH'].astype(np.float64)*constants.angstrom
        #flux in [W/m2/m] at the stellar surface:
        self.modelflux = modeldata[logg_string].astype(np.float64)\
                              *constants.erg/constants.centi**2/constants.angstrom

    def extrapolate_RJ(self):
        max_wavelength = self.lambda_grid[-1]
        prop_constant = max_wavelength**4*self.modelflux[-1]
        RJ_wavelength = np.logspace(np.log10(max_wavelength*1.05),
                                    np.log10(self.max_RJ_wavelength),100)
        RJ_flux = prop_constant/RJ_wavelength**4
        self.original_lambda_grid = self.lambda_grid.copy()
        self.lambda_grid = np.concatenate((self.lambda_grid,RJ_wavelength))
        self.modelflux = np.concatenate((self.modelflux,RJ_flux))

    def calibrate_with_luminosity(self):
        self.ref_distance = 1*constants.au
        uncalibrated_luminosity = self.luminosity()
        self.modelflux *= self.obs_luminosity/uncalibrated_luminosity
        assert np.isclose(self.obs_luminosity,self.luminosity(),rtol=1e-6,atol=0)

    def calibrate_with_spectrum(self):
        cal_wave = self.calibration_spec['wave']
        cal_flux = self.calibration_spec['flux']
        self.ref_distance = self.calibration_spec['ref_distance']
        try:
            cal_errors = self.calibration_spec['error']
        except KeyError:
            cal_errors = np.ones_like(cal_flux)
        def residual2(scaling):
            flux = self.flux(wavelength=cal_wave,
                             distance=self.calibration_spec['ref_distance'])
            scaled_model_flux = scaling*flux
            res = cal_flux-scaled_model_flux
            return np.sum(res**2/cal_errors**2)
        x0 = 1
        optimisation = optimize.minimize(residual2,x0,method='Nelder-Mead')
        assert optimisation.success
        self.spec_calibration_scaling = optimisation.x[0]
        if self.verbose:
            print('optimal calibration scaling: {:g}'.format(
                                     self.spec_calibration_scaling))
        self.modelflux *= self.spec_calibration_scaling

    def plot_model(self,label=None):
        ax = StellarAtmosphere.plot_model(self,label='final flux')
        if hasattr(self,'calibration_scaling'):
            ax.plot(self.lambda_grid/constants.nano,self.modelflux,'.-',
                    label='before calibration')
            plot_cal_flux = self.calibration['flux']\
                              *(self.calibration['ref_distance']/self.ref_distance)**2
            ax.plot(self.calibration['wave']/constants.nano,plot_cal_flux,
                    label='calibration')
        for lamb,lab in zip((self.original_lambda_grid[-1],self.max_RJ_wavelength),
                            ('RJ region',None)):
            ax.axvline(lamb/constants.nano,color='black',linestyle='dashed',label=lab)
            ax.legend(loc='best')
                    

class betaPicObsSpectrum(StellarAtmosphere):
    #from Alexis email
    model_filepath = 'bPicNormFlux1AU.txt'
    cutoff_flux = 15832
    max_cutoff_wavelength = 1*constants.micro

    def __init__(self,dilution=1,scaling=None):
        self.ref_distance = 1*constants.au
        model_data = np.loadtxt(self.model_filepath)
        data_wave = model_data[:,0]*constants.angstrom
        data_flux = 10**model_data[:,1]*constants.erg/constants.centi**2\
                          /constants.angstrom #W/m2/m
        self.min_betaPic_data_wave = np.min(data_wave)
        self.max_betaPic_data_wave = np.max(data_wave)
        betaPic_ATLAS_atm = ATLASModelAtmosphere(Teff=8052,metallicity=0.05,logg=4.15,
                                                 obs_luminosity=8.7*L_Sun)
        left_ATLAS_region = betaPic_ATLAS_atm.lambda_grid < self.min_betaPic_data_wave
        left_ATLAS_wave = betaPic_ATLAS_atm.lambda_grid[left_ATLAS_region]
        right_ATLAS_region = betaPic_ATLAS_atm.lambda_grid > self.max_betaPic_data_wave
        right_ATLAS_wave = betaPic_ATLAS_atm.lambda_grid[right_ATLAS_region]
        self.lambda_grid = np.concatenate((left_ATLAS_wave,data_wave,right_ATLAS_wave))
        left_ATLAS_flux = betaPic_ATLAS_atm.flux(wavelength=left_ATLAS_wave,
                                                      distance=self.ref_distance)
        right_ATLAS_flux = betaPic_ATLAS_atm.flux(wavelength=right_ATLAS_wave,
                                                       distance=self.ref_distance)
        self.modelflux = np.concatenate((left_ATLAS_flux,data_flux,right_ATLAS_flux))
        #apply dilution:
        self.lambda_grid = self.lambda_grid[::dilution]
        self.modelflux = self.modelflux[::dilution]
        #the original spectrum has some unphysical plateaus, so just put those to 0
        #(although to be honest, who knows if that's better...)
        cutoff_region = (self.modelflux<self.cutoff_flux)\
                            & (self.lambda_grid<self.max_cutoff_wavelength)
        self.modelflux[cutoff_region] = 0
        if scaling is not None:
            self._scale_spectrum(scaling=scaling)

    def plot_model(self):
        ax = StellarAtmosphere.plot_model(self,label='final beta Pic flux')
        for lamb,lab in zip((self.min_betaPic_data_wave,self.max_betaPic_data_wave),
                            ('beta Pic data region',None)):
            ax.axvline(lamb/constants.nano,linestyle='dashed',color='red',label=lab)
        ax.legend(loc='best')


class CrossSection():

    def crosssection(self,wavelength):
        raise NotImplementedError

    def plot(self,wavelength=None,title=None):
        lamb = self.lambda_grid if wavelength is None else wavelength
        fig,ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.plot(lamb/constants.micro,self.crosssection(lamb))
        ax.set_xlabel('wavelength [um]')
        ax.set_ylabel('cs [m2]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        return ax


class PhotodissociationCrossSection(CrossSection):

    data_folderpath = os.path.join(this_folder,'crosssections')

    def __init__(self,filename):
        data = h5py.File(os.path.join(self.data_folderpath,filename),'r')
        self.lambda_grid = data['wavelength'][()] * constants.nano
        self.data_cs = data['photodissociation'][()] *constants.centi**2
        self.min_data_wavelength = np.min(self.lambda_grid)
        self.max_data_wavelength = np.max(self.lambda_grid)

    def crosssection(self,wavelength):
        return np.interp(wavelength,self.lambda_grid,self.data_cs,left=0,
                         right=0)


Osterbrock_a_T = {'C':12.2e-18*constants.centi**2,'O':2.94e-18*constants.centi**2}
Osterbrock_s = {'C':2,'O':1}
Osterbrock_beta = {'C':3.32,'O':2.66}
Osterbrock_alpha_R = {'C':4.66e-13*constants.centi**3,'O':3.31e-13*constants.centi**3}#m3/s
Osterbrock_f = {'C':0.5,'O':0.26}
Osterbrock_alpha_d = {'C':np.array([1.91e-13,1.84e-14,1.63e-13])*constants.centi**3,
                      'O':np.array([7.35e-14,7.62e-14,7.74e-14])*constants.centi**3}
Osterbrock_alpha_d_T = np.array([7500,10000,15000])


class OsterbrockIonisationCrossSection(CrossSection):

    def __init__(self,element):
        self.element = element
        self.a_T = Osterbrock_a_T[self.element]
        self.s = Osterbrock_s[self.element]
        self.beta = Osterbrock_beta[self.element]
        self.nu_T = ionisation_potential[self.element]/constants.h #Hz

    def crosssection(self,wavelength):
        nu = constants.c/wavelength
        cs = self.a_T * (self.beta*(nu/self.nu_T)**-self.s
                          + (1-self.beta)*(nu/self.nu_T)**(-self.s-1))
        return np.where(nu>=self.nu_T,cs,0) #m2


class OsterbrockRecombination():

    def __init__(self,element):
        self.element = element
        self.alpha_R = Osterbrock_alpha_R[self.element]
        self.f = Osterbrock_f[self.element]
        self.alpha_d = Osterbrock_alpha_d[self.element]
        self.min_T,self.max_T = np.min(Osterbrock_alpha_d_T),np.max(Osterbrock_alpha_d_T)

    @staticmethod
    def phi(T):
        return np.interp(T,[5000,7500,10000,15000,20000],[1.317,1.13,1,.839,.732])

    def interpolated_alpha_d(self,T):
        assert np.all(self.min_T <= T) and np.all(T <= self.max_T),\
                             'requested temperature out of interpolation range'
        return np.interp(T,Osterbrock_alpha_d_T,self.alpha_d)

    def recombination_coeff(self,T):
        return self.alpha_R*np.sqrt(10000./T)*(self.f+(1-self.f)*self.phi(T))\
               + self.interpolated_alpha_d(T) #m3/s


Nahar_data_folderpath = os.path.join(this_folder,'Nahar_atomic_data')

class NaharIonisationCrossSection(CrossSection):

    def __init__(self,element):
        self.element = element
        filepath = os.path.join(
                     Nahar_data_folderpath,
                     '{:s}I_photoionisation_cs_groundstate.txt'.format(self.element))
        cs_data = np.loadtxt(filepath)
        #invert with ::-1 to have increasing wavelength
        self.energy = cs_data[:,0][::-1] * Rydberg_J
        self.cs = cs_data[:,1][::-1] * constants.mega*1e-28 #m2
        self.lambda_grid = constants.h*constants.c/self.energy
        assert np.all(np.diff(self.lambda_grid)>0)

    def crosssection(self,wavelength):
        E = constants.h*constants.c/wavelength
        above_ionisation = E >= ionisation_potential[self.element]
        interp_cs = np.interp(wavelength,self.lambda_grid,self.cs,left=0,right=0)
        return np.where(above_ionisation,interp_cs,0) #m2


class NaharRecombination():

    def __init__(self,element):
        self.element = element
        filepath = os.path.join(Nahar_data_folderpath,
                               '{:s}I_recombination_total.txt'.format(self.element))
        recomb_data = np.loadtxt(filepath)
        self.logT = recomb_data[:,0]
        self.recomb = recomb_data[:,-1]*constants.centi**3 #m3/s

    def recombination_coeff(self,T):
        logT = np.log10(T)
        assert np.all(logT >= np.min(self.logT)) and np.all(logT < np.max(self.logT))
        return np.interp(logT,self.logT,self.recomb,left=np.nan,right=np.nan)


class Rate():

    def __init__(self,crosssection,ISF_scaling=(lambda wavelength: 1),
                 stellar_atmosphere=None):
        self.crosssection = crosssection
        self.isf = ISF(scaling=ISF_scaling)
        self.stellar_atmosphere = stellar_atmosphere
        self.construct_lambda_grid()
        self.compute_ref_rates()

    def construct_lambda_grid(self):
        #make the lambda_grid a combination of the lambda_grids of
        #crosssection, isf and stellar_atmosphere, to be sure to capture all features
        lambda_grids = [self.isf.lambda_grid,]
        if hasattr(self.crosssection,'lambda_grid'):
            lambda_grids.append(self.crosssection.lambda_grid)
        if self.stellar_atmosphere is not None:
            lambda_grids.append(self.stellar_atmosphere.lambda_grid)
        self.lambda_grid = np.concatenate(lambda_grids)
        self.lambda_grid = np.unique(self.lambda_grid)

    def rate(self,flux,**flux_kwargs):
        photon_energy = constants.h*constants.c/self.lambda_grid
        photon_flux = flux(wavelength=self.lambda_grid,**flux_kwargs)/photon_energy
        cs = self.crosssection.crosssection(wavelength=self.lambda_grid)
        return np.trapz(photon_flux*cs,self.lambda_grid)

    def compute_ref_rates(self):
        self.isf_rate = self.rate(flux=self.isf.flux)
        if self.stellar_atmosphere is not None:
            self.stellar_rate_at_ref_distance = self.rate(
                                       flux=self.stellar_atmosphere.flux,
                                       distance=self.stellar_atmosphere.ref_distance)

    def stellar_rate(self,distance):
        dist_scaling = (self.stellar_atmosphere.ref_distance/distance)**2
        return self.stellar_rate_at_ref_distance*dist_scaling

    def plot_rate_per_wavelength(self,flux,title=None,**flux_kwargs):
        photon_energy = constants.h*constants.c/self.lambda_grid
        photon_flux = flux(self.lambda_grid,**flux_kwargs)/photon_energy #photons/s/m2/m
        cs = self.crosssection.crosssection(wavelength=self.lambda_grid)
        rate_per_lamb = photon_flux*cs
        plt.figure()
        plt.title(title)
        plt.plot(self.lambda_grid/constants.micro,rate_per_lamb*self.lambda_grid)
        plt.xlabel('wavelength [um]')
        plt.ylabel('rate per log(wavelength)')
        plt.xscale('log')
        plt.yscale('log')

    def plot_ISF_rate(self):
        self.plot_rate_per_wavelength(flux=self.isf.flux,title='isf')

    def plot_stellar_rate(self,distance):
        self.plot_rate_per_wavelength(flux=self.stellar_atmosphere.flux,title='star',
                                      distance=distance)

    @staticmethod
    def print_rate(ID,rate):
        print(f'{ID} rate: {rate} s-1')
        lifetime = 1/rate
        print('{:s} lifetime: {:g} days ({:g} years)'.format(ID,lifetime/constants.day,
                                                              lifetime/constants.year))

    def print_rates(self,distance=None):
        self.print_rate(ID='ISF',rate=self.isf_rate)
        if self.stellar_atmosphere is not None:
            stellar_rate = self.stellar_rate(distance=distance)
            self.print_rate(ID='stellar',rate=stellar_rate)


class PhotodissociationRate(Rate):

    def __init__(self,crosssection,stellar_atmosphere,
                 ISF_scaling=(lambda wavelength: 1)):
        Rate.__init__(self,crosssection=crosssection,ISF_scaling=ISF_scaling,
                      stellar_atmosphere=stellar_atmosphere)

    def total_rate(self,distance):
        stellar_rate = self.stellar_rate(distance=distance)
        return self.isf_rate + stellar_rate

    def print_rates(self,distance):
        Rate.print_rates(self,distance=distance)
        tot_rate = self.total_rate(distance)
        Rate.print_rate(ID='total',rate=tot_rate)


class IonisationRate(Rate):

    CR_ionisation_rate=1e-16 # CR ionisations/s/atom; Indriolo 2012

    def total_rate(self,distance):
        stellar_rate = self.stellar_rate(distance=distance)
        return self.isf_rate + stellar_rate + self.CR_ionisation_rate

    def print_rates(self,distance):
        Rate.print_rates(self,distance=distance)
        Rate.print_rate(ID='CR',rate=self.CR_ionisation_rate)
        tot_rate = self.total_rate(distance)
        Rate.print_rate(ID='total',rate=tot_rate)


class IonisationBalance():

    def __init__(self,ionisation_rate,recombination):
        self.ionisation_rate = ionisation_rate
        self.recombination = recombination

    def get_ionisation_rate(self,distance):
        return self.ionisation_rate.total_rate(distance=distance)

    def get_recomb_coeff(self,T):
        return self.recombination.recombination_coeff(T=T)

    def determine_n_neutral(self,n_ion,n_e,distance,T):
        return n_ion*n_e*self.get_recomb_coeff(T=T)\
                              /self.get_ionisation_rate(distance=distance)

    def determine_n_neutral_e_from_ion(self,n_ion,distance,T):
        '''Assuming that the ion density equals the electron density'''
        return self.determine_n_neutral(n_ion=n_ion,n_e=n_ion,distance=distance,T=T)

    def determine_n_ion(self,n_neutral,n_e,distance,T):
        return n_neutral*self.get_ionisation_rate(distance=distance)\
                                     /(n_e*self.get_recomb_coeff(T))

    def determine_n_ion_e_from_ion(self,n_neutral,distance,T):
        recomb_coeff = self.get_recomb_coeff(T)
        io_rate = self.get_ionisation_rate(distance=distance)
        return np.sqrt(n_neutral*io_rate/recomb_coeff)

    @staticmethod
    def densities(n,n_neutral):
        n_ion = n-n_neutral
        return {'n_neutral':n_neutral,'n_ion':n_ion}

    @staticmethod
    def ionisation_fraction(n_neutral,n_ion):
        return n_ion/(n_neutral+n_ion)

    def determine_ionisation_balance(self,n,n_e,distance,T):
        recomb_coeff = self.get_recomb_coeff(T=T)
        io_rate = self.get_ionisation_rate(distance=distance)
        n_neutral = n*n_e*recomb_coeff / (io_rate+n_e*recomb_coeff)
        return self.densities(n=n,n_neutral=n_neutral)

    def determine_ionisation_balance_e_from_ion(self,n,distance,T):
        '''Assuming that the ion density equals the electron density'''
        recomb_coeff = self.get_recomb_coeff(T=T)
        io_rate = self.get_ionisation_rate(distance=distance)
        sqrt_term = np.sqrt(io_rate**2+4*io_rate*n*recomb_coeff)
        n_neutral = (io_rate+2*n*recomb_coeff-sqrt_term) / (2*recomb_coeff)
        return self.densities(n=n,n_neutral=n_neutral)



if __name__ == '__main__':
    sun = ATLASModelAtmosphere(Teff=5780,metallicity=0.01,logg=4.43,Rstar=6.955e8,
                               calibration_spec=None,verbose=True)
    sun.plot_model()
    betaPic = betaPicObsSpectrum()
    betaPic.plot_model()
    for mol in ('CO','OH','H2O'):
        cs = PhotodissociationCrossSection(f'{mol}.hdf5')
        cs.plot(title=f'{mol} cross section')
        pd_rate = PhotodissociationRate(stellar_atmosphere=betaPic,crosssection=cs)
        print(mol)
        pd_rate.print_rates(distance=100*constants.au)
        print('\n')
        pd_rate.plot_ISF_rate()

    T = np.linspace(7500,15000,100)
    nahar_T = np.linspace(20,1000,100)
    for element in ('C','O'):
        nahar_cs = NaharIonisationCrossSection(element)
        ost_cs = OsterbrockIonisationCrossSection(element)

        plt.figure(element)
        plt.title(f'{element} cross section')
        plt.plot(betaPic.lambda_grid/constants.micro,
                 nahar_cs.crosssection(betaPic.lambda_grid),label='nahar')
        plt.plot(betaPic.lambda_grid/constants.micro,ost_cs.crosssection(betaPic.lambda_grid),
                 label='osterbrock')
        plt.xscale('log')
        plt.xlabel('wavelength [um]')
        plt.ylabel('cross section [m2]')
        plt.legend(loc='best')

        nahar_recomb = NaharRecombination(element)
        ost_recomb = OsterbrockRecombination(element)
        plt.figure()
        plt.title(f'{element} recombination')
        plt.plot(T,nahar_recomb.recombination_coeff(T),label='nahar')
        plt.plot(T,ost_recomb.recombination_coeff(T),label='osterbrock')
        plt.xlabel('T [K]')
        plt.ylabel('recomb coeff')
        plt.legend(loc='best')

        plt.figure()
        plt.title(f'nahar {element} recombination')
        plt.plot(nahar_T,nahar_recomb.recombination_coeff(nahar_T))
        plt.xlabel('T [K]')
        plt.ylabel('recomb coeff')

        ionisation_rate = IonisationRate(crosssection=nahar_cs,
                                         ISF_scaling=lambda wavelength: 1,
                                         stellar_atmosphere=betaPic)
        balance = IonisationBalance(ionisation_rate=ionisation_rate,
                                    recombination=nahar_recomb)
        n = 500/constants.centi**3
        dist = 100*constants.au
        densities = balance.determine_ionisation_balance_e_from_ion(
                                n=n,distance=dist,T=50)
        io_frac = balance.ionisation_fraction(**densities)
        print('ionisation fraction for n_{:s}={:g} cm-3 at {:g} au: {:g}'.format(
                element,n*constants.centi**3,dist/constants.au,io_frac))