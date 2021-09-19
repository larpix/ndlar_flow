import numpy as np
import logging

from h5flow.core import H5FlowResource, resources

from module0_flow.util.compat import assert_compat_version
import module0_flow.util.units as units


class LArData(H5FlowResource):
    '''
        Provides helper functions for calculating properties of liquid argon.
        Values will be saved and/or loaded from metadata within the output file.

        Requires ``RunData`` resource within workflow.

        Parameters:
         - ``path``: ``str``, path to stored lar data within file
         - ``electron_mobility_params``: ``list``, electron mobility calculation parameters, see ``LArData.electron_mobility``

        Provides:
         - ``v_drift``: electron drift velocity in mm/us
         - ``ionization_w``: ionization W-value
         - ``density``: LAr density
         - ``ionization_recombination(dedx)``: helper function for calculating recombination factor

        Example usage::

            from h5flow.core import resources

            resources['LArData'].v_drift

        Example config::

            resources:
                - classname: LArData
                  params:
                    path: 'lar_info'

    '''
    class_version = '0.1.0'

    default_path = 'lar_info'
    default_electron_mobility_params = np.array([551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053])

    def __init__(self, **params):
        super(LArData, self).__init__(**params)

        self.path = params.get('path', self.default_path)

        self.electron_mobility_params = np.array(params.get('electron_mobility_params', self.default_electron_mobility_params))

    def init(self, source_name):
        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            # no data stored in file, generate it
            self.v_drift
            self.density
            self.ionization_w
            self.data['classname'] = self.classname
            self.data['class_version'] = self.class_version
            self.data['electron_mobility_params'] = self.electron_mobility_params
            self.data_manager.set_attrs(self.path, **self.data)
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

        logging.info(f'v_drift: {self.v_drift}')
        logging.info(f'density: {self.density}')

    @property
    def ionization_w(self):
        ''' Ionization W-value in LAr in keV/e-. Fixed value of 0.0236 '''
        if 'ionization_w' in self.data:
            return self.data['ionization_w']

        self.data['ionization_w'] = 23.6 * units.eV / units.e
        return self.ionization_w

    def ionization_recombination(self, dedx):
        '''
            Calculate recombination factor using Birks Model with parameters:

             - ``A = 0.8``
             - ``K = 0.0486`` (units = g/(MeV cm^2) kV/cm)

        '''
        A = 0.8
        K = (0.0486 * units.kV * units.g / units.MeV / (units.cm)**3)
        eps = resources['RunData'].e_field * self.density

        return A / (1 + K / eps * dedx)

    @property
    def A(self):
        return 18

    @property
    def Z(self):
        return 39.948

    @property
    def density(self):
        ''' Liquid argon density in g/mm^3. Fixed value of 0.0013962 '''
        if 'density' in self.data:
            return self.data['density']

        self.data['density'] = 0.0013962
        return self.density

    @property
    def v_drift(self):
        ''' Electron drift velocity in kV/mm '''
        if 'v_drift' in self.data:
            return self.data['v_drift']

        # get electric field from run data
        e_field = resources['RunData'].e_field

        # calculate drift velocity
        self.data['v_drift'] = self.electron_mobility(e_field) * e_field

        return self.v_drift

    def electron_mobility(self, e, t=87.17):
        '''
            Calculation of the electron mobility w.r.t temperature and electric
            field.

            References:
             - https://lar.bnl.gov/properties/trans.html (summary)
             - https://doi.org/10.1016/j.nima.2016.01.073 (parameterization)

            :param e: electric field in kV/mm

            :param t: temperature in K

            :returns: electron mobility in mm^2/kV/us

        '''
        a0, a1, a2, a3, a4, a5 = self.electron_mobility_params

        e = e / (units.kV / units.cm)
        t = t / (units.K)

        num = a0 + a1 * e + a2 * np.power(e, 1.5) + a3 * np.power(e, 2.5)
        denom = 1 + (a1 / a0) * e + a4 * np.power(e, 2) + a5 * np.power(e, 3)
        temp_corr = np.power(t / 89, -1.5)

        mu = num / denom * temp_corr

        mu = mu * ((units.cm**2) / units.V / units.s)

        return mu
