# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:33:56 2022

@author: arvensyla
"""

import sys, os, socket
import pandas as pd
import numpy as np
import logging
from glob import glob

import pyomo.environ as po

from grimsel.core.io import IO
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
from grimsel.auxiliary.aux_m_func import cols2tuplelist
import grimsel.auxiliary.maps as maps
from grimsel import _get_logger

import grimsel.grimsel_config as config

import multiprocess as mp

logger = _get_logger(__name__)
logger.setLevel('DEBUG')


class ModelLoopModifier():
    '''
    The purpose of this class is to modify the parameters of the BaseModel
    objects in dependence on the current run_id.

    This might include the modification of parameters like CO2 prices,
    capacities, but also more profound modifications like the
    redefinition of constraints.
    '''

    def __init__(self, ml):
        '''
        To initialize, the ModelBase object is made an instance attribute.
        '''

        self.ml = ml

    def select_swiss_scenarios(self, dict_ch=None):

        if dict_ch is None:
            # dict_ch = {0: 'default', 1: 'hi_gas', 2: 'hi_rem'}
            dict_ch = {8: 'default', 7: 'REF', 6: 'EMR1', 5:'EMR12', 4:'EMR12+',  3: 'REF_noNUC', 2: 'EMR1_noNUC', 0:'EMR12_noNUC', 1:'EMR12+_noNUC'      }

        slct_ch = dict_ch[self.ml.dct_step['swch']]

        self.ml.dct_vl['swch_vl'] = slct_ch

        return slct_ch
    
    def select_v2g_scenarios(self, dict_v2g=None): 
        
        ''' Select V2G scenario for archetypes for future years based on different capacities '''
        
        if dict_v2g is None:
            dict_v2g = {0: 'noV2G', 1: 'biweekly', 2: 'weekly', 3: '2xweek'}

        slct_v2g = dict_v2g[self.ml.dct_step['swv2g']]

        self.ml.dct_vl['swv2g_vl'] = slct_v2g

        return slct_v2g
    
    
    def select_ev_scenarios(self, dict_ev=None): 
        
        ''' Select EV scenario for archetypes for future years '''
        
        if dict_ev is None:
            dict_ev = {3: 'basis', 2: 'ss', 1:'full', 0: 'bass'}

        slct_ev = dict_ev[self.ml.dct_step['swev']]

        self.ml.dct_vl['swev_vl'] = slct_ev

        return slct_ev    

    def select_hp_scenarios(self, dict_hp=None): 
        
        ''' Select heat-pumps scenario for archetypes for future years '''
        
        if dict_hp is None:
            dict_hp = {2: 'bau', 1: 'fossil', 0: 'full'}

        slct_hp = dict_hp[self.ml.dct_step['swhp']]

        self.ml.dct_vl['swhp_vl'] = slct_hp

        return slct_hp
    
    def select_transmission_scenarios(self, dict_tr=None): 
        
        ''' Select transmission scenario for archetypes for future years '''
        
        if dict_tr is None:
            dict_tr = {2: 'default', 4: 'min_5pc', 3: 'plus_5pc', 1: 'plus_10pc', 0: 'plus_15pc'}

        slct_tr = dict_tr[self.ml.dct_step['swtr']]

        self.ml.dct_vl['swtr_vl'] = slct_tr

        return slct_tr
    
    def select_transmission_scenarios_ntc(self, dict_ntc=None): 
        
        ''' Select transmission capacity for CH-neighbor interconnections '''
        
        if dict_ntc is None:
            dict_ntc = {0: 'default', 1: 'min_30pc', 2: 'min_70pc', 3: 'min_11pc', 4: 'min_12pc'}

        slct_ntc = dict_ntc[self.ml.dct_step['swntc']]

        self.ml.dct_vl['swntc_vl'] = slct_ntc

        return slct_ntc
    
    def select_retrofit_scenarios(self, dict_rf=None): 
        
        ''' Select retrofit scenario for archetypes for future years '''
        
        if dict_rf is None:
            dict_rf = {2: 'default', 0: 'retr_1pc', 1: 'retr_2pc'}

        slct_rf = dict_rf[self.ml.dct_step['swrf']]

        self.ml.dct_vl['swrf_vl'] = slct_rf

        return slct_rf
    
    def select_cooling_scenarios(self, dict_cl=None): 
        
        ''' Select cooling scenario for archetypes for future years '''
        
        if dict_cl is None:
            dict_cl = {1: 'low', 1: 'mid', 0: 'high'}
            
        slct_cl = dict_cl[self.ml.dct_step['swcl']]

        self.ml.dct_vl['swcl_vl'] = slct_cl

        return slct_cl
        
    def select_demand_profile_res(self, dict_dpf=None): 
        
        ''' Select demand profile (EE), for EL archetypes for future years '''
        
        if dict_dpf is None:
            dict_dpf = {2: 'default', 1: 'ee', 0: 'best'}

        slct_dpf = dict_dpf[self.ml.dct_step['swdpf']]

        self.ml.dct_vl['swdpf_vl'] = slct_dpf

        return slct_dpf
  
    # def set_future_year(self, slct_ch, slct_ev, slct_v2g, slct_tr, slct_hp, slct_tsp, slct_rf, slct_cl, slct_dpf, dict_fy=None):
    def set_future_year(self, slct_ch, slct_ev, slct_v2g, slct_tr, slct_ntc, slct_hp, slct_rf, slct_cl, slct_dpf, dict_fy=None):
        list_hp_aw = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('HP_AW')].pp_id.tolist()
        list_hp_aw_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('STO_HT')].pp_id.tolist()
        
        list_hp_ww = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('HP_WW')].pp_id.tolist()
        
        list_dhw_aw = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_AW')].pp_id.tolist()
        list_dhw_aw_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_HP_STO')].pp_id.tolist()
        
        list_dhw_ww = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_WW')].pp_id.tolist()
        
        list_dhw_bo = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_BO')].pp_id.to_list()
        list_dhw_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_STO')].pp_id.to_list()
        
        df_pho = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('_PHO')]
        list_pho_arch = df_pho.loc[~df_pho.pp.str.contains('SOL')].pp_id.tolist()

        df_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('_STO')]
        
        # list_sto_el_arch = df_sto.loc[~df_sto.pp.str.contains('HYD|CH0|HT')].pp_id.tolist()
        list_sto_el_arch = df_sto.loc[~df_sto.pp.str.contains('HYD|CH0|AT|DE|FR|IT')].pp_id.tolist()

        list_sto_pho_arch = list_pho_arch + list_sto_el_arch
                
        list_hp_arch = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('HP_AW')].pp_id.tolist()
        list_ht_sto_arch = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('STO_HT')].pp_id.tolist()
        list_ht_pp_arch = list_hp_arch + list_ht_sto_arch
        
        list_all_pp_arch = list_sto_pho_arch + list_ht_pp_arch
        list_all_add = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.set_def_add == 1].pp_id.tolist()
        
        # nd_arch = self.ml.m.df_def_node.loc[np.invert(self.ml.m.df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0','IND_H2','TRANSP_H2','TRANSP_EL']))].nd_id.tolist()
        nd_nat = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('CH0|AT0|IT0|FR0|DE0')].nd_id.tolist()
        nd_arch = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH|OCO|IND')].nd_id.tolist()
        nd_arch_cl = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('CL')].nd_id.tolist()
        nd_arch_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('HT')].nd_id.tolist()
        nd_arch_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH')].nd_id.tolist()
        nd_arch_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('DSR')].nd_id.tolist()
        nd_arch_ev_res = self.ml.m.df_def_node.loc[(self.ml.m.df_def_node.nd.str.contains('EV')) & (self.ml.m.df_def_node.nd.str.contains('SFH|MFH'))].nd_id.tolist()
        nd_arch_ev_ind_oco = self.ml.m.df_def_node.loc[(self.ml.m.df_def_node.nd.str.contains('EV')) & (self.ml.m.df_def_node.nd.str.contains('IND|OCO'))].nd_id.tolist()
        nd_public_charging = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('Public_Charging')].nd_id.tolist()
        nd_arch_el = list(set(nd_arch) - set(nd_arch_cl) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev_res) - set(nd_arch_ev_ind_oco))
        nd_arch_el_res = list(set(nd_arch_res) - set(nd_arch_cl) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev_res) - set(nd_arch_ev_ind_oco))
        nd_arch_el_non_res = list(set(nd_arch_el) - set(nd_arch_el_res))
        nd_nat = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0'])].nd_id.tolist()
        nd_all_el_non_res = nd_nat + nd_arch_el_non_res + nd_public_charging
        

        if nd_arch_el:    
            tm_id_el = self.ml.m.dict_nd_tm_id[nd_arch_el[0]]
        else:
            pass
            # tm_id_el = 0
        if nd_arch_ht:    
            tm_id_ht = self.ml.m.dict_nd_tm_id[nd_arch_ht[0]]
        else:
            pass        
            # tm_id_ht = 1 # Here it's juast to avoid some error on some test when not selecting heating nodes
        if nd_arch_dsr:    
            tm_id_dsr = self.ml.m.dict_nd_tm_id[nd_arch_dsr[0]]
        else:
            pass
            # tm_id_dsr = 1        
        if nd_public_charging:    
            tm_id_public_charging = self.ml.m.dict_nd_tm_id[nd_public_charging[0]]
        else:
            pass
        if nd_arch_ev_res:    
            tm_id_ev = self.ml.m.dict_nd_tm_id[nd_arch_ev_res[0]]
        else:
            pass
        if nd_arch_cl:    
            tm_id_cl = self.ml.m.dict_nd_tm_id[nd_arch_cl[0]]
        else:
            pass

        if dict_fy is None:
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }


        rng_fy = np.arange(len(dict_fy))
        rng_fyp = np.ones(len(rng_fy)) * {val: key for key, val in dict_fy.items()}[2015]
        dict_previous = dict(np.array([rng_fy, rng_fyp]).T)


        slct_fy = self.ml.dct_step['swfy']

        str_fy = '_yr' + str(dict_fy[slct_fy]) if dict_fy[slct_fy] != 2015 else ''
        str_fyp = '_yr' + str(dict_fy[dict_previous[slct_fy]]) if dict_fy[dict_previous[slct_fy]] != 2015 else ''

        #######################################################################
        def set_fuel_prices(str_fy=None):
            ''' Select fuel price values for selected year. '''

            if not str_fy:
                str_fy = ''

            slct_col = 'vc_fl' + str_fy
            msg = ('Setting vc_fl to values' + str_fy.replace('_', ' ') +
                   ' from column {}'.format(slct_col))
            logger.info(msg)

            par = self.ml.m.dict_par['vc_fl']
            df_new = self.ml.m.df_fuel_node_encar[['fl_id', 'nd_id', slct_col]]
            df_new = df_new.rename(columns={slct_col: 'vc_fl'})
            col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'
            par.init_update(df_new, col_mt_fact)



        #######################################################################
        def set_cap_pwr_leg(slct_ch, str_fy=None, excl_pp=[]):
            ''' Select power plant capacities for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'cap_pwr_leg' + str_fy
            msg = ('Setting cap_pwr_leg to values' + str_fy.replace('_', ' ') +
                   ' from column {}'.format(slct_col))
            logger.info(msg)

            mask_excl = -self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
            dct_cap = (self.ml.m.df_plant_encar.loc[mask_excl]
                             .set_index(['pp_id', 'ca_id'])[slct_col]
                             .to_dict())
        
            if slct_ch != 'default':

                df = self.ml.m.df_plant_encar_scenarios.copy()
                df = df.loc[df.scenario == slct_ch]

                dct_cap_scen = (df.set_index(['pp_id', 'ca_id'])[slct_col]
                                  .to_dict())

                dct_cap.update(dct_cap_scen)
                

                
            for kk, vv in dct_cap.items():
                # self.ml.m.cap_pwr_leg[kk] = vv
                self.ml.m.cap_pwr_leg[kk] = float(vv)

        #######################################################################
        def set_cap_avlb(str_fy=None):

            if str_fy == None:
                str_fy = ''

            col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'

            msg = ('Setting cap_avlb monthly adjustment to values'
                   + ' from column {}'.format(col_mt_fact))

            logger.info(msg)

            par = self.ml.m.dict_par['cap_avlb']
            mask_pp = self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.pp)
            df_new = self.ml.m.df_plant_encar.loc[mask_pp,
                                                  ['pp_id', 'ca_id', 'cap_avlb']]

            par.init_update(df_new, col_mt_fact)

            
        #######################################################################
        # def set_dmnd(slct_ev, slct_hp, slct_rf, slct_dpf, slct_hw, str_fy=None):
        def set_dmnd(slct_ev, slct_hp, slct_rf, slct_dpf, slct_cl, str_fy=None):
        
            ''' Scale demand profiles for selected year. '''

            if str_fy == None:
                str_fy = ''
            
            slct_col = 'dmnd_sum' + str_fy
            slct_col_prev = 'dmnd_sum' + str_fyp
            # msg = ('Setting demand normal dmnd_sum to values' + str_fy.replace('_', ' ')
            #       + ' from column {}'.format(slct_col))

            # logger.info(msg)
            last_dmnd = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])[slct_col_prev]
            next_dmnd = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])[slct_col]       
            
            ht_dmnd_2050 = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])['dmnd_sum_yr2050']
          
            scaling_factor = (next_dmnd / last_dmnd).rename('scale')
            scaling_factor_ht = (next_dmnd / ht_dmnd_2050).rename('scale')*0.882151# Climate correction
            scaling_factor_dhw = (next_dmnd / ht_dmnd_2050).rename('scale')# No Climate correction
            
            scaling_factor_ht = (
                scaling_factor_ht
                .groupby(level="nd_id")           # group by nd_id only
                .transform(lambda x: x.fillna(x.mean()))  # fill NaN with mean (or same value)
                )
            
            scaling_factor_dhw = (
                scaling_factor_dhw
                .groupby(level="nd_id")           # group by nd_id only
                .transform(lambda x: x.fillna(x.mean()))  # fill NaN with mean (or same value)
                )
            scaling_factor_cl = scaling_factor_dhw.copy()

            # Inlcude DSR node as not changing (default ee scen)
            # df_el = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & ~(
                                                            # IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res+nd_arch_dsr+nd_arch_ev_res + nd_arch_ev_ind_oco))]
            df_el = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (
                                                            IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_nat+nd_arch_el_non_res))]
            df_el = df_el.join(scaling_factor, on=scaling_factor.index.names)
            df_el['scale'] = df_el['scale'].fillna(0) 
            df_el['value_new'] = df_el.value * df_el.scale
            df_el_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (
                                                            IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res+nd_arch_dsr))]
            
            # RES with no scaling factor and non RES with scaling factor
            dct_dmnd_el = (df_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
            dct_dmnd_el_res = (df_el_res.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
           
            # Non-RES
            for kk, vv in dct_dmnd_el.items():
                self.ml.m.dmnd[kk] = vv
            # RES + DSR 
            for kk, vv in dct_dmnd_el_res.items():
                self.ml.m.dmnd[kk] = vv
             # DHW Boilers (we remove here to reduce the energy demand of DHW_BO throughout the years)                  
            # for kk, vv in dct_dmnd_dhw_bo.items():
            #     self.ml.m.dmnd[kk] = vv
            
            # HEAT
            path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
            path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
            df_def_profile = pd.read_csv(path_prof_id)
            df_profdmnd = pd.read_csv(path_dmnd)  
            
            pf_ca_id_1 = df_def_profile.loc[df_def_profile.pf.str.contains('HT_AW_hourly')]['pf_id'].to_list()
            pf_ca_id_1_daily = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 1]['dmnd_pf_id'].to_list()
            pf_ca_id_2 = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 2]['dmnd_pf_id'].to_list()
            dct_pf_id_dmnd_ht = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([1,2])].set_index('dmnd_pf_id')['nd_id'].to_dict()
            list_pf_id_dmnd_ht = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([1,2])]['dmnd_pf_id'].to_list()
            # dict_tm_soy = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_ht][['sy','doy']].set_index('doy')['sy'].to_dict()
            dict_tm_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_el][['sy','hy']].set_index('hy')['sy'].to_dict()

            # DHW AW 
            pf_ca_id_4 = df_def_profile.loc[df_def_profile.pf.str.contains('DHW_HP_hourly')]['pf_id'].to_list()
            pf_ca_id_4_daily = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 4]['dmnd_pf_id'].to_list()
            pf_ca_id_5 = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 5]['dmnd_pf_id'].to_list()
            dct_pf_id_dmnd_dhw = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin(pf_ca_id_4)].set_index('dmnd_pf_id')['nd_id'].to_dict()
            dct_pf_id_dmnd_dhw_other = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin(pf_ca_id_4_daily)].set_index('dmnd_pf_id')['nd_id'].to_dict()            
            list_pf_id_dmnd_dhw_other = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([4,5])]['dmnd_pf_id'].to_list()
            # dict_tm_soy = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_ht][['sy','doy']].set_index('doy')['sy'].to_dict()
            
            # Hourly water heating: DHW
            #     ### Work for hourly DHW !!! --> Paper 3   
             
            df_def_profile_dhw_hourly = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('DHW_HP_hourly'))]
            list_pf_id_dmnd_dhw_hourly = df_def_profile_dhw_hourly['pf_id'].to_list()                             
            list_nd_name_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_ht)].nd.to_list()
            dct_nd_id_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_ht)].set_index('nd')['nd_id'].to_dict()
            # Convert dictionary into Series mapping
            map_ndname_to_id_dhw = pd.Series(dct_nd_id_ht)
            # Create mapping pf_id -> nd_id using .map()
            dct_pf_id_dmnd_dhw_hourly = (
                        df_def_profile_dhw_hourly
                        .assign(nd_id=df_def_profile_dhw_hourly["primary_nd"].map(map_ndname_to_id_dhw))
                        .loc[df_def_profile_dhw_hourly["primary_nd"].isin(map_ndname_to_id_dhw.index)]
                        .set_index("pf_id")["nd_id"]
                        .to_dict()
                        )
            df_dhw = df_profdmnd.loc[df_profdmnd.dmnd_pf_id.isin(list_pf_id_dmnd_dhw_hourly)].copy()
            df_dhw = df_profdmnd.loc[df_profdmnd.dmnd_pf_id.isin(dct_pf_id_dmnd_dhw_hourly)].copy()
            df_dhw['nd_id'] = df_dhw['dmnd_pf_id'].map(dct_pf_id_dmnd_dhw_hourly)                 
            df_dhw['sy'] = df_dhw['hy'].map(dict_tm_hour)                
            df_dhw['ca_id'] = 0
            df_dhw.loc[df_dhw.dmnd_pf_id.isin(list_pf_id_dmnd_dhw_hourly),'ca_id'] = 4.0                
            df_dhw = df_dhw.loc[~df_dhw.sy.isna()]
                                              
            # ready profiles for DHW hourly:                
            df_dhw = df_dhw.join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)              
            df_dhw['value_new'] = df_dhw.value * df_dhw.scale                                                   
            dct_dmnd_dhw = (df_dhw.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                                     
            for kk, vv in dct_dmnd_dhw.items():
                self.ml.m.dmnd[kk] = vv   
                            
            df_def_profile_cooling_hourly = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('cooling_SFH|cooling_MFH'))]
            list_pf_id_dmnd_cooling_hourly = df_def_profile_cooling_hourly['pf_id'].to_list()                             
            list_nd_name_cooling = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_cl)].nd.to_list()
            dct_nd_id_cooling = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_cl)].set_index('nd')['nd_id'].to_dict()
            # Convert dictionary into Series mapping
            map_ndname_to_id_cooling = pd.Series(dct_nd_id_cooling)
            # Create mapping pf_id -> nd_id using .map()
            dct_pf_id_dmnd_cooling_hourly = (
                        df_def_profile_cooling_hourly
                        .assign(nd_id=df_def_profile_cooling_hourly["primary_nd"].map(map_ndname_to_id_cooling))
                        .loc[df_def_profile_cooling_hourly["primary_nd"].isin(map_ndname_to_id_cooling.index)]
                        .set_index("pf_id")["nd_id"]
                        .to_dict()
                        )
            df_cooling = df_profdmnd.loc[df_profdmnd.dmnd_pf_id.isin(list_pf_id_dmnd_cooling_hourly)].copy()
            df_cooling = df_profdmnd.loc[df_profdmnd.dmnd_pf_id.isin(dct_pf_id_dmnd_cooling_hourly)].copy()
            df_cooling['nd_id'] = df_cooling['dmnd_pf_id'].map(dct_pf_id_dmnd_cooling_hourly)                 
            df_cooling['sy'] = df_cooling['hy'].map(dict_tm_hour)                
            df_cooling['ca_id'] = 0
            df_cooling.loc[df_cooling.dmnd_pf_id.isin(list_pf_id_dmnd_cooling_hourly),'ca_id'] = 0.0                
            df_cooling = df_cooling.loc[~df_cooling.sy.isna()]
                                              
            # ready profiles for cooling hourly:                
            df_cooling = df_cooling.join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)                                          
            # df_cooling['value_new'] = df_cooling.value * df_cooling.scale                        
            
            if slct_cl == 'low':
                
                if slct_rf == 'retr_1pc':
                    df_cooling['value_new'] = df_cooling.erg_tot_retr_1pc * df_cooling.scale 
                    dct_dmnd_cooling = (df_cooling.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                    for kk, vv in dct_dmnd_cooling.items():
                        self.ml.m.dmnd[kk] = vv * 0.0 
                        
                elif slct_rf == 'retr_2pc':
                    df_cooling['value_new'] = df_cooling.erg_tot_retr_2pc * df_cooling.scale 
                    dct_dmnd_cooling = (df_cooling.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                    for kk, vv in dct_dmnd_cooling.items():
                        self.ml.m.dmnd[kk] = vv * 0.0  
            
            if slct_cl == 'mid':
                
                if slct_rf == 'retr_1pc':
                    df_cooling['value_new'] = df_cooling.erg_tot_retr_1pc * df_cooling.scale 
                    dct_dmnd_cooling = (df_cooling.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                    for kk, vv in dct_dmnd_cooling.items():
                        self.ml.m.dmnd[kk] = vv * 0.5 
                        
                elif slct_rf == 'retr_2pc':
                    df_cooling['value_new'] = df_cooling.erg_tot_retr_2pc * df_cooling.scale 
                    dct_dmnd_cooling = (df_cooling.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                    for kk, vv in dct_dmnd_cooling.items():
                        self.ml.m.dmnd[kk] = vv * 0.5    

            if slct_cl == 'high':
                
                if slct_rf == 'retr_1pc':
                    df_cooling['value_new'] = df_cooling.erg_tot_retr_1pc * df_cooling.scale 
                    dct_dmnd_cooling = (df_cooling.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                    for kk, vv in dct_dmnd_cooling.items():
                        self.ml.m.dmnd[kk] = vv * 1  
                        
                elif slct_rf == 'retr_2pc':
                    df_cooling['value_new'] = df_cooling.erg_tot_retr_2pc * df_cooling.scale 
                    dct_dmnd_cooling = (df_cooling.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict()) 
                    for kk, vv in dct_dmnd_cooling.items():
                        self.ml.m.dmnd[kk] = vv * 1                                                                         
            
            #Set the demand for DHW_BO electric boilers depending from HP scenarios
            #  Decrease DHW boilers demand when full of fossil
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]                                 

            df_dhw_bo = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id == 3]
            df_dhw_bo = df_dhw_bo.join(scaling_factor, on=scaling_factor.index.names)
            df_dhw_bo['scale'] = df_dhw_bo['scale'].fillna(0) 
            df_dhw_bo['value_new'] = df_dhw_bo.value * df_dhw_bo.scale
            
            dct_dmnd_dhw_bo = (df_dhw_bo.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())                     
            
            if slct_hp == 'bau':
                
                dmnd_dhw_2050_pct = 1
                msg_dhw_bo = ('Setting DHW demand supply by boilers for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(round(dmnd_dhw_2050_pct * 100,1)) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                logger.info(msg_dhw_bo)
                for kk, vv in dct_dmnd_dhw_bo.items():
                    self.ml.m.dmnd[kk] = vv

            if slct_hp == 'fossil':
                
                dmnd_dhw_2050_pct = 0.448
                msg_dhw_bo = ('Setting DHW demand supply by boilers for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(round(dmnd_dhw_2050_pct * 100,1)) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)
                        
            if slct_hp == 'full':

                dmnd_dhw_2050_pct = 0.197
                msg_dhw_bo = ('Setting DHW demand supply by boilers for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(round(dmnd_dhw_2050_pct * 100,1)) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv                
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)
                                         
                    
            if slct_rf == 'default':
                
                msg = ('Setting special heating dmnd_sum to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col) + ' for retrofit scenario ' + slct_rf + ' and for HP scenario ' + slct_hp)

                logger.info(msg)
                #HEAT
                df_ht = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id.isin([1,2])]
                df_ht = df_ht.join(scaling_factor_ht, on=scaling_factor_ht.index.names)
                df_ht['value_new'] = df_ht.value * df_ht.scale
    
                dct_dmnd_ht = (df_ht.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_ht_fossil = df_ht_other.copy()
                df_ht_fossil['value_new'] = df_ht_fossil.erg_tot_fossil * df_ht_fossil.scale
                
                df_ht_fossil = df_ht_fossil.loc[~df_ht_fossil.sy.isna()]
                dct_dmnd_ht_fossil = (df_ht_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                # DHW for AW and WW
                df_dhw = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id.isin([4,5])]
                df_dhw = df_dhw.join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)
                df_dhw['value_new'] = df_dhw.value * df_dhw.scale
    
                dct_dmnd_dhw = (df_dhw.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_dhw_fossil = df_dhw_other.copy()
                df_dhw_fossil['value_new'] = df_dhw_fossil.erg_tot_fossil * df_dhw_fossil.scale
                
                df_dhw_fossil = df_dhw_fossil.loc[~df_dhw_fossil.sy.isna()]
                dct_dmnd_dhw_fossil = (df_dhw_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                if slct_hp == 'full':
               
                    for kk, vv in dct_dmnd_ht.items():
                        self.ml.m.dmnd[kk] = vv
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv
                        
                elif slct_hp == 'bau':
                  
                    for kk, vv in dct_dmnd_ht.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                        
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv * 0.568 
                        
                elif slct_hp == 'fossil':
                    
                    for kk, vv in dct_dmnd_ht_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                    
                    for kk, vv in dct_dmnd_dhw_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                        
            elif slct_rf == 'retr_1pc':
                
                msg = ('Setting special heating dmnd_sum to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col) + ' for retrofit scenrio ' + slct_rf + ' and for HP scenario ' + slct_hp)

                logger.info(msg)
                
                # HEAT
                
                # Space heating with default temperature set_point    
                path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
                path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
                df_def_profile = pd.read_csv(path_prof_id)
                df_profdmnd = pd.read_csv(path_dmnd)                 
                df_def_profile_ht_hourly = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('HT_AW_hourly'))]
                list_pf_id_dmnd_ht_hourly = df_def_profile_ht_hourly['pf_id'].to_list()
                
                list_nd_name_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_ht)].nd.to_list()
                dct_nd_id_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_ht)].set_index('nd')['nd_id'].to_dict()
                # Convert dictionary into Series mapping
                map_ndname_to_id = pd.Series(dct_nd_id_ht)
                # Create mapping pf_id -> nd_id using .map()
                # dct_pf_id_dmnd_ht_hourly = df_def_profile_ht_hourly.assign(nd_id=df_def_profile_ht_hourly["primary_nd"].map(map_ndname_to_id)).set_index("pf_id")["nd_id"].to_dict()
                dct_pf_id_dmnd_ht_hourly = df_def_profile_ht_hourly.assign(nd_id=df_def_profile_ht_hourly["primary_nd"].map(map_ndname_to_id)).loc[df_def_profile_ht_hourly["primary_nd"].isin(map_ndname_to_id.index)].set_index("pf_id")["nd_id"].to_dict()
                
                df_ht_retr_1pc = df_profdmnd.loc[df_profdmnd.dmnd_pf_id.isin(dct_pf_id_dmnd_ht_hourly)].copy()           
                df_ht_retr_1pc['nd_id'] = df_ht_retr_1pc['dmnd_pf_id'].map(dct_pf_id_dmnd_ht_hourly)                 
                df_ht_retr_1pc['sy'] = df_ht_retr_1pc['hy'].map(dict_tm_hour)
                # df_ht_retr_1pc['sy'] = df_ht_retr_1pc[df_ht_retr_1pc["hy"].isin(self.ml.m.df_tm_soy_full["hy"])]
                df_ht_retr_1pc['ca_id'] = 0              
                df_ht_retr_1pc = df_ht_retr_1pc.loc[~df_ht_retr_1pc.sy.isna()]
                
                # ready profiles for space heating hourly:
                df_ht_retr_1pc = df_ht_retr_1pc.join(scaling_factor_ht, on=scaling_factor_ht.index.names)
                df_ht_retr_1pc['value_new'] = df_ht_retr_1pc.erg_tot_retr_1pc * df_ht_retr_1pc.scale
                
                dct_dmnd_ht_retr_1pc = (df_ht_retr_1pc.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                                
                df_ht_retr_1pc_fossil = df_ht_retr_1pc.copy()
                df_ht_retr_1pc_fossil['value_new'] = df_ht_retr_1pc_fossil.erg_tot_retr_1pc_fossil * df_ht_retr_1pc_fossil.scale
                
                df_ht_retr_1pc_fossil = df_ht_retr_1pc_fossil.loc[~df_ht_retr_1pc_fossil.sy.isna()]
                dct_dmnd_ht_retr_1pc_fossil = (df_ht_retr_1pc_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                if slct_hp == 'full':
                 
                    for kk, vv in dct_dmnd_ht_retr_1pc.items():
                        self.ml.m.dmnd[kk] = vv

                if slct_hp == 'bau':
                   
                    for kk, vv in dct_dmnd_ht_retr_1pc.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                        
                if slct_hp == 'fossil':
                    
                    for kk, vv in dct_dmnd_ht_retr_1pc_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                        
                    for kk, vv in dct_dmnd_dhw_fossil.items():
                        self.ml.m.dmnd[kk] = vv                                                                                                                                            
                        
            elif slct_rf == 'retr_2pc':
                
                #HEAT
                msg = ('Setting special heating dmnd_sum to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col) + ' for retrofit scenrio ' + slct_rf + ' and for HP scenario ' + slct_hp)

                logger.info(msg)
                
                # HEAT
                
                # Space heating with default temperature set_point    
                path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
                path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
                df_def_profile = pd.read_csv(path_prof_id)
                df_profdmnd = pd.read_csv(path_dmnd)                 
                df_def_profile_ht_hourly = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('HT_AW_hourly'))]
                list_pf_id_dmnd_ht_hourly = df_def_profile_ht_hourly['pf_id'].to_list()
                
                list_nd_name_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_ht)].nd.to_list()
                dct_nd_id_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_ht)].set_index('nd')['nd_id'].to_dict()
                # Convert dictionary into Series mapping
                map_ndname_to_id = pd.Series(dct_nd_id_ht)
                # Create mapping pf_id -> nd_id using .map()
                # dct_pf_id_dmnd_ht_hourly = df_def_profile_ht_hourly.assign(nd_id=df_def_profile_ht_hourly["primary_nd"].map(map_ndname_to_id)).set_index("pf_id")["nd_id"].to_dict()
                dct_pf_id_dmnd_ht_hourly = df_def_profile_ht_hourly.assign(nd_id=df_def_profile_ht_hourly["primary_nd"].map(map_ndname_to_id)).loc[df_def_profile_ht_hourly["primary_nd"].isin(map_ndname_to_id.index)].set_index("pf_id")["nd_id"].to_dict()
                
                df_ht_retr_2pc = df_profdmnd.loc[df_profdmnd.dmnd_pf_id.isin(dct_pf_id_dmnd_ht_hourly)].copy()           
                df_ht_retr_2pc['nd_id'] = df_ht_retr_2pc['dmnd_pf_id'].map(dct_pf_id_dmnd_ht_hourly)                 
                df_ht_retr_2pc['sy'] = df_ht_retr_2pc['hy'].map(dict_tm_hour)
                # df_ht_retr_2pc['sy'] = df_ht_retr_2pc[df_ht_retr_2pc["hy"].isin(self.ml.m.df_tm_soy_full["hy"])]
                df_ht_retr_2pc['ca_id'] = 0                 
                df_ht_retr_2pc = df_ht_retr_2pc.loc[~df_ht_retr_2pc.sy.isna()]
                
                # ready profiles for space heating hourly:
                df_ht_retr_2pc = df_ht_retr_2pc.join(scaling_factor_ht, on=scaling_factor_ht.index.names)
                df_ht_retr_2pc['value_new'] = df_ht_retr_2pc.erg_tot_retr_2pc * df_ht_retr_2pc.scale
                
                dct_dmnd_ht_retr_2pc = (df_ht_retr_2pc.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                
                df_ht_retr_2pc_fossil = df_ht_retr_2pc.copy()
                df_ht_retr_2pc_fossil['value_new'] = df_ht_retr_2pc_fossil.erg_tot_retr_2pc_fossil * df_ht_retr_2pc_fossil.scale
                
                df_ht_retr_2pc_fossil = df_ht_retr_2pc_fossil.loc[~df_ht_retr_2pc_fossil.sy.isna()]
                dct_dmnd_ht_retr_2pc_fossil = (df_ht_retr_2pc_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())                
                
                if slct_hp == 'full':
                 
                    for kk, vv in dct_dmnd_ht_retr_2pc.items():
                        self.ml.m.dmnd[kk] = vv

                elif slct_hp == 'bau':
                   
                    for kk, vv in dct_dmnd_ht_retr_2pc.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil':
                    
                    for kk, vv in dct_dmnd_ht_retr_2pc_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                        
                    for kk, vv in dct_dmnd_dhw_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                                                                                          

                        
            if slct_dpf == 'ee': # Energy efficiency
            
                slct_col = 'dmnd_sum' + str_fy
                slct_col_prev = 'dmnd_sum' + str_fyp
                
                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting energy efficiency curves to residential and DSR for ' + str_fy.replace('_', ' ')
                      + ' from column {}'.format(slct_col) + ' for ee scenario ' + slct_dpf)
    
                logger.info(msg)
                
                # Load before as we need both for EE and DSR
                path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
                path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
                df_def_profile = pd.read_csv(path_prof_id)
                df_dmnd = pd.read_csv(path_dmnd)[['dmnd_pf_id', 'hy', 'value', 'doy']]

                # Energy efficiency
                
                list_nd_name_res_el = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].nd.to_list()
                dct_nd_id_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].set_index('nd')['nd_id'].to_dict()

                df_def_profile_ee = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('diff'))&
                                                              (df_def_profile.primary_nd.isin(list_nd_name_res_el))&
                                                              ~(df_def_profile.pf.str.contains('best'))]
                df_def_profile_ee['nd_id'] = df_def_profile_ee['primary_nd'].map(dct_nd_id_res)
                df_def_profile_ee_2035 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2035_2015')]
                df_def_profile_ee_2050 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2050_2015')]
                
                df_dmnd_ee_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2035 = pd.merge(df_dmnd_ee_2035,df_def_profile_ee_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_ee_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2050 = pd.merge(df_dmnd_ee_2050,df_def_profile_ee_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_el][['sy','hy']].set_index('hy')['sy'].to_dict()
                df_dmnd_ee_2035['sy'] = df_dmnd_ee_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_ee_2050['sy'] = df_dmnd_ee_2050['hy'].map(dict_tm_soy_hour)
                df_el_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res))]
                
                # DSR load under energy efficiency
                
                list_nd_name_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].nd.to_list()
                dct_nd_id_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].set_index('nd')['nd_id'].to_dict()
    
                df_def_profile_dsr = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('DSR'))&
                                                               (df_def_profile.primary_nd.isin(list_nd_name_dsr))&
                                                              ~(df_def_profile.pf.str.contains('best'))]
                df_def_profile_dsr['nd_id'] = df_def_profile_dsr['primary_nd'].map(dct_nd_id_dsr)
                # df_def_profile_dsr_2015 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2015')]
                df_def_profile_dsr_2035 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2035')]
                df_def_profile_dsr_2050 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2050')]
                
                # df_dmnd_dsr_2015 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2015.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                # df_dmnd_dsr_2015 = pd.merge(df_dmnd_dsr_2015,df_def_profile_dsr_2015,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2035 = pd.merge(df_dmnd_dsr_2035,df_def_profile_dsr_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2050 = pd.merge(df_dmnd_dsr_2050,df_def_profile_dsr_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_dsr][['sy','hy']].set_index('hy')['sy'].to_dict()
                # df_dmnd_dsr_2015['sy'] = df_dmnd_dsr_2015['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2035['sy'] = df_dmnd_dsr_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2050['sy'] = df_dmnd_dsr_2050['hy'].map(dict_tm_soy_hour)
                df_el_dsr = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_dsr))]

                if fy_val == 2015:
                    
                    dct_dmnd_res = (df_el_res.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_res.items():
                        self.ml.m.dmnd[kk] = vv
                    dct_dmnd_dsr = (df_el_dsr.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_dsr.items():
                        self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(1/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(1/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                   
                elif fy_val == 2025:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(2/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(2/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(3/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(3/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(4/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(4/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(5/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(5/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(6/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(6/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(7/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(7/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                        
            
            elif slct_dpf == 'best': # BEST Energy efficiency
            
                slct_col = 'dmnd_sum' + str_fy
                slct_col_prev = 'dmnd_sum' + str_fyp
                
                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting energy efficiency curves to residential and DSR for ' + str_fy.replace('_', ' ')
                      + ' from column {}'.format(slct_col) + ' for ee scenario ' + slct_dpf)
    
                logger.info(msg)
                    
                # Load before as we need both for EE and DSR
                path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
                path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
                df_def_profile = pd.read_csv(path_prof_id)
                df_dmnd = pd.read_csv(path_dmnd)[['dmnd_pf_id', 'hy', 'value', 'doy']]
                # Energy efficiency
                
                list_nd_name_res_el = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].nd.to_list()
                dct_nd_id_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].set_index('nd')['nd_id'].to_dict()

                df_def_profile_ee = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('best_2035|2050'))&(df_def_profile.primary_nd.isin(list_nd_name_res_el))]
                df_def_profile_ee['nd_id'] = df_def_profile_ee['primary_nd'].map(dct_nd_id_res)
                df_def_profile_ee_2035 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2035_2015')]
                df_def_profile_ee_2050 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2050_2015')]
                
                df_dmnd_ee_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2035 = pd.merge(df_dmnd_ee_2035,df_def_profile_ee_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_ee_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2050 = pd.merge(df_dmnd_ee_2050,df_def_profile_ee_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_el][['sy','hy']].set_index('hy')['sy'].to_dict()
                df_dmnd_ee_2035['sy'] = df_dmnd_ee_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_ee_2050['sy'] = df_dmnd_ee_2050['hy'].map(dict_tm_soy_hour)
                # df_dmnd_ee_2035 = df_dmnd_ee_2035.loc[~df_dmnd_ee_2035]
                df_el_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res))]

                # DSR load under energy efficiency
                
                list_nd_name_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].nd.to_list()
                dct_nd_id_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].set_index('nd')['nd_id'].to_dict()
    
                df_def_profile_dsr = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('DSR_best_2035|DSR_2050'))&
                                                               (df_def_profile.primary_nd.isin(list_nd_name_dsr))]
                df_def_profile_dsr['nd_id'] = df_def_profile_dsr['primary_nd'].map(dct_nd_id_dsr)
                # df_def_profile_dsr_2015 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2015')]
                df_def_profile_dsr_2035 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2035')]
                df_def_profile_dsr_2050 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2050')]
                
                # df_dmnd_dsr_2015 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2015.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                # df_dmnd_dsr_2015 = pd.merge(df_dmnd_dsr_2015,df_def_profile_dsr_2015,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2035 = pd.merge(df_dmnd_dsr_2035,df_def_profile_dsr_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2050 = pd.merge(df_dmnd_dsr_2050,df_def_profile_dsr_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_dsr][['sy','hy']].set_index('hy')['sy'].to_dict()
                # df_dmnd_dsr_2015['sy'] = df_dmnd_dsr_2015['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2035['sy'] = df_dmnd_dsr_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2050['sy'] = df_dmnd_dsr_2050['hy'].map(dict_tm_soy_hour)
                df_el_dsr = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_dsr))]
                 
                if fy_val == 2015:
                    
                    dct_dmnd_res = (df_el_res.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_res.items():
                        self.ml.m.dmnd[kk] = vv
                    dct_dmnd_dsr = (df_el_dsr.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_dsr.items():
                        self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(1/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(1/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                   
                elif fy_val == 2025:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(2/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(2/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(3/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(3/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(4/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(4/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(5/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(5/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(6/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(6/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(7/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(7/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
           
            if slct_ev == 'basis': # Basis scenario for EV
                            
                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]

                msg = ('Setting Electric car curves to residential+ind_oco+public_charging for ' + str_fy.replace('_', ' ')
                      + ' for EV scenario ' + slct_ev)
    
                logger.info(msg)
                
                
                df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res+nd_arch_ev_ind_oco+nd_public_charging))]
                # df_dmnd_ev_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res))]
                # df_dmnd_ev_ind_oco = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_ind_oco)]
                # df_dmnd_public_charging = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_public_charging)]
                
                if fy_val == 2015:
                    
                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 0)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 5.830284714)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
   
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                   
                elif fy_val == 2025:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 22.1720089)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 55.53867249)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 106.4236385)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 167.8407792)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 233.3600357)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 302.9374627)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
            
            elif slct_ev == 'ss': #100% EV steady growth is foreseen

                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting Electric car curves to residential+ind_oco+public_charging for ' + str_fy.replace('_', ' ')
                      + ' for EV scenario ' + slct_ev)
    
                logger.info(msg)
                
                                
                df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res+nd_arch_ev_ind_oco+nd_public_charging))]
                # df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev))]
                # df_dmnd_ev_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res))]
                # df_dmnd_ev_ind_oco = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_ind_oco)]
                # df_dmnd_public_charging = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_public_charging)]
                
                if fy_val == 2015:
                    
                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 0)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 5.830284714)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
   
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                   
                elif fy_val == 2025:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 25.19629004)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 76.18225986)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 170.9707094)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 299.0384097)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 423.4295553)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 521.9884918)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv               
            
            
            elif slct_ev == 'full': # 100% EV

                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting Electric car curves to residential+ind_oco+public_charging for ' + str_fy.replace('_', ' ')
                      + ' for EV scenario ' + slct_ev)
    
                logger.info(msg)
                
                
                df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res+nd_arch_ev_ind_oco+nd_public_charging))]
                # df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev))]
                # df_dmnd_ev_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res))]
                # df_dmnd_ev_ind_oco = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_ind_oco)]
                # df_dmnd_public_charging = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_public_charging)]
                
                if fy_val == 2015:
                    
                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 0)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 5.830284714)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
   
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                   
                elif fy_val == 2025:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 45.53000509)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 121.7429501)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 234.867546)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 384.9037931)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 571.8516911)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 795.670272)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv               
        
            
            elif slct_ev == 'bass': # 100% EV - based on Bass Model penetration 

    
                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting Electric car curves to residential+ind_oco+public_charging for ' + str_fy.replace('_', ' ')
                      + ' for EV scenario ' + slct_ev ,  fy_val)
    
                logger.info(msg)
                
                
                df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res+nd_arch_ev_ind_oco+nd_public_charging))]
                # df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev))]
                # df_dmnd_ev_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_res))]
                # df_dmnd_ev_ind_oco = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev_ind_oco)]
                # df_dmnd_public_charging = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_public_charging)]               

                if fy_val == 2015:
                    
                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 0)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 5.830284714)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
   
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                   
                elif fy_val == 2025:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 50.93637298)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 299.4442147)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 669.6658585)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 779.0235069)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 793.7439832)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    
                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:

                    df_dmnd_ev = df_dmnd_ev.assign(
                                               value_new = lambda x: x.value * 795.670272)
                    dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_ev.items():
                          self.ml.m.dmnd[kk] = vv  
                                                                         

        #######################################################################
        def set_dhw_capacity(slct_hp, str_fy=None):
            ''' Decrease DHW boilers capacity when full of fossil'''
            
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            list_dhw_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_STO')].pp_id.tolist()
            list_dhw_bo = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_BO')].pp_id.tolist()

            # Decrease capacity of the boilers 
            dct_capdhwbosto = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_bo + list_dhw_sto)]
                                 .set_index(['pp_id', 'ca_id'])['cap_pwr_leg']
                                 .to_dict())

            if slct_hp == 'bau':
                
                dmnd_dhw_2050_pct = 1
                msg_dhw_bo = ('Setting DHW boilers capacity for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(round(dmnd_dhw_2050_pct * 100,1)) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                logger.info(msg_dhw_bo)
                for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
               
            if slct_hp == 'fossil':
                
                dmnd_dhw_2050_pct = 0.448
                msg_dhw_bo = ('Setting DHW boilers capacity for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(round(dmnd_dhw_2050_pct * 100,1)) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)
                        
            if slct_hp == 'full':

                dmnd_dhw_2050_pct = 0.197
                msg_dhw_bo = ('Setting DHW boilers capacity for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(round(dmnd_dhw_2050_pct * 100,1)) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)                        
                        
        #######################################################################        
        def set_v2g_capacity(slct_v2g, str_fy=None):
            ''' Set V2G capacity based on EV scenarios and number of days that V2G will provide'''
            
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            list_v2g = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('V2G')].pp_id.tolist()            
           
            dct_capv2g = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_v2g)]
                                 .set_index(['pp_id', 'ca_id'])['cap_pwr_leg']
                                 .to_dict())
            
            if slct_v2g == 'noV2G':
                if (slct_ev == 'basis') or (slct_ev == 'ss') or (slct_ev == 'full') or (slct_ev == 'bass'):
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 0%' + 'for V2G scenario ' + slct_v2g + "and EV scenario: " + slct_ev)
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2040:
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                            
            if slct_v2g == 'biweekly':
                
                if slct_ev == 'basis':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 7.12%' + 'for V2G scenario ' + slct_v2g + "and EV scenario: " + slct_ev)
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 174.11
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 333.63
                    elif fy_val == 2040:
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 526.16
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 731.56 
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 949.67
               
                elif slct_ev == 'ss':                
                    
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 7.12%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)

                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 238.8228272 
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 535.9739689
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 937.4518238 
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1327.404093 
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1636.375288

                elif slct_ev == 'full':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 7.12%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)  
                      
                    if fy_val == 2015:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2020:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 381.6504732
                    elif fy_val == 2035:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 736.2833745
                    elif fy_val == 2040:
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 1206.630155
                    elif fy_val == 2045:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 1792.690816
                    elif fy_val == 2050:
                            logger.info(msg_v2g)
                            for kk, vv in dct_capv2g.items():
                                self.ml.m.cap_pwr_leg[kk] = vv * 2494.336926
                       
                elif slct_ev == 'bass':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 7.12%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                    
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 938.723977
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2099.327244
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2442.151187
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2488.298227
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2494.336926
                    
                    
            elif slct_v2g == 'weekly':

                if slct_ev == 'basis':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 14.25%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                    
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 348.21
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 667.25
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1052.32
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1463.11
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1899.35

                elif slct_ev == 'ss':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 14.25%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 477.6456544
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1071.947938
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1874.903648
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2654.808185
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 3272.750575

                elif slct_ev == 'full':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 14.25%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                    
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 763.3009464
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1472.566749
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2413.260311
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 3585.381632
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 4988.673852

                elif slct_ev == 'bass':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                                   +  ' to values 14.25%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                    
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1877.447954
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 4198.654487
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 4884.302375
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 4976.596454
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 4988.673852
                                
                        
            elif slct_v2g == '2xweek':
                
                if slct_ev == 'basis':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                               +  ' to values 28.49%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                                   

                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 696.43
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1334.50
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2104.65
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2926.23 
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 3798.70 

                elif slct_ev == 'ss':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                               +  ' to values 28.49%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)

                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 955.2913088 
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2143.895876
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 3749.807295 
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 5309.61637
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 6545.50115        


                elif slct_ev == 'full':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                               +  ' to values 28.49%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)

                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 1526.601893
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 2945.133498
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 4826.520622
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 7170.763265
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 9977.347704
        

                elif slct_ev == 'bass':
                    msg_v2g = ('Setting V2G capacity for' + str_fy.replace('_', ' ')
                               +  ' to values 28.49%' + 'for V2G scenario ' + slct_v2g + ' and EV scenario: ' + slct_ev)
                    if fy_val == 2015:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0 
                    elif fy_val == 2020:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2025:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 0
                    elif fy_val == 2030:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 3754.895908
                    elif fy_val == 2035:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 8397.308974
                    elif fy_val == 2040:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 9768.604749
                    elif fy_val == 2045:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 9953.192909
                    elif fy_val == 2050:
                        logger.info(msg_v2g)
                        for kk, vv in dct_capv2g.items():
                            self.ml.m.cap_pwr_leg[kk] = vv * 9977.347704         

        #######################################################################
        
        def set_co2_price(str_fy=None):
            ''' Select CO2 price for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'price_co2' + str_fy
            msg = ('Setting price_co2  to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col))
            logger.info(msg)

            df_new = self.ml.m.df_def_node[['nd_id', slct_col]]
            par = self.ml.m.dict_par['price_co2']
            col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'
            par.init_update(df_new, col_mt_fact)

        #######################################################################

        def set_erg_inp(slct_ch, str_fy=None, excl_fl=[]):
            ''' Select exogenous energy production for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'erg_inp' + str_fy
            msg = ('Setting erg_inp to values ' + str_fy.replace('_', '')
                   + 'from column {}.'.format(slct_col))
            logger.info(msg)

            ind_col = ['nd_id', 'ca_id', 'fl_id']
            mask_excl = -self.ml.m.df_fuel_node_encar.fl_id.isin(excl_fl)
            dct_erg_inp = (self.ml.m.df_fuel_node_encar.loc[mask_excl]
                               .set_index(ind_col)[slct_col].to_dict())

            # if slct_ch != 'default':

            #     df = self.ml.m.df_fuel_node_encar_scenarios.copy()
            #     df = df.loc[df.scenario == slct_ch]
            #     dct_erg_inp_scen = (df.set_index(ind_col)[slct_col].to_dict())
            #     dct_erg_inp.update(dct_erg_inp_scen)

            for kk, vv in dct_erg_inp.items():
                self.ml.m.erg_inp[kk] = vv
        #######################################################################

        def set_erg_chp(str_fy=None, excl_pp=[]):
            ''' Select exogenous chp energy production for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'erg_chp' +  str_fy
            msg = ('Setting erg_chp to values ' + str_fy.replace('_', '')
                   + ' from column {}.'.format(slct_col))
            logger.info(msg)

            dct_erg_chp = (self.ml.m.df_plant_encar
                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.chp)]
                               .set_index(['pp_id', 'ca_id'])[slct_col]
                               .to_dict())

            for kk, vv in dct_erg_chp.items():
                self.ml.m.erg_chp[kk] = vv

        #######################################################################
        
        def set_fc_cap(str_fy=None, excl_pp=[]):
            ''' Select capital costs for power plants for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'fc_cp_ann' +  str_fy
            msg = ('Setting fc_cp_ann to values ' + str_fy.replace('_', '')
                   + ' from column {}.'.format(slct_col))
            logger.info(msg)

            dct_fc_cp_ann = (self.ml.m.df_plant_encar
                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.add)]
                               .set_index(['pp_id', 'ca_id'])[slct_col]
                               .to_dict())

            for kk, vv in dct_fc_cp_ann.items():
                self.ml.m.fc_cp_ann[kk] = vv

#           To chekc if it is working (*0.01)            
#            dct_fc_cp_ann_sto = (self.ml.m.df_plant_encar
#                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(list_pho_arch)
#                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.add)]
#                               .set_index(['pp_id', 'ca_id'])[slct_col]
#                               .to_dict())
#
#            for kk, vv in dct_fc_cp_ann_sto.items():
#                self.ml.m.fc_cp_ann[kk] = vv * 0.01
        #######################################################################
        
        def set_fc_om(str_fy=None, excl_pp=[]):
            ''' Select O&M costs for power plants for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'fc_om' +  str_fy
            msg = ('Setting fc_om to values ' + str_fy.replace('_', '')
                   + ' from column {}.'.format(slct_col))
            logger.info(msg)

            dct_fc_om = (self.ml.m.df_plant_encar
                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.add)]
                               .set_index(['pp_id', 'ca_id'])[slct_col]
                               .to_dict())

            for kk, vv in dct_fc_om.items():
                self.ml.m.fc_om[kk] = vv
        

         #######################################################################   
        
        def set_hp_capacity(slct_hp, slct_rf, str_fy=None):
            ''' Select heat pump scenario for archetypes'''
            
            if str_fy == None:
                str_fy = ''
                
            if slct_rf == 'default':
                
                slct_col = 'cap_pwr_leg' + str_fy
                slct_col_fossil = 'cap_pwr_leg' + str_fy + '_hp_' + slct_hp
                
                msg = ('Setting cap_pwr_leg to values for HP' + str_fy.replace('_', ' ') +
                       ' from column {}'.format(slct_col) + ' for scenario ' + slct_hp + ' and retrofit ' + slct_rf)
                logger.info(msg)
                
                dct_caphp = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_aw_sto + list_hp_ww)] #list_hp_aw_sto 
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                dct_capdhw = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_aw_sto + list_dhw_ww)] #list_dhw_aw_sto
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                
                if slct_hp == 'full': # full heat pumps in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                 
                elif slct_hp == 'bau': # bau heat pumps deployement in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil': # fossil fuel replacement heat pumps deployement in 2050
                    
                    dct_caphp_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                
                    dct_capdhw_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                    
                    for kk, vv in dct_caphp_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
            
            elif slct_rf == 'retr_1pc':
                
                slct_col = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf
                slct_col_fossil = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf + '_' + slct_hp

                msg = ('Setting cap_pwr_leg to values for HP' + str_fy.replace('_', ' ') +
                       ' from column {}'.format(slct_col) + ' for scenario ' + slct_hp + ' and retrofit ' + slct_rf)
                logger.info(msg)
                
                dct_caphp = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_aw_sto + list_hp_ww)] #list_hp_aw_sto 
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                dct_capdhw = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_aw_sto + list_dhw_ww)] #list_dhw_aw_sto
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                
                if slct_hp == 'full': # full heat pumps in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                 
                elif slct_hp == 'bau': # bau heat pumps deployement in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil': # fossil fuel replacement heat pumps deployement in 2050
                    
                    dct_caphp_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                
                    dct_capdhw_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                    
                    for kk, vv in dct_caphp_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv

           
            elif slct_rf == 'retr_2pc':
                
                slct_col = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf
                slct_col_fossil = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf + '_' + slct_hp

                msg = ('Setting cap_pwr_leg to values for HP' + str_fy.replace('_', ' ') +
                       ' from column {}'.format(slct_col) + ' for scenario ' + slct_hp + ' and retrofit ' + slct_rf)
                logger.info(msg)
                
                dct_caphp = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_aw_sto + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                dct_capdhw = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_aw_sto + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                
                if slct_hp == 'full': # full heat pumps in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                 
                elif slct_hp == 'bau': # bau heat pumps deployement in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil': # fossil fuel replacement heat pumps deployement in 2050
                    
                    dct_caphp_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                
                    dct_capdhw_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                    
                    for kk, vv in dct_caphp_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv


         #######################################################################            
            
        def set_cap_trm_leg(slct_tr, str_fy=None):
            ''' Select transmission capacity scenario for archetypes. '''
            
            # nd_arch = self.ml.m.df_def_node.loc[np.invert(self.ml.m.df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0']))].nd_id.tolist()
            # nd_arch_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('HT')].nd_id.tolist()
            # nd_arch_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH')].nd_id.tolist()
            # nd_arch_el = list(set(nd_arch) - set(nd_arch_ht))
            # nd_arch_el_res = list(set(nd_arch_res) - set(nd_arch_ht))
            # nd_arch_el_non_res = list(set(nd_arch_el) - set(nd_arch_el_res))
            
            # nd_arch = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH|OCO|IND')].nd_id.tolist()
            # nd_arch_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('HT')].nd_id.tolist()
            # nd_arch_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH')].nd_id.tolist()
            # nd_arch_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('DSR')].nd_id.tolist()
            # nd_arch_ev = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('EV')].nd_id.tolist()
            # nd_arch_el = list(set(nd_arch) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev))
            # nd_arch_el_res = list(set(nd_arch_res) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev))
            # nd_arch_el_non_res = list(set(nd_arch_el) - set(nd_arch_el_res))
            # nd_nat = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0'])].nd_id.tolist()
            # nd_all_el_non_res = nd_nat + nd_arch_el_non_res
            
            if str_fy == None:
                str_fy = ''

            slct_col_trme = 'cap_trme_leg' 
            slct_col_trmi = 'cap_trmi_leg' #+ (str_hy if not reset else '')
            msg = ('Setting transmission capacity for archetypes'+ str_fy.replace('_', ' ') +
                    ' for scenario ' + slct_tr)
            logger.info(msg)
            
            dct_cap_trme_leg = (self.ml.m.df_node_connect
                                   .loc[self.ml.m.df_node_connect.nd_2_id.isin(nd_arch_el)]
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col_trme]
                                   .to_dict())
            dct_cap_trmi_leg = (self.ml.m.df_node_connect
                                   .loc[self.ml.m.df_node_connect.nd_2_id.isin(nd_arch_el)]
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col_trmi]
                                   .to_dict())
            
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            if fy_val == 2015:
                
                for kk, vv in dct_cap_trme_leg.items():
                    self.ml.m.cap_trme_leg[kk] = vv
                for kk, vv in dct_cap_trmi_leg.items():
                    self.ml.m.cap_trmi_leg[kk] = vv
                    
            else:
                
                if slct_tr == 'default':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv
                
                elif slct_tr == 'min_5pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*0.95
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*0.95
                
                elif slct_tr == 'plus_5pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*1.05
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*1.05
                
                elif slct_tr == 'plus_10pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*1.1
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*1.1
                        
                elif slct_tr == 'plus_15pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*1.15
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*1.15
                                    
 
        #######################################################################

        ### NTC reduction between neighbouring countries and SWISS electricity system!
        
        def set_cap_trm_leg_ntc(slct_ntc, str_fy=None):
            ''' Select transmission capacity scenario between neighbouring and Swiss (NTC). '''         
            
            if str_fy == None:
                str_fy = ''

            slct_col_trme = 'cap_trme_leg' 
            slct_col_trmi = 'cap_trmi_leg' #+ (str_hy if not reset else '')
            msg = ('Setting transmission capacity for CH-neighbor interconnections '+ str_fy.replace('_', ' ') +
                    ' for scenario ' + slct_ntc)
            logger.info(msg)
            
            # ---------------------------------------------------
            # Select CH <-> neighboring nodes in both directions
            # nd_nat = CH
            # nd_nt  = AT, DE, FR, IT
            # ---------------------------------------------------
            nd_nat = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('CH0')].nd_id.tolist()
            nd_nt = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('AT0|IT0|FR0|DE0')].nd_id.tolist()
            
            mask_ch_nt = (
                (self.ml.m.df_node_connect.nd_id.isin(nd_nat) &
                 self.ml.m.df_node_connect.nd_2_id.isin(nd_nt))
                |
                (self.ml.m.df_node_connect.nd_id.isin(nd_nt) &
                 self.ml.m.df_node_connect.nd_2_id.isin(nd_nat))
                )   
            
            dct_cap_trme_leg = (self.ml.m.df_node_connect
                                   .loc[mask_ch_nt]
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col_trme]
                                   .to_dict())
            dct_cap_trmi_leg = (self.ml.m.df_node_connect
                                   .loc[mask_ch_nt]
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col_trmi]
                                   .to_dict())
            
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            if fy_val in [2015, 2020, 2025]:
                
                for kk, vv in dct_cap_trme_leg.items():
                    self.ml.m.cap_trme_leg[kk] = vv
                for kk, vv in dct_cap_trmi_leg.items():
                    self.ml.m.cap_trmi_leg[kk] = vv                    
                    
            else:
                
                if slct_ntc == 'default':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv
                
                elif slct_ntc == 'min_30pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*0.7
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*0.7
                
                elif slct_ntc == 'min_70pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*0.3
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*0.3                
                                    
 
        #######################################################################

        ### NTC upgrade between neighbouring countries only, excluding CH.
        
        def set_neighbor_ntc_to_artelys(str_fy=None):
            '''Interpolate selected neighbour-neighbour NTCs to Artelys 2035 targets.

            2015 and 2020 keep GRIMSEL input values. 2025 and 2030 move linearly
            from the GRIMSEL value to the Artelys 2035 target. From 2035 onward,
            the Artelys target is applied. CH borders are intentionally excluded.
            '''
            
            if str_fy == None:
                str_fy = ''
            
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            interp_weight_by_year = {
                2015: 0.0,
                2020: 0.0,
                2025: 1.0 / 3.0,
                2030: 2.0 / 3.0,
                2035: 1.0,
                2040: 1.0,
                2045: 1.0,
                2050: 1.0,
            }
            interp_weight = interp_weight_by_year.get(fy_val, 0.0)
            
            msg = ('Setting neighbour-neighbour NTCs to Artelys path ' +
                   str_fy.replace('_', ' ') + ' with interpolation weight ' +
                   str(round(interp_weight, 3)))
            logger.info(msg)
            
            if interp_weight == 0.0:
                return
            
            # Directional Artelys NTC targets for 2035 in MW. These exclude CH.
            artelys_ntc_target_mw = {
                ('DE0', 'FR0'): 4800.0,
                ('DE0', 'AT0'): 7500.0,
                ('AT0', 'DE0'): 7500.0,
                ('AT0', 'IT0'): 1125.0,
                ('FR0', 'DE0'): 4800.0,
                ('FR0', 'IT0'): 4605.0,
                ('IT0', 'AT0'): 945.0,
                ('IT0', 'FR0'): 2310.0,
            }
            
            dct_nd = (self.ml.m.df_def_node
                        .set_index('nd')['nd_id']
                        .to_dict())
            
            def set_directional_cap(from_node, to_node, target_mw):
                from_id = dct_nd[from_node]
                to_id = dct_nd[to_node]
                
                mask_direct = (
                    (self.ml.m.df_node_connect.nd_id == from_id) &
                    (self.ml.m.df_node_connect.nd_2_id == to_id)
                )
                mask_reverse = (
                    (self.ml.m.df_node_connect.nd_id == to_id) &
                    (self.ml.m.df_node_connect.nd_2_id == from_id)
                )
                
                for _, row in self.ml.m.df_node_connect.loc[mask_direct].iterrows():
                    kk = (row.mt_id, row.nd_id, row.nd_2_id, row.ca_id)
                    base_mw = float(row.cap_trme_leg)
                    self.ml.m.cap_trme_leg[kk] = base_mw + interp_weight * (target_mw - base_mw)
                
                for _, row in self.ml.m.df_node_connect.loc[mask_reverse].iterrows():
                    kk = (row.mt_id, row.nd_id, row.nd_2_id, row.ca_id)
                    base_mw = float(row.cap_trmi_leg)
                    self.ml.m.cap_trmi_leg[kk] = base_mw + interp_weight * (target_mw - base_mw)
            
            for (from_node, to_node), target_mw in artelys_ntc_target_mw.items():
                set_directional_cap(from_node, to_node, target_mw)
                                    
 
        #######################################################################

        
         # resetting everything to base year values
        # Note: inflow profiles are static and scaled by erg_inp in the constraint


#        mask_sfh = self.ml.m.df_def_plant.pp.str.contains('SFH')
#        excl_pp = self.ml.m.df_def_plant.loc[mask_sfh].pp_id.tolist()
#        excl_pp = list_sto_pho_arch  # for pv and st opt
        # excl_pp = list_all_add  # for all add
        
        # excl_pp = list_all_add + list_dhw_bo + list_dhw_sto
        excl_pp = list_all_add + list_dhw_bo + list_dhw_sto + list_dhw_aw + list_dhw_aw_sto + list_hp_aw + list_hp_aw_sto
#        excl_pp = list_sto_arch # for stopt
#        excl_pp = list_pho_arch # exclude PHO pp for archetypes not for CH0  # for pvopt
#        excl_fl = []

        set_fuel_prices(str_fy)
        set_cap_pwr_leg(slct_ch, str_fy, excl_pp=excl_pp)
        set_cap_avlb(str_fy)
        # set_dmnd(slct_ev, slct_hp, slct_rf, slct_cl, slct_dpf, str_fy)
        set_dmnd(slct_ev, slct_hp, slct_rf, slct_dpf, slct_cl, str_fy)
        set_co2_price(str_fy)
        set_erg_inp(slct_ch, str_fy)
        set_erg_chp(str_fy)
        set_fc_cap(str_fy)
        set_fc_om(str_fy)
        set_cap_trm_leg(slct_tr, str_fy)
        set_cap_trm_leg_ntc(slct_ntc, str_fy)
        set_neighbor_ntc_to_artelys(str_fy)
#        set_supprof(str_hy)
#        set_priceprof(str_hy)
        set_dhw_capacity(slct_hp, str_fy)
        set_v2g_capacity(slct_v2g, str_fy)
        set_hp_capacity(slct_hp, slct_rf, str_fy)      
        

        self.ml.dct_vl['swfy_vl'] = 'yr' + str(dict_fy[slct_fy])
        
    def keep_cap_new(self, dict_fy=None): 
        ''' Keep the output capacity for power plants in set add for selected year '''
        
#        df_pho = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('_PHO')]
#        list_pho_arch = df_pho.loc[~df_pho.pp.str.contains('SOL')].pp_id.tolist()
#        
#        df_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('_STO')]
#        list_sto_arch = df_sto.loc[~df_sto.pp.str.contains('HYD')].pp_id.tolist()
#        
#        list_sto_pho_arch = list_pho_arch + list_sto_arch
        list_all_add = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.set_def_add == 1].pp_id.tolist()

        
        if dict_fy is None:
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
        
        slct_fy = self.ml.dct_step['swfy']
        fy_val = dict_fy[slct_fy]
#        self.ml.dct_vl['swch_vl'] = slct_ch
            
        if fy_val == 2015:
            
            slct_col = 'cap_pwr_leg'
            msg = ('Setting cap_pwr_leg for PP ADD to values' + str(fy_val) +
                   ' from column {}'.format(slct_col))
            logger.info(msg)

            dct_cap = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_all_add)] # list_sto_arch for stopt
                             .set_index(['pp_id', 'ca_id'])[slct_col]
                             .to_dict())
            
            for kk, vv in dct_cap.items():
                self.ml.m.cap_pwr_leg[kk] = vv
            
        
            for kk, vv in dct_cap.items():
                self.ml.m.cap_pwr_new[kk] = 0
                
            self.ml.m.cap_pwr_new.fix()         
            
        else:
            
            self.ml.m.cap_pwr_new.unfix()
            
            slct_col = 'var_yr_cap_pwr_new'
            msg = ('Setting previous capacity added for PP ADD ' + str(fy_val) +
                   ' from column {}'.format(slct_col))
            logger.info(msg)
        
    
            filt = [(key, [val]) for key, val in self.ml.dct_step.items()
                 if not key == 'swfy'] + [('swfy', [self.ml.dct_step['swfy'] - 1])]
               
            if self.ml.io.modwr.output_target == 'hdf5':
                qry = ' & '.join(str(f[0] + '=' + '"%s"'%f[1][0]) for f in filt)
                slct_run_id = pd.read_hdf(self.ml.io.cl_out, 'def_run',
                                          where=qry,
                                          columns=['run_id']).run_id.tolist()
                df_cpn = pd.read_hdf(self.ml.io.cl_out, 'var_yr_cap_pwr_new',
                                    where='run_id in %s and pp_id in %s'%(str(slct_run_id),str(list_all_add))) #list_sto_arch for stopt
        
                df_cpl = pd.read_hdf(self.ml.io.cl_out, 'par_cap_pwr_leg',
                                    where='run_id in %s and pp_id in %s'%(str(slct_run_id),str(list_all_add)))
                
                dict_cpn_add = (df_cpn.set_index(['pp_id', 'ca_id'])['value'] 
                                + df_cpl.set_index(['pp_id', 'ca_id'])['value']).to_dict()
                
            elif self.ml.io.modwr.output_target == 'psql':
                raise RuntimeError('exogenous_node_ops: Fix nd_weight for psql')
 
            elif self.ml.io.modwr.output_target == 'fastparquet':
                qry = ' & '.join(str(f[0] + '==' + '"%s"'%f[1][0]) for f in filt)
                
                # if mp.active_children():
                if not mp.current_process().name == 'MainProcess':
                    msg = ('MULTIPROCESSING')
                    logger.info(msg)
                    df_def_run = pd.concat([pd.read_csv(fn) for fn in glob(self.ml.io.cl_out + '/def_run*.csv')])
                
                else:
                    msg = ('NO MULTIPROCESSING')
                    logger.info(msg)
                    df_def_run= pd.read_parquet(self.ml.io.cl_out + '/def_run.parq')
                                
                slct_run_id = df_def_run.query(qry)#.to_list()
                assert len(slct_run_id) == 1, 'Found multiple run_ids'
                slct_run_id = slct_run_id.run_id.reset_index(drop=True).get(0)
                
                df_cpn = pd.read_parquet(self.ml.io.cl_out + '/var_yr_cap_pwr_new' + '_{:04d}.parq'.format(slct_run_id))
                df_cpn = df_cpn.loc[df_cpn.pp_id.isin(list_all_add)]
                                
                df_cpl = pd.read_parquet(self.ml.io.cl_out + '/par_cap_pwr_leg' + '_{:04d}.parq'.format(slct_run_id))
                df_cpl = df_cpl.loc[df_cpl.pp_id.isin(list_all_add)]

                
                dict_cpn_add = (df_cpn.set_index(['pp_id', 'ca_id'])['value'] 
                                + df_cpl.set_index(['pp_id', 'ca_id'])['value']).to_dict()
                
            for kk, vv in dict_cpn_add.items():
                self.ml.m.cap_pwr_leg[kk] = vv                                                       
